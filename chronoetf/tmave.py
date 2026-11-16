"""T-MAVE 的下行证据写入、创新读取与配对回放。"""

from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from backtest import (
    BASE_COST, BOOTSTRAP_REPS, BOOTSTRAP_SEED, CODES, PER_YEAR, build_targets,
    check_engine, drawdown, load_market, metrics, simulate,
)
from risk import (
    EVENT_QUANTILE, HORIZON, LONG_HALF_LIFE, SHORT_HALF_LIFE, TEST_YEAR, VALIDATION_YEAR,
    VAR_FLOOR, fit_readout, fit_risk, qlike, readout_inputs,
)


REPLAY_END = pd.Timestamp("2026-10-30")
FREEZE_DATE = pd.Timestamp("2026-11-15")
MIN_FRESH_SESSIONS = 126
FEES = (5, 10, 25)
SHORT_WRITE = 0.50
LONG_WRITE = 0.18
GUARD_SCALE = 0.75
CDAR_GAIN = 0.15
MAX_CAGR_LOSS = 0.01
REPLAY_EXPOSURE_TOL = 0.02
STRATEGIES = {
    "ewma": "动量 + EWMA",
    "fixed": "双记忆固定写入",
    "old_old": "现有记忆 × 波动目标",
    "old_new": "现有记忆 × 创新读取",
    "new_old": "T-MAVE 写入 × 波动目标",
    "new_new": "T-MAVE 写入 × 创新读取",
    "impulse": "仅冲击幅度 × 创新读取",
    "downside": "T-MAVE × 下行半方差",
}
INNOVATION_ARMS = ("old_new", "new_new", "impulse", "downside")
REGIMES = ("急跌反弹", "持续下跌", "震荡反复")


@dataclass
class MemoryStudy:
    forecasts: dict
    downside: pd.DataFrame
    state: dict
    evidence: dict
    folds: pd.DataFrame


def anchored_states(candidate, fast_gate, anchor_gate):
    """先用旧锚点推进短期状态，再按当天证据执行有界写入。"""
    shape = candidate.shape
    results = np.empty((*shape, 2))
    short = np.zeros(shape[1])
    long = np.zeros(shape[1])
    fast_decay = np.exp2(-1 / SHORT_HALF_LIFE)
    slow_decay = np.exp2(-1 / LONG_HALF_LIFE)
    for i, value in enumerate(candidate):
        short_prior = long + fast_decay * (short - long)
        long_prior = slow_decay * long
        short = short_prior + fast_gate[i] * (value - short_prior)
        long = long_prior + anchor_gate[i] * (value - long_prior)
        results[i, :, 0] = short
        results[i, :, 1] = long
    return results


def fit_memory(market, original):
    dates = market.close.index
    returns = original.log_returns
    squared = returns.pow(2)
    downside_squared = returns.clip(upper=0).pow(2)
    downside = downside_squared.rolling(HORIZON).mean().shift(-HORIZON)
    anchor = original.forecasts["rolling"]
    fast_variance = squared.rolling(20, min_periods=18).mean().clip(lower=VAR_FLOOR)
    volume = market.volume
    rel_volume = np.log(volume / volume.rolling(20, min_periods=18).median().shift(1))
    common = readout_inputs(fast_variance, anchor, np.log(squared / anchor + 0.05).clip(-3, 3), rel_volume)
    downside_z = -returns / np.sqrt(anchor.shift(1))
    candidate = np.log1p(downside_z.clip(lower=0).pow(2)).clip(0, 3).fillna(0).to_numpy()
    forecasts = {
        key: pd.DataFrame(np.nan, index=dates, columns=CODES)
        for key in ("tmave", "impulse", "semi")
    }
    forecasts["semi_ewma"] = downside_squared.ewm(
        alpha=1 - original.ewma_lambda, adjust=False, min_periods=20, ignore_na=True,
    ).mean().clip(lower=VAR_FLOOR)
    states = {key: np.full((*returns.shape, 2), np.nan) for key in ("tmave", "impulse")}
    evidence = {
        key: np.full(returns.shape, np.nan)
        for key in ("downside_z", "downside_shock", "recent_shocks", "equity_sync", "long_confirmed",
                    "short_gate", "long_gate", "impulse_long_gate", "threshold")
    }
    fold_rows = []
    anchor_values = anchor.to_numpy()
    realized = original.realized.to_numpy()
    downside_values = downside.to_numpy()
    for fold in original.folds.itertuples():
        start = dates.searchsorted(pd.Timestamp(fold.oos_first))
        end = dates.searchsorted(pd.Timestamp(fold.oos_last), side="right")
        eligible = np.arange(end) + HORIZON < start
        z = downside_z.iloc[:end]
        threshold = float(np.nanquantile(z.to_numpy()[eligible], EVENT_QUANTILE))
        shocks = z.ge(threshold)
        recent = shocks.rolling(HORIZON, min_periods=1).sum().to_numpy()
        hit = shocks.to_numpy()
        equity_sync = np.zeros_like(hit)
        equity_sync[:, :3] = (hit[:, :3].sum(axis=1) >= 2)[:, None]
        confirmed = hit & ((recent >= 2) | equity_sync)
        short = SHORT_WRITE * hit
        long = LONG_WRITE * confirmed
        # 冲击强度对照与完整模型使用相同的短写入和折内长期写入总量。
        impulse_rate = long[eligible].sum() / hit[eligible].sum()
        impulse = impulse_rate * hit
        original_range = slice(start - int(fold.year == VALIDATION_YEAR), end)
        for key, gate in (("tmave", long), ("impulse", impulse)):
            history = anchored_states(candidate[:end], short, gate)
            states[key][original_range] = history[original_range]
            features = np.concatenate((common[:end], history), axis=2)
            value, count = fit_readout(features, anchor_values[:end], realized[:end], eligible, original_range)
            forecasts[key].iloc[original_range] = value
            if key == "tmave":
                semi, _ = fit_readout(features, anchor_values[:end], downside_values[:end], eligible, original_range)
                forecasts["semi"].iloc[original_range] = semi
        for key, values in (
            ("downside_z", z.to_numpy()), ("downside_shock", hit), ("recent_shocks", recent),
            ("equity_sync", equity_sync), ("long_confirmed", confirmed), ("short_gate", short),
            ("long_gate", long), ("impulse_long_gate", impulse), ("threshold", np.full_like(z, threshold, dtype=float)),
        ):
            evidence[key][original_range] = values[original_range]
        fold_rows.append({
            "year": fold.year, "last_training_signal": fold.train_signal_max,
            "last_training_label": fold.train_label_max, "first_prediction": fold.oos_first,
            "training_rows": count, "downside_threshold": threshold, "impulse_long_write": impulse_rate,
        })
    return MemoryStudy(forecasts, downside, states, evidence, pd.DataFrame(fold_rows))


def period_ranges(dates, freeze_date=FREEZE_DATE):
    validation = dates.searchsorted(pd.Timestamp(VALIDATION_YEAR, 1, 1))
    replay = dates.searchsorted(pd.Timestamp(TEST_YEAR, 1, 1))
    replay_end = dates.searchsorted(REPLAY_END, side="right")
    fresh = dates.searchsorted(freeze_date, side="right")
    periods = {"development": (validation, replay), "replay": (replay, replay_end)}
    if len(dates) - fresh >= MIN_FRESH_SESSIONS:
        periods["independent"] = (fresh, len(dates))
    return periods, len(dates) - fresh


def weekly_signals(dates, start, end):
    weeks = dates.to_period("W-SUN")
    transitions = np.flatnonzero(weeks[1:] != weeks[:-1]) + 1
    orders = np.r_[start, transitions[(transitions > start) & (transitions < end)]]
    return orders - 1


def innovation_overlay(raw, basket, memory, dates, periods):
    selected = basket > 0
    contrast = memory[:, :, 0] - memory[:, :, 1]
    score = np.sum(np.where(selected, contrast * basket, 0), axis=1)
    train = weekly_signals(dates, *periods["development"])
    enter = max(0.02, float(np.nanquantile(score[train], 0.80)))
    exit_ = min(enter * 0.75, float(np.nanquantile(score[train], 0.45)))
    schedule = np.unique(np.concatenate([weekly_signals(dates, *p) for p in periods.values()]))
    guard = False
    timeline = np.zeros(len(dates), dtype=bool)
    action = np.full(len(dates), "", dtype="<U8")
    restart = {start - 1 for start, _ in periods.values()}
    for i in schedule:
        if i in restart:
            guard = False
        before = guard
        if np.isfinite(score[i]):
            guard = score[i] > exit_ if guard else score[i] >= enter
        timeline[i] = guard
        action[i] = "reduce" if guard and not before else "restore" if before and not guard else "hold"
    applied = raw * np.where(timeline, GUARD_SCALE, 1.0)[:, None]
    return applied, {
        "innovation": score, "guarded": timeline, "controller_action": action,
        "entry_threshold": enter, "exit_threshold": exit_,
    }


def book_exposures(path):
    equity = path[[f"w_{code}" for code in CODES[:3]]].sum(axis=1).mean()
    return np.array([equity, path["cash_weight"].mean()])


def scale_targets(raw, has_equity, multipliers):
    adjusted = raw * np.where(has_equity, multipliers[0], multipliers[1])[:, None]
    return adjusted / np.maximum(1, adjusted.sum(axis=1))[:, None]


def targets_and_controls(market, original, memory, periods):
    forecast_group = {**original.forecasts, **memory.forecasts, "semi_equivalent": memory.forecasts["semi"] * 2}
    risk = replace(original, forecasts=forecast_group)
    raw = build_targets(market, risk, {
        "ewma": "ewma", "fixed": "fixed", "old_old": "chrono",
        "new_old": "tmave", "impulse": "impulse", "downside": "semi_equivalent",
    })
    basket = raw.pop("momentum")
    has_equity = basket[:, :3].sum(axis=1) > 0
    dates = market.close.index
    target_reference = book_exposures(simulate(market, raw["old_old"], "old_old", *periods["development"], BASE_COST))
    raw["old_new"] = raw["old_old"]
    raw["new_new"] = raw["new_old"]
    info = {}
    for key, states in (
        ("old_new", original.dynamic_states), ("new_new", memory.state["tmave"]),
        ("impulse", memory.state["impulse"]), ("downside", memory.state["tmave"]),
    ):
        raw[key], info[key] = innovation_overlay(raw[key], basket, states, dates, periods)

    calibrated = {}
    records = []
    for key in STRATEGIES:
        signal = raw[key]

        def residual(parameters):
            matched = scale_targets(signal, has_equity, parameters)
            path = simulate(market, matched, key, *periods["development"], BASE_COST)
            return book_exposures(path) - target_reference

        multipliers = np.ones(2)
        if key != "old_old":
            result = least_squares(residual, multipliers, bounds=(0.5, 2), diff_step=0.002, gtol=1e-8)
            multipliers = result.x
        calibrated[key] = scale_targets(signal, has_equity, multipliers)
        equity_diff, cash_diff = residual(multipliers)
        details = info.get(key, {})
        records.append({
            "strategy": key, "equity_basket_scale": multipliers[0], "other_basket_scale": multipliers[1],
            "development_equity_diff": equity_diff, "development_cash_diff": cash_diff,
            "entry_threshold": details.get("entry_threshold"), "exit_threshold": details.get("exit_threshold"),
        })
    return calibrated, info, pd.DataFrame(records)


def cdar(returns):
    """把最深 5% 的日回撤记为正值，包含首日之前的初始净值。"""
    path = np.cumprod(1 + np.asarray(returns), axis=-1)
    depth = 1 - path / np.maximum.accumulate(np.maximum(path, 1), axis=-1)
    n = depth.shape[-1]
    k = math.ceil(n * .05)
    return np.partition(depth, n - k, axis=-1)[..., -k:].mean(axis=-1)


def summarize(path):
    values = metrics(path)
    values["cdar95"] = float(cdar(path["net_return"]))
    values["mean_cash_weight"] = path["cash_weight"].mean()
    return values


def evaluate(market, targets, controls, periods):
    rows = []
    saved = {}
    for stage, (start, end) in periods.items():
        for key, target in targets.items():
            for bps in FEES:
                path = simulate(market, target, key, start, end, bps)
                label = {"strategy": key, "phase": stage, "one_way_bps": bps}
                rows.append({**label, "year": "all", **summarize(path)})
                if bps != BASE_COST:
                    continue
                path["drawdown"] = drawdown(path["net_return"])
                path["guarded"] = False
                path["innovation"] = np.nan
                path["controller_action"] = ""
                if key in controls:
                    signals = market.close.index.get_indexer(path["signal_date"].ffill())
                    path["guarded"] = controls[key]["guarded"][signals]
                    path["innovation"] = controls[key]["innovation"][signals]
                    path["controller_action"] = np.where(
                        path["scheduled"], controls[key]["controller_action"][signals], "",
                    )
                saved[(stage, key)] = path
                for year, annual in path.groupby(path.index.year):
                    rows.append({**label, "year": str(year), **summarize(annual)})
    check_engine(saved)
    return pd.DataFrame(rows), saved


def scenario_windows(returns, periods):
    equity = returns.iloc[:, :3].mean(axis=1, skipna=False)
    prior = equity.rolling(60, min_periods=55).std().shift(1)
    values = equity.to_numpy()
    start, end = periods["replay"]
    next_start = start
    matches = []
    candidates = 0
    for t in range(start, end - 10):
        if not equity.iloc[t] <= -1.5 * prior.iloc[t]:
            continue
        candidates += 1
        if t < next_start or not np.isfinite(values[t:t + 11]).all():
            continue
        after = values[t + 1:t + 11]
        scale = prior.iloc[t]
        turns = np.sum(np.sign(after[:-1]) * np.sign(after[1:]) < 0)
        if after[:5].sum() + values[t] >= 0:
            label = REGIMES[0]
        elif after.sum() <= -scale and (after < 0).sum() >= 6:
            label = REGIMES[1]
        elif turns >= 4 and abs(after.sum()) <= 1.5 * scale:
            label = REGIMES[2]
        else:
            continue
        matches.append({
            "start": equity.index[t], "end": equity.index[t + 10], "scenario": label,
            "equity_shock": float(np.expm1(values[t])), "trailing_daily_vol": scale,
            "equity_forward_5d": float(np.expm1(after[:5].sum())), "equity_forward_10d": float(np.expm1(after.sum())),
        })
        next_start = t + 11
    return matches, candidates


def scenario_results(windows, paths, strategies=STRATEGIES):
    results = []
    for window in windows:
        for key in strategies:
            segment = paths[("replay", key)].loc[window["start"]:window["end"]]
            values = segment["net_return"].to_numpy()
            growth = np.cumprod(1 + values)
            regained = np.flatnonzero(growth >= 1)
            results.append({
                **window, "strategy": key,
                "net_return": growth[-1] - 1,
                "max_drawdown": drawdown(segment["net_return"]).min(),
                "rebounds_5d": (1 + values[1:6]).prod() - 1,
                "days_to_recover_pre_shock_nav": int(regained[0]) if len(regained) else np.nan,
                "recovered": bool(len(regained)),
                "turnover": segment["turnover"].sum(),
                "mean_cash": segment["cash_weight"].mean(),
                "guarded_days": int(segment["guarded"].sum()),
            })
    return pd.DataFrame(results)


def bootstrap(paths):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    comparisons = (
        ("new_new", "old_old"), ("new_new", "new_old"), ("new_new", "old_new"),
        ("new_new", "impulse"), ("new_new", "ewma"), ("new_new", "fixed"),
    )

    def cagr(r):
        return np.expm1(PER_YEAR * np.log1p(r).mean(axis=-1))

    for phase in ("replay", "independent"):
        if (phase, "new_new") not in paths:
            continue
        returns = {key: paths[(phase, key)]["net_return"].to_numpy() for key in STRATEGIES}
        length = len(returns["new_new"])
        for block in (10, 20, 40):
            starts = rng.integers(length, size=(BOOTSTRAP_REPS, math.ceil(length / block)))
            idx = ((starts[:, :, None] + np.arange(block)) % length).reshape(BOOTSTRAP_REPS, -1)[:, :length]
            for left, right in comparisons:
                a, b = returns[left], returns[right]
                boot_a, boot_b = a[idx], b[idx]
                for measure, estimate, values in (
                    ("relative_cdar_gain", 1 - cdar(a) / cdar(b),
                     1 - cdar(boot_a) / np.maximum(cdar(boot_b), VAR_FLOOR)),
                    ("cagr_difference", cagr(a) - cagr(b), cagr(boot_a) - cagr(boot_b)),
                ):
                    lower, upper = np.quantile(values, (.025, .975))
                    rows.append({"phase": phase, "comparison": right, "block": block, "measure": measure,
                                 "estimate": estimate, "lower": lower, "upper": upper})
    return pd.DataFrame(rows)


def forecast_results(original, memory, periods):
    rows = []
    forecasters = {
        **{k: ("variance", original.forecasts[k]) for k in ("ewma", "fixed", "chrono")},
        **{k: ("variance", memory.forecasts[k]) for k in ("tmave", "impulse")},
        **{k: ("downside", memory.forecasts[k]) for k in ("semi_ewma", "semi")},
    }
    for period, (start, end) in periods.items():
        for target, labels in (("variance", original.realized), ("downside", memory.downside)):
            scores = slice(start, end - HORIZON)
            actual = labels.iloc[scores]
            predictions = {key: value.iloc[scores] for key, (kind, value) in forecasters.items() if kind == target}
            common = actual.notna()
            for value in predictions.values():
                common &= value.notna()
            actual = actual.where(common)
            for key, value in predictions.items():
                aligned = value.where(common)
                loss = qlike(actual, aligned) if target == "variance" else (actual - aligned).abs()
                rows.append({"phase": period, "target": target, "model": key, "asset_days": int(common.sum().sum()),
                             "loss": float(loss.stack().mean())})
    return pd.DataFrame(rows)


def risk_detail(market, original, memory, periods):
    dates = market.close.index
    actual, downside = original.realized.copy(), memory.downside.copy()
    for _, end in periods.values():
        actual.iloc[end - HORIZON:end] = np.nan
        downside.iloc[end - HORIZON:end] = np.nan
    fields = {
        "realized_var": actual, "realized_semi": downside,
        **{key: original.forecasts[key] for key in ("ewma", "fixed", "chrono")},
        **memory.forecasts, "old_short": original.dynamic_states[:, :, 0], "old_long": original.dynamic_states[:, :, 1],
        "new_short": memory.state["tmave"][:, :, 0], "new_long": memory.state["tmave"][:, :, 1],
        "impulse_short": memory.state["impulse"][:, :, 0], "impulse_long": memory.state["impulse"][:, :, 1],
        **memory.evidence,
    }
    masks = np.zeros(len(dates), dtype=object)
    for name, (start, end) in periods.items():
        masks[start:end] = name
    active = masks != 0
    index = pd.MultiIndex.from_product((dates[active], CODES), names=["date", "symbol"])
    return pd.DataFrame(
        {"phase": np.repeat(masks[active], len(CODES)),
         **{key: np.asarray(values)[active].ravel() for key, values in fields.items()}}, index=index,
    )


def check_mechanism(memory, new_paths):
    assert (memory.folds["last_training_label"] < memory.folds["first_prediction"]).all()
    evidence = memory.evidence
    hit = evidence["downside_shock"] == 1
    confirmed = hit & ((evidence["recent_shocks"] >= 2) | (evidence["equity_sync"] == 1))
    np.testing.assert_array_equal(evidence["long_gate"] > 0, confirmed)
    assert (evidence["short_gate"][hit] == SHORT_WRITE).all()
    assert not np.any(evidence["equity_sync"][:, 3:] == 1)
    assert np.nanmin(memory.state["tmave"]) >= 0 and np.nanmax(memory.state["tmave"]) <= 3
    # 孤立冲击不改变长期锚点。下一日的短期状态按固定半衰期衰减。
    state = anchored_states(np.ones((2, 1)), np.array([[SHORT_WRITE], [0]]), np.zeros((2, 1)))
    np.testing.assert_allclose(state[:, 0], [[SHORT_WRITE, 0], [SHORT_WRITE * np.exp2(-1 / SHORT_HALF_LIFE), 0]])
    for (stage, key), path in new_paths.items():
        if key in INNOVATION_ARMS:
            assert not path.loc[~path["scheduled"], "controller_action"].ne("").any()


def run_study(folder, now):
    check_engine()
    market, _, hashes = load_market()
    original = fit_risk(market.close, market.volume)
    memory = fit_memory(market, original)
    periods, fresh_sessions = period_ranges(market.close.index)
    targets, controls, calibration = targets_and_controls(market, original, memory, periods)
    performance, paths = evaluate(market, targets, controls, periods)
    check_mechanism(memory, paths)
    windows, candidates = scenario_windows(original.log_returns, periods)
    scenarios = scenario_results(windows, paths)
    intervals = bootstrap(paths)
    forecast = forecast_results(original, memory, periods)
    folder.mkdir(parents=True, exist_ok=False)
    for name, data, include_index in (
        ("portfolio_metrics", performance, False),
        ("daily_backtest", pd.concat(paths, names=["phase", "strategy"]), True),
        ("risk_states", risk_detail(market, original, memory, periods), True),
        ("scenarios", scenarios, False),
    ):
        data.to_csv(folder / f"{name}.csv", index=include_index, encoding="utf-8-sig", float_format="%.12g")
    from report import write_tmave_report

    write_tmave_report(folder, now, performance, paths, scenarios, candidates, calibration, forecast,
                       intervals, memory.folds, fresh_sessions, hashes)
    print(performance.query("phase=='replay' and year=='all' and one_way_bps==5")[
        ["strategy", "cdar95", "cagr", "max_drawdown", "sharpe", "mean_equity_weight", "mean_cash_weight"]
    ].to_string(index=False))
    print(f"T-MAVE 研报与图表：{folder}")
