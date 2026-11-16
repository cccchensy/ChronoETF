"""因子层与 T-MAVE 分组风控。"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from backtest import BASE_COST, CODES, check_engine, drawdown, load_market
from risk import EVENT_QUANTILE, HORIZON
from strategy import (
    FAST_GATE, GROUPS, MODES, SLOW_GATE, account_path, allocations, annotate_account, base_portfolio,
    check_allocation, check_clocks, covers_objectives, development_pick, flow_jump, paired_intervals,
    performance_stats, time_flow,
)
from tmave import (
    FEES, MIN_FRESH_SESSIONS, period_ranges, scenario_results, scenario_windows, summarize, weekly_signals,
)


FREEZE_DATE = pd.Timestamp("2026-11-16")
FIT_END = pd.Timestamp("2021-01-01")
GUARD_SCALE = .75
EXPOSURE_TOL = .01
CAGR_TOL = .005
NAMES = ("equity", "gold", "bond")
SLEEVES = {"equity": "股票", "gold": "黄金", "bond": "国债"}
GROUP_ARMS = {"tm_equity": ("equity",), "tm_gold": ("gold",), "tm_both": ("equity", "gold")}
LABELS = {
    "core": "120 日风险预算主干",
    "tm_equity": "T-MAVE 只控股票",
    "tm_gold": "T-MAVE 只控黄金",
    "tm_both": "T-MAVE 股金独立控仓",
    "rolling": "同因子 · 滚动阈值",
    "fixed": "同因子 · 固定双记忆",
    "trade": "T-MAVE · 交易日",
    "natural": "T-MAVE · 交易日加自然日",
    "event": "T-MAVE · 交易日加事件",
    "static": "同主干 · 静态敞口匹配",
}


@dataclass
class Factors:
    frame: pd.DataFrame
    scores: dict
    calibration: pd.DataFrame
    observed: np.ndarray


def group_returns(close):
    """股票信号篮子用上一收盘可知的组内权重，金、债各自计算。"""
    simple = close.pct_change(fill_method=None)
    log_returns = np.log(close).diff()
    inverse = log_returns.rolling(60, min_periods=55).std().shift().replace(0, np.nan).rdiv(1)
    weights = inverse.iloc[:, :3].div(inverse.iloc[:, :3].sum(axis=1), axis=0)
    equity = (simple.iloc[:, :3] * weights).sum(axis=1, skipna=False)
    group = pd.DataFrame(dict(equity=equity, gold=simple["518880"], bond=simple["511010"]))
    return np.log1p(group), log_returns / log_returns.rolling(60, min_periods=55).std().shift().replace(0, np.nan)


def build_factors(close):
    dates = close.index
    history = dates < FIT_END
    group, asset_z = group_returns(close)
    volatility = group.rolling(60, min_periods=55).std().replace(0, np.nan)
    prior_vol = volatility.shift()
    momentum = {
        f"momentum_{h}": group.rolling(h, min_periods=math.ceil(.9 * h)).sum() / (volatility * np.sqrt(h))
        for h in (15, 60, 120)
    }
    trend = (momentum["momentum_15"] + momentum["momentum_60"] + momentum["momentum_120"]) / 3
    past_20 = group.rolling(20, min_periods=18).sum()
    negative_squared = group.clip(upper=0).pow(2)
    downside = np.sqrt(negative_squared.rolling(20, min_periods=18).mean())
    relative = downside / prior_vol
    center = relative.loc[history].median()
    z = group / prior_vol
    raw = -.4 * np.tanh(trend / 2) + .35 * np.tanh((relative - center) / .5) - .25 * np.tanh(z / 2)
    observed = raw.notna().to_numpy()

    group_cut = z.loc[history].abs().quantile(EVENT_QUANTILE)
    asset_cut = asset_z.loc[history].abs().quantile(EVENT_QUANTILE)
    stock_z = asset_z.iloc[:, :3]
    positive_votes = stock_z.ge(asset_cut.iloc[:3]).sum(axis=1)
    negative_votes = stock_z.le(-asset_cut.iloc[:3]).sum(axis=1)
    consensus = pd.DataFrame(False, index=dates, columns=NAMES)
    consensus["equity"] = ((positive_votes >= 2) & z["equity"].gt(0)) | ((negative_votes >= 2) & z["equity"].lt(0))
    own = z.abs().ge(group_cut) & raw.notna()
    hit = (own | consensus) & raw.notna()
    direction = np.sign(z.where(hit, 0)).astype(int)
    repeat = (
        (direction.eq(1) & direction.eq(1).rolling(HORIZON, min_periods=1).sum().ge(2))
        | (direction.eq(-1) & direction.eq(-1).rolling(HORIZON, min_periods=1).sum().ge(2))
    )
    confirmed = (hit & (repeat | consensus)).to_numpy()
    others = hit.sum(axis=1).to_numpy()[:, None] - hit.to_numpy()
    ticks = hit.to_numpy() + .50 * confirmed + .25 * others
    fast = FAST_GATE * hit.to_numpy()
    slow = SLOW_GATE * confirmed
    event_mean = ticks[history].sum(axis=0) / observed[history].sum(axis=0)
    constant_fast = fast[history].sum(axis=0) / observed[history].sum(axis=0)
    constant_slow = slow[history].sum(axis=0) / observed[history].sum(axis=0)

    fields = {
        "log_return": group, "volatility_60_lagged": prior_vol, **momentum, "momentum_blend": trend,
        "trend_return_20": past_20, "downside_20": downside, "downside_relative": relative, "shock_z": z,
        "pressure_candidate": raw, "group_shock": own, "event_direction": direction, "repeated": repeat,
        "equity_consensus": consensus & hit, "anchor_confirmed": confirmed,
        "other_groups_active": others, "event_increment": ticks,
        "event_increment_lag": np.vstack((np.zeros(len(NAMES)), ticks[:-1])),
        "fast_gate": fast, "anchor_gate": slow, "fixed_fast_gate": observed * constant_fast,
        "fixed_anchor_gate": observed * constant_slow,
    }
    gap = np.r_[0, np.diff(dates.to_numpy(dtype="datetime64[D]")).astype(int)]
    fields["calendar_days"] = np.broadcast_to(gap[:, None], group.shape)
    fields["trading_days"] = np.broadcast_to(np.r_[0, np.ones(len(dates) - 1)][:, None], group.shape)
    fields["calendar_units"] = fields["calendar_days"] * (5 / 7)
    fields["prior_event_mean"] = np.broadcast_to(event_mean, group.shape)
    fields["group_shock_threshold"] = np.broadcast_to(group_cut.to_numpy(), group.shape)
    fields["event_class"] = np.select(
        [(consensus & hit).to_numpy(), repeat.to_numpy(), hit.to_numpy(), others > 0],
        ["co_move", "repeat", "single", "other_active"], default="quiet",
    )

    candidate = raw.fillna(0).to_numpy()
    scores = {"rolling": raw.rolling(20, min_periods=18).mean().where(raw.notna()).to_numpy()}
    for mode, (include_calendar, include_events) in MODES.items():
        elapsed = time_flow(dates, ticks, event_mean, include_calendar, include_events)
        state = flow_jump(
            candidate, elapsed, observed * constant_fast if mode == "fixed" else fast,
            observed * constant_slow if mode == "fixed" else slow,
        )
        short_state, anchor = state[:, :, 0], state[:, :, 1]
        pressure = .60 * anchor + .40 * np.maximum(short_state - anchor, 0)
        scores[mode] = np.where(observed, pressure, np.nan)
        fields[f"elapsed_{mode}"] = elapsed
        fields[f"short_{mode}"] = short_state
        fields[f"anchor_{mode}"] = anchor
    for key, values in scores.items():
        fields[f"pressure_{key}"] = values
    fields["forward_return_5"] = np.expm1(group.rolling(HORIZON).sum().shift(-HORIZON))
    fields["forward_downside_5"] = np.sqrt(negative_squared.rolling(HORIZON).mean().shift(-HORIZON))
    index = pd.MultiIndex.from_product((dates, NAMES), names=["date", "sleeve"])
    frame = pd.DataFrame({key: np.asarray(values).ravel() for key, values in fields.items()}, index=index)
    calibration = pd.DataFrame({
        "sleeve": NAMES, "downside_median": center.to_numpy(), "shock_quantile": group_cut.to_numpy(),
        "event_mean": event_mean, "fixed_fast_gate": constant_fast, "fixed_anchor_gate": constant_slow,
    })
    for code in CODES[:3]:
        calibration[f"equity_vote_z_{code}"] = asset_cut[code]
    return Factors(frame, scores, calibration, observed)


def fixed_session_states(factors):
    """同一候选与平均写入率下，仅用交易日推进快慢状态。"""
    inputs = factors.frame
    raw = inputs["pressure_candidate"].unstack("sleeve").loc[:, list(NAMES)].fillna(0).to_numpy()
    elapsed = np.ones_like(raw)
    elapsed[0] = 0.
    short = inputs["fixed_fast_gate"].unstack("sleeve").loc[:, list(NAMES)].to_numpy()
    anchor = inputs["fixed_anchor_gate"].unstack("sleeve").loc[:, list(NAMES)].to_numpy()
    return flow_jump(raw, elapsed, short, anchor)


def weekly_limits(scores, dates):
    start, end = dates.searchsorted(pd.Timestamp("2018-01-01")), dates.searchsorted(FIT_END)
    sample = weekly_signals(dates, start, end)
    sample = sample[sample >= start]
    entry = np.maximum(.02, np.nanquantile(scores[sample], .80, axis=0))
    exit_ = np.minimum(entry * .75, np.nanquantile(scores[sample], .45, axis=0))
    return entry, exit_


def guard_states(scores, dates, periods, limits):
    entry, exit_ = limits
    guarded = np.zeros(scores.shape, dtype=bool)
    actions = np.full(scores.shape, "", dtype="<U8")
    for interval in periods.values():
        status = np.zeros(len(NAMES), dtype=bool)
        for i in weekly_signals(dates, *interval):
            previous = status
            next_status = np.where(status, scores[i] > exit_, scores[i] >= entry)
            status = np.where(np.isfinite(scores[i]), next_status, status)
            guarded[i] = status
            actions[i] = np.where(status & ~previous, "reduce", np.where(~status & previous, "restore", "hold"))
    return guarded, actions


def scaled_targets(base, multipliers):
    out = base.copy()
    for j, group in enumerate(GROUPS):
        out[:, group] *= multipliers[:, j, None]
    return out


def book_exposure(path):
    weights = path[[f"w_{symbol}" for symbol in CODES]].to_numpy()
    return np.array([*(weights[:, group].sum(axis=1).mean() for group in GROUPS), path["cash_weight"].mean()])


def make_design(market, factors, periods):
    dates = market.close.index
    signal = np.unique(np.concatenate([weekly_signals(dates, *interval) for interval in periods.values()]))
    books = allocations(market.close, signal)
    base = base_portfolio(market, books["budget_120"])
    full_limits = weekly_limits(factors.scores["full"], dates)
    targets, decisions = {"core": base}, {}

    def add(key, groups, mode):
        scores = factors.scores[mode]
        limits = weekly_limits(scores, dates) if mode in ("rolling", "fixed") else full_limits
        flags, actions = guard_states(scores, dates, periods, limits)
        enabled = np.isin(NAMES, groups)
        flags &= enabled
        actions[:, ~enabled] = ""
        targets[key] = scaled_targets(base, np.where(flags, GUARD_SCALE, 1.))
        decisions[key] = dict(mode=mode, groups=groups, scores=scores, active=flags, actions=actions, limits=limits)

    development = periods["development"]
    for key, groups in GROUP_ARMS.items():
        add(key, groups, "full")
    calibration_paths = {key: account_path(market, v, key, development) for key, v in targets.items()}
    score = {key: summarize(path) for key, path in calibration_paths.items()}
    ref = score["core"]
    joint = [key for key in GROUP_ARMS if covers_objectives(score[key], ref)]
    bounded = [
        key for key in GROUP_ARMS if score[key]["cagr"] >= ref["cagr"] - CAGR_TOL
        and score[key]["cdar95"] < ref["cdar95"] and score[key]["max_drawdown"] > ref["max_drawdown"]
    ]
    selected, frontier = development_pick(score, joint or bounded or GROUP_ARMS)
    selected_groups = GROUP_ARMS[selected]
    for key in ("rolling", "fixed", "trade", "natural", "event"):
        add(key, selected_groups, key)

    active = np.isin(NAMES, selected_groups)
    reference = book_exposure(calibration_paths[selected])

    def exposure_residual(fitted):
        multiplier = np.ones(len(NAMES))
        multiplier[active] = fitted
        scaled = scaled_targets(base, np.broadcast_to(multiplier, (len(dates), len(NAMES))))
        return book_exposure(account_path(market, scaled, "static", development)) - reference

    static_scale = np.ones(len(NAMES))
    fitted = least_squares(
        exposure_residual, np.full(active.sum(), .95), bounds=(GUARD_SCALE, 1), diff_step=.002, gtol=1e-9,
    )
    static_scale[active] = fitted.x
    targets["static"] = scaled_targets(base, np.broadcast_to(static_scale, (len(dates), len(NAMES))))
    selection = pd.DataFrame([
        {
            "strategy": key, **values, "joint_improvement": key in joint, "bounded_improvement": key in bounded,
            "pareto_in_selection_pool": key in frontier, "selected_for_ablation": key == selected,
            "deployment_choice": key == (selected if joint or bounded else "core"),
        }
        for key, values in score.items()
    ])
    for key, target in targets.items():
        groups = GROUP_ARMS[key] if key in GROUP_ARMS else selected_groups if key != "core" else ()
        for group, name in zip(GROUPS, NAMES):
            if name not in groups:
                np.testing.assert_array_equal(target[:, group], base[:, group])
        assert (target >= 0).all() and (target.sum(axis=1) <= 1 + 1e-10).all()
    return targets, decisions, selection, selected, static_scale


def evaluate_sleeves(market, targets, decisions, periods, order_days=None):
    paths, rows = {}, []
    order_days = order_days or {}
    for phase, interval in periods.items():
        for key, target in targets.items():
            for fee in FEES:
                path = account_path(market, target, key, interval, fee, order_days.get(key))
                tag = {"phase": phase, "strategy": key, "one_way_bps": fee}
                rows.append({**tag, "year": "all", **performance_stats(path)})
                for year, section in path.groupby(path.index.year):
                    rows.append({**tag, "year": str(year), **performance_stats(section)})
                if fee != BASE_COST:
                    continue
                annotate_account(market, path)
                path["drawdown"] = drawdown(path["net_return"])
                path["guarded"] = False
                details = decisions.get(key)
                if details is not None:
                    positions = market.close.index.get_indexer(path["signal_date"].ffill())
                    path["guarded"] = details["active"][positions].any(axis=1)
                for j, sleeve in enumerate(NAMES):
                    enabled = details is not None and sleeve in details["groups"]
                    path[f"target_{sleeve}"] = path[[f"target_{symbol}" for symbol in CODES[GROUPS[j]]]].sum(axis=1)
                    path[f"guard_{sleeve}"] = details["active"][positions, j] if enabled else False
                    path[f"pressure_{sleeve}"] = details["scores"][positions, j] if enabled else np.nan
                    path[f"enter_{sleeve}"] = details["limits"][0][j] if enabled else np.nan
                    path[f"exit_{sleeve}"] = details["limits"][1][j] if enabled else np.nan
                    path[f"action_{sleeve}"] = (
                        np.where(path["scheduled"], details["actions"][positions, j], "") if enabled else ""
                    )
                paths[(phase, key)] = path
    check_engine(paths)
    return pd.DataFrame(rows), paths


def factor_detail(factors, dates, periods):
    frame = factors.frame.copy()
    phase = np.full(len(dates), "unscored", dtype="<U12")
    phase[dates < FIT_END] = "formation"
    for key, (start, end) in periods.items():
        phase[start:end] = key
    label_end = dates.to_series().shift(-HORIZON)
    scores = np.zeros(len(dates), dtype=bool)
    for start, end in [(0, dates.searchsorted(FIT_END)), *periods.values()]:
        if end - start > HORIZON:
            scores[start:end - HORIZON] = True
    frame.insert(0, "phase", np.repeat(phase, len(NAMES)))
    frame["label_end"] = np.repeat(label_end.where(scores).to_numpy(), len(NAMES))
    frame.loc[~np.repeat(scores, len(NAMES)), ["forward_return_5", "forward_downside_5"]] = np.nan
    return frame


def factor_diagnostics(frame):
    rows = []
    signals = ("momentum_15", "momentum_60", "momentum_120", "momentum_blend", "downside_20", "shock_z",
               "pressure_candidate")
    data = frame.reset_index()
    for phase, subset in data.loc[~data.phase.isin(("formation", "unscored"))].groupby("phase", sort=False):
        windows = [(phase, subset)]
        if phase == "replay":
            mask = subset.date.dt.year.eq(2026) & subset.label_end.dt.year.eq(2026)
            windows.append(("replay_2026", subset.loc[mask]))
        for label, sample in windows:
            if sample.empty:
                continue
            dates = sample.date.drop_duplicates().to_numpy()[::HORIZON]
            every_five = sample.loc[sample.date.isin(dates)]
            for sleeve, values in every_five.groupby("sleeve", sort=False):
                for signal in signals:
                    fields = [signal, "forward_return_5", "forward_downside_5"]
                    pairs = values[fields].dropna()
                    correlations = pairs.corr(method="spearman")
                    rows.append({
                        "phase": label, "sleeve": sleeve, "factor": signal, "calendar_samples": len(values),
                        "factor_coverage": values[signal].notna().mean(), "paired_samples": len(pairs),
                        "rank_corr_return": correlations.at[signal, "forward_return_5"],
                        "rank_corr_downside": correlations.at[signal, "forward_downside_5"],
                    })
    return pd.DataFrame(rows)


def joint_weakness(factors, paths):
    trends = factors.frame["trend_return_20"].unstack("sleeve")
    previous = trends.shift()
    simultaneous = previous["equity"].lt(0) & previous["gold"].lt(0)
    rows = []
    for (phase, key), path in paths.items():
        if phase != "replay":
            continue
        part = path.loc[path.index.year == 2026]
        initial = part["nav"].iloc[0] / (1 + part["net_return"].iloc[0])
        both = simultaneous.loc[part.index]
        for state, mask in (("joint_weak", both), ("other", ~both)):
            subset = part.loc[mask]
            gross = subset[["pnl_equity", "pnl_gold", "pnl_bond", "pnl_fee"]].sum() / initial
            rows.append({
                "strategy": key, "condition": state, "sessions": len(subset), **gross.to_dict(),
                "return_contribution": gross.sum(), "mean_cash": subset["cash_weight"].mean(),
                "equity_guard_days": subset["guard_equity"].sum(), "gold_guard_days": subset["guard_gold"].sum(),
            })
    return pd.DataFrame(rows)


def guard_episodes(paths):
    records = []
    for (phase, key), path in paths.items():
        for sleeve in ("equity", "gold"):
            start = None
            episode = 0
            for pos, (day, action) in enumerate(path[f"action_{sleeve}"].items()):
                if action == "reduce":
                    start = (pos, day)
                    episode += 1
                elif action == "restore":
                    records.append({
                        "phase": phase, "strategy": key, "sleeve": sleeve, "episode": episode,
                        "reduce_date": start[1], "restore_date": day, "held_sessions": pos - start[0], "closed": True,
                    })
                    start = None
            if start is not None:
                records.append({
                    "phase": phase, "strategy": key, "sleeve": sleeve, "episode": episode,
                    "reduce_date": start[1], "restore_date": pd.NaT, "held_sessions": len(path) - start[0], "closed": False,
                })
    return pd.DataFrame(records)


def verify_transitions(factors):
    dates = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-10", "2020-01-13",
                            "2020-01-17", "2020-01-20", "2020-01-23", "2020-02-03"])
    scores = np.zeros((len(dates), 3))
    scores[0, :2] = [.20, .03]
    scores[2, :2] = [np.nan, .20]
    scores[4, :2] = [.03, .06]
    scores[6, :2] = [.08, .03]
    guarded, actions = guard_states(scores, dates, {"test": (1, 8)}, (np.full(3, .10), np.full(3, .05)))
    np.testing.assert_array_equal(guarded[::2], [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(actions[::2, :2], [["reduce", "hold"], ["hold", "reduce"],
                                                    ["restore", "hold"], ["hold", "restore"]])
    assert factors.frame.loc[(slice(None), ["gold", "bond"]), "equity_consensus"].eq(False).all()
    confirmed = factors.frame["event_direction"].ne(0) & (factors.frame["repeated"] | factors.frame["equity_consensus"])
    np.testing.assert_array_equal(confirmed, factors.frame["anchor_gate"] > 0)
    for mode in MODES:
        states = factors.frame[[f"short_{mode}", f"anchor_{mode}"]].to_numpy()
        assert np.abs(states).max() <= 1


def run_sleeves(folder, now):
    check_engine()
    check_clocks()
    check_allocation()
    market, _, hashes = load_market()
    dates = market.close.index
    periods, fresh = period_ranges(dates, FREEZE_DATE)
    factors = build_factors(market.close)
    verify_transitions(factors)
    targets, decisions, selection, selected, static_scale = make_design(market, factors, periods)
    performance, paths = evaluate_sleeves(market, targets, decisions, periods)
    evidence = factor_detail(factors, dates, periods)
    diagnostics = factor_diagnostics(evidence)
    windows, candidates = scenario_windows(np.log(market.close).diff(), periods)
    scenarios = scenario_results(windows, paths, LABELS)
    joint = joint_weakness(factors, paths)
    episodes = guard_episodes(paths)
    interval_paths = dict(paths)
    interval_paths.update({("replay_2026", key): path.loc[path.index.year == 2026]
                           for (phase, key), path in paths.items() if phase == "replay"})
    intervals = paired_intervals(
        interval_paths, {"policy": selected, "core": "core"}, benchmarks=("core", "static", "rolling", "fixed"),
        phases=("replay", "replay_2026", "independent"), drawdown_interval=True,
    )
    calibration = factors.calibration.copy()
    calibration["static_multiplier"] = static_scale
    for method in ("full", "rolling", "fixed"):
        entry, exit_ = weekly_limits(factors.scores[method], dates)
        calibration[f"{method}_entry"] = entry
        calibration[f"{method}_exit"] = exit_
    folder.mkdir(parents=True, exist_ok=False)
    exports = {
        "portfolio_metrics": performance, "selection": selection, "factor_validation": diagnostics,
        "factor_calibration": calibration, "scenarios": scenarios, "year_2026_attribution": joint,
        "guard_episodes": episodes, "paired_intervals": intervals,
    }
    for name, table in exports.items():
        table.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    pd.concat(paths, names=["phase", "strategy"]).to_csv(
        folder / "daily_backtest.csv", encoding="utf-8-sig", float_format="%.12g",
    )
    evidence.to_csv(folder / "factor_states.csv", encoding="utf-8-sig", float_format="%.12g")
    from report import write_sleeve_report

    write_sleeve_report(folder, now, performance, paths, selection, selected, evidence, diagnostics, calibration,
                        scenarios, candidates, joint, episodes, intervals, fresh, hashes)
    results = performance.query("phase == 'replay' and year == '2026' and one_way_bps == @BASE_COST")
    print(results[["strategy", "total_return", "sharpe", "max_drawdown", "cdar95", "mean_cash_weight"]].to_string(index=False))
    print(f"分组实验：{selected}；结果目录：{folder}")
