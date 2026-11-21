"""完整 T-MAVE 对 EWMA 已兑现误差的残差校准；所有筛选只用历史向前预测。"""
import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backtest import PER_YEAR, CODES, check_engine, load_market, simulate
from risk import VAR_FLOOR, qlike
from sleeves import group_returns, build_factors, evaluate_sleeves
from strategy import (allocations, base_portfolio, flow_jump, time_flow, MODES, circular_blocks,
                      bootstrap_difference, performance_stats, check_clocks, check_allocation)
from tmave import period_ranges, cdar
from horizon import holm_adjust
from volatility_policy import holding_schedule, risk_inputs, calibrate_goal, risk_multiplier, tail_stats

FREEZE = "2026-11-16"
FORMATION_END = pd.Timestamp("2018-01-01")
READOUTS = ("static", "fixed", "trade", "natural", "event", "full", "price",
            "joint_static", "joint_fixed", "joint_trade", "joint_natural", "joint_event", "joint_full")
PRIMARY = ("ewma", "calibrated", "joint_static_paired", "joint_fixed_paired", "joint_trade_paired")
ECONOMIC = ("ewma", "calibrated", "joint_static_paired", "joint_fixed_paired", "joint_full_paired")
GAMMAS = np.array([0., .25, .5, .75, 1.])


def error_states(market):
    dates = market.close.index
    group, _ = group_returns(market.close)
    group.columns = range(3)
    asset = np.log(market.close.iloc[:, :3]).diff()

    def mature_error(returns):
        variance = returns.pow(2).ewm(alpha=.06, adjust=False, min_periods=20, ignore_na=True).mean()
        actual = returns.pow(2).rolling(5, min_periods=5).mean()
        return np.log(actual.clip(lower=VAR_FLOOR) / variance.shift(5).clip(lower=VAR_FLOOR))

    error = mature_error(group)
    asset_error = mature_error(asset)
    history = dates < FORMATION_END
    threshold = error.loc[history].abs().quantile(.90)
    asset_threshold = asset_error.loc[history].abs().quantile(.90)
    observed = error.notna().to_numpy()
    pos, neg = error.ge(threshold), error.le(-threshold)
    consensus = pd.DataFrame(False, index=dates, columns=range(3))
    consensus[0] = ((asset_error.ge(asset_threshold).sum(axis=1).ge(2) & error[0].gt(0))
                    | (asset_error.le(-asset_threshold).sum(axis=1).ge(2) & error[0].lt(0)))
    hits = (pos | neg | consensus) & error.notna()
    direction = np.sign(error.where(hits, 0)).astype(int)
    repeated = ((direction.eq(1) & direction.eq(1).rolling(5, min_periods=1).sum().ge(2))
                | (direction.eq(-1) & direction.eq(-1).rolling(5, min_periods=1).sum().ge(2)))
    confirmed = (hits & (repeated | consensus)).to_numpy()
    other = hits.sum(axis=1).to_numpy()[:, None] - hits.to_numpy()
    ticks = hits.to_numpy() + .5 * confirmed + .25 * other
    fast, slow = .50 * hits.to_numpy(), .18 * confirmed
    event_mean = ticks[history].sum(axis=0) / observed[history].sum(axis=0)
    average_fast = fast[history].sum(axis=0) / observed[history].sum(axis=0)
    average_slow = slow[history].sum(axis=0) / observed[history].sum(axis=0)
    candidate = np.tanh(error.fillna(0).to_numpy() / 2)
    states, elapsed = {}, {}
    for mode in ("trade", "natural", "event", "full", "fixed"):
        natural, events = MODES[mode]
        elapsed[mode] = time_flow(dates, ticks, event_mean, natural, events)
        states[mode] = flow_jump(candidate, elapsed[mode], observed * average_fast if mode == "fixed" else fast,
                                 observed * average_slow if mode == "fixed" else slow)
    fields = {"mature_error": error.to_numpy(), "candidate": candidate, "observed": observed,
              "threshold": np.broadcast_to(threshold, error.shape), "direction": direction.to_numpy(),
              "confirmed": confirmed, "consensus": consensus.to_numpy(), "repeated": repeated.to_numpy(),
              "ticks": ticks, "ticks_lag": np.vstack([np.zeros(3), ticks[:-1]]), "fast_gate": fast, "slow_gate": slow}
    for mode, values in states.items():
        fields[f"fast_{mode}"], fields[f"anchor_{mode}"] = values[:, :, 0], values[:, :, 1]
        fields[f"elapsed_{mode}"] = elapsed[mode]
    detail = pd.DataFrame({k: v.ravel() for k, v in fields.items()},
                          index=pd.MultiIndex.from_product([dates, ["equity", "gold", "bond"]], names=["date", "group"]))
    return states, candidate, detail


def prepare(market, freeze_date=FREEZE):
    dates = market.close.index
    periods, fresh = period_ranges(dates, freeze_date)
    schedule = holding_schedule(dates, periods)
    ids = schedule.signal_row.to_numpy()
    base = base_portfolio(market, allocations(market.close, ids)["budget_60"])
    book = base[ids]
    risk = risk_inputs(market, schedule, book)
    states, candidate, state_detail = error_states(market)
    group_weights = np.column_stack((book[:, :3].sum(axis=1), book[:, 3]))
    group_weights /= group_weights.sum(axis=1, keepdims=True)
    projected = {mode: np.einsum("sg,sgk->sk", group_weights, value[ids, :2]) for mode, value in states.items()}
    original = build_factors(market.close, FORMATION_END).frame
    # Archive both complete state mechanisms, excluding forward labels from state evidence.
    price_evidence = original.drop(columns=["forward_return_5", "forward_downside_5"]).copy()
    price_evidence.index = price_evidence.index.set_names(state_detail.index.names)
    state_detail = state_detail.join(price_evidence.add_prefix("price_"), validate="one_to_one")
    price = np.stack((original.short_full.to_numpy().reshape(-1, 3), original.anchor_full.to_numpy().reshape(-1, 3)), axis=-1)
    projected["price"] = np.einsum("sg,sgk->sk", group_weights, price[ids, :2])
    raw_error = (candidate[ids, :2] * group_weights).sum(axis=1)
    common = np.column_stack((np.log(risk[["var5", "var20", "anchor"]].to_numpy() / risk.ewma.to_numpy()[:, None]),
                              np.log(risk.down20 / risk.ewma + .05), risk.correlation_gap, raw_error))
    features = {"static": common, **{mode: np.column_stack((common, state)) for mode, state in projected.items()}}
    # Match the existing flow-jump neutral candidate on unobserved pressure days; do not interpolate future values.
    pressure = original.pressure_candidate.fillna(0).to_numpy().reshape(-1, 3)[ids, :2]
    joint_common = np.column_stack((common, (pressure * group_weights).sum(axis=1)))
    features["joint_static"] = joint_common
    for mode in ("fixed", "trade", "natural", "event", "full"):
        old = np.stack((original[f"short_{mode}"].to_numpy().reshape(-1, 3),
                        original[f"anchor_{mode}"].to_numpy().reshape(-1, 3)), axis=-1)
        projected_price = np.einsum("sg,sgk->sk", group_weights, old[ids, :2])
        features[f"joint_{mode}"] = np.column_stack((joint_common, projected[mode], projected_price))
    # Regimes are identifiable at signal close, never based on future return signs.
    matrix = state_detail.reset_index()
    under = (matrix.confirmed & matrix.direction.eq(1)).to_numpy().reshape(-1, 3)[:, :2].any(axis=1)
    over = (matrix.confirmed & matrix.direction.eq(-1)).to_numpy().reshape(-1, 3)[:, :2].any(axis=1)
    under = pd.Series(under).rolling(5, min_periods=1).max().to_numpy(dtype=bool)[ids]
    over = pd.Series(over).rolling(5, min_periods=1).max().to_numpy(dtype=bool)[ids]
    schedule["regime"] = np.select([under & over, under, over], ["mixed", "confirmed_under", "confirmed_over"], default="quiet")
    return schedule, risk, features, state_detail, base, periods, fresh


def fit_model(x, offset, actual, train, predict):
    eligible = train & np.isfinite(x).all(axis=1) & np.isfinite(offset) & np.isfinite(actual)
    center, scale = x[eligible].mean(axis=0), x[eligible].std(axis=0)
    scale = np.maximum(scale, 1e-6)
    design = np.column_stack((np.ones(len(x)), np.clip((x - center) / scale, -3, 3)))
    xx = design[eligible]
    yy = np.maximum(actual[eligible], VAR_FLOOR) / offset[eligible]

    def objective(beta):
        linear = xx @ beta
        relative = yy * np.exp(-linear)
        penalty = beta.copy()
        penalty[0] = 0
        return (np.mean(relative + linear) + .05 * (penalty @ penalty),
                xx.T @ (1 - relative) / len(xx) + .10 * penalty)

    result = minimize(objective, np.zeros(xx.shape[1]), jac=True, method="L-BFGS-B",
                      bounds=[(-2, 2)] * xx.shape[1], options={"ftol": 1e-12, "maxiter": 500})
    if not result.success:
        raise RuntimeError(result.message)
    log_correction = design[predict] @ result.x
    return log_correction, result.x, center, scale, int(eligible.sum())


def shrink_choice(actual, calibrated, raw, paired=False):
    delta = np.log(raw / calibrated)
    losses = np.column_stack([qlike(actual, calibrated * np.exp(gamma * delta)) for gamma in GAMMAS])
    mean = losses.mean(axis=0)
    best = int(mean.argmin())
    chunks = [losses[i:i + 4, best].mean() for i in range(0, len(losses) - 3, 4)]
    se = np.std(chunks, ddof=1) / np.sqrt(len(chunks)) if len(chunks) >= 2 else 0.
    if paired:
        errors = losses - losses[:, best, None]
        blocks = np.stack([errors[i:i + 4].mean(axis=0) for i in range(0, len(errors) - 3, 4)])
        paired_se = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
        selected = np.flatnonzero(mean - mean[best] <= paired_se + 1e-14)[0]
        gamma, se = GAMMAS[selected], float(paired_se[selected])
    else:
        gamma = GAMMAS[np.flatnonzero(mean <= mean[best] + se)[0]]
    return gamma, mean, se


def forecast(schedule, risk, features, readouts=READOUTS):
    predictions = pd.DataFrame({"ewma": risk.ewma, "calibrated": np.nan})
    for method in readouts:
        predictions[f"{method}_raw"] = np.nan
        predictions[f"{method}_shrunk"] = np.nan
        predictions[f"{method}_paired"] = np.nan
    folds, coefficients, shrinkage = [], [], []
    eligible_schedule = schedule.execution_date.dt.year.ge(2018)
    for year, section in schedule[eligible_schedule].groupby(schedule.execution_date.dt.year):
        first = int(section.signal_row.iloc[0])
        train = schedule.label_row.lt(first).to_numpy()
        valid = train & np.isfinite(risk.realized) & np.isfinite(risk.ewma)
        calibration = float((risk.realized[valid] / risk.ewma[valid]).mean())
        ix = section.index.to_numpy()
        predictions.loc[ix, "calibrated"] = risk.ewma[ix] * calibration
        validation = (schedule.execution_date.dt.year.ge(max(2018, year - 3)) & schedule.label_row.lt(first)
                      & np.isfinite(predictions.calibrated) & np.isfinite(risk.realized)).to_numpy()
        for method in readouts:
            raw_log, beta, center, scale, n = fit_model(features[method], risk.ewma.to_numpy(), risk.realized.to_numpy(), train, ix)
            raw = predictions.loc[ix, "calibrated"].to_numpy() * np.exp(np.clip(raw_log - np.log(calibration), -np.log(2), np.log(2)))
            predictions.loc[ix, f"{method}_raw"] = raw
            history = validation & np.isfinite(predictions[f"{method}_raw"])
            if history.sum() >= 40:
                gamma, losses, se = shrink_choice(risk.realized[history].to_numpy(), predictions.calibrated[history].to_numpy(), predictions[f"{method}_raw"][history].to_numpy())
            else:
                gamma, losses, se = 0., np.full(5, np.nan), np.nan
            calibrated = predictions.loc[ix, "calibrated"].to_numpy()
            predictions.loc[ix, f"{method}_shrunk"] = calibrated * np.exp(gamma * np.log(raw / calibrated))
            if history.sum() >= 40:
                paired_gamma, _, paired_se = shrink_choice(risk.realized[history].to_numpy(), predictions.calibrated[history].to_numpy(), predictions[f"{method}_raw"][history].to_numpy(), True)
            else:
                paired_gamma, paired_se = 0., np.nan
            predictions.loc[ix, f"{method}_paired"] = calibrated * np.exp(paired_gamma * np.log(raw / calibrated))
            for j, strength in enumerate(GAMMAS):
                shrinkage.append({"year": year, "model": method, "gamma": strength, "validation_qlike": losses[j],
                                  "best_se": se, "selected": bool(strength == gamma), "paired_selected": bool(strength == paired_gamma),
                                  "paired_selected_se": paired_se, "validation_weeks": int(history.sum()),
                                  "last_validation_label": int(schedule.label_row[history].max()) if history.any() else -1})
            folds.append({"year": year, "model": method, "first_signal_row": first,
                          "last_train_label": int(schedule.label_row[valid].max()), "training_weeks": n,
                          "ewma_calibration": calibration, "gamma": gamma, "paired_gamma": paired_gamma})
            for j, value in enumerate(beta):
                coefficients.append({"year": year, "model": method, "feature": j, "beta": value,
                                     "center": center[j - 1] if j else 0, "scale": scale[j - 1] if j else 1})
    return predictions, pd.DataFrame(folds), pd.DataFrame(coefficients), pd.DataFrame(shrinkage)


def prediction_scores(schedule, risk, predictions):
    rows, losses = [], []
    for phase in ("development", "replay", "independent"):
        use = schedule.phase.eq(phase) & schedule.label_in_phase & np.isfinite(risk.realized) & predictions.notna().all(axis=1)
        if not use.any():
            continue
        subgroups = [("all", use), *[(str(y), use & schedule.execution_date.dt.year.eq(y)) for y in schedule.execution_date[use].dt.year.unique()],
                     *[(regime, use & schedule.regime.eq(regime)) for regime in ("quiet", "confirmed_under", "confirmed_over", "mixed")]]
        for name, chosen in subgroups:
            actual = risk.realized[chosen].to_numpy()
            if not len(actual):
                continue
            for model in predictions:
                estimate = predictions.loc[chosen, model].to_numpy()
                ratio = actual / estimate
                rows.append({"phase": phase, "subset": name, "model": model, "weeks": len(actual), "sparse": len(actual) < 30,
                             "qlike": float(qlike(actual, estimate).mean()), "mean_actual_over_forecast": ratio.mean(),
                             "underestimate_twice": (ratio > 2).mean(), "overestimate_twice": (ratio < .5).mean(),
                             "mae_log_ratio": np.abs(np.log(np.maximum(ratio, 1e-12))).mean()})
        for i in schedule.index[use]:
            for model in predictions:
                losses.append({"phase": phase, "signal_date": schedule.signal_date[i], "model": model,
                               "realized": risk.realized[i], "forecast": predictions.loc[i, model],
                               "qlike": qlike(risk.realized[i], predictions.loc[i, model]), "regime": schedule.regime[i]})
    return pd.DataFrame(rows), pd.DataFrame(losses)


def prediction_tests(losses):
    rows = []
    rng = np.random.default_rng(42)
    for phase, section in losses.groupby("phase", sort=False):
        wide = section.pivot(index="signal_date", columns="model", values="qlike")
        a = wide.joint_full_paired.to_numpy()
        for block in (2, 4, 8):
            indices = circular_blocks(rng, len(a), block)
            for comparator in PRIMARY:
                difference = wide[comparator].to_numpy() - a
                rows.append({"phase": phase, "benchmark": comparator, "block_weeks": block,
                             **bootstrap_difference(difference.mean(), difference[indices].mean(axis=1), True)})
    frame = pd.DataFrame(rows)
    for _, part in frame.groupby(["phase", "block_weeks"]):
        frame.loc[part.index, "p_holm"] = holm_adjust(part.p_raw.to_numpy())
    return frame


def books(market, schedule, base, predictions, periods, methods=None, economic=None):
    active = schedule.phase.ne("formation")
    indices = schedule.signal_row[active].to_numpy()
    dev = schedule.loc[active, "phase"].eq("development").to_numpy()
    capital_days = base[indices, :4].sum(axis=1) * schedule.days_held[active].to_numpy()
    targets, rows = {}, []
    methods = methods if methods is not None else ["ewma", "calibrated", "static_paired", "fixed_paired", "full_paired", "price_paired", "full_shrunk",
               "joint_static_paired", "joint_fixed_paired", "joint_trade_paired", "joint_natural_paired",
               "joint_event_paired", "joint_full_paired", "joint_full_raw"]
    economic = ECONOMIC if economic is None else economic

    def target(model, goal):
        result = base.copy()
        scale = risk_multiplier(goal, predictions.loc[active, model].to_numpy())
        result[indices, :4] *= scale[:, None]
        return result

    for level in (75, 90):
        variance = predictions.loc[active, "ewma"].to_numpy()
        goal = calibrate_goal(variance[dev], capital_days[dev], level / 100)
        for model in methods:
            key = f"common_{model}_{level}"
            targets[key] = target(model, goal)
            rows.append({"strategy": key, "goal_daily_sd": goal, "lane": "same_budget"})
        baseline = performance_stats(simulate(market, targets[f"common_ewma_{level}"], "ewma", *periods["development"], 5))["annual_vol"]
        for model in economic:
            if model == "ewma":
                matching_goal = goal
            else:
                low, high = goal * .05, goal * 20
                for _ in range(24):
                    midpoint = (low + high) / 2
                    vol = performance_stats(simulate(market, target(model, midpoint), model, *periods["development"], 5))["annual_vol"]
                    if vol < baseline:
                        low = midpoint
                    else:
                        high = midpoint
                matching_goal = (low + high) / 2
            key = f"matched_{model}_{level}"
            targets[key] = target(model, matching_goal)
            realized = performance_stats(simulate(market, targets[key], model, *periods["development"], 5))["annual_vol"]
            rows.append({"strategy": key, "goal_daily_sd": matching_goal, "lane": "development_vol_matched",
                         "target_development_vol": baseline, "actual_development_vol": realized, "gap": realized - baseline})
    for target_array in targets.values():
        assert np.isfinite(target_array).all() and (target_array >= 0).all() and (target_array.sum(axis=1) <= 1 + 1e-10).all()
        np.testing.assert_array_equal(target_array[:, 4], base[:, 4])
    return targets, pd.DataFrame(rows)


def economic_tests(paths):
    def statistics(returns):
        nav = np.cumprod(1 + returns, axis=-1)
        return {"cagr": np.expm1(252 * np.log1p(returns).mean(axis=-1)),
                "vol": returns.std(axis=-1, ddof=1) * np.sqrt(252), "cdar": cdar(returns),
                "max_drawdown": (nav / np.maximum.accumulate(np.maximum(nav, 1), axis=-1) - 1).min(axis=-1)}
    rng, rows = np.random.default_rng(42), []
    for phase in ("development", "replay"):
        for level in (75, 90):
            a = paths[phase, f"matched_joint_full_paired_{level}"].net_return.to_numpy()
            b = paths[phase, f"matched_ewma_{level}"].net_return.to_numpy()
            sa, sb = statistics(a), statistics(b)
            for block in (10, 20, 40):
                ix = circular_blocks(rng, len(a), block)
                aa, bb = statistics(a[ix]), statistics(b[ix])
                for name in sa:
                    rows.append({"phase": phase, "level": level, "metric": name, "block_days": block,
                                 **bootstrap_difference(sa[name] - sb[name], aa[name] - bb[name], True)})
    result = pd.DataFrame(rows)
    for _, part in result.groupby(["phase", "block_days"]):
        result.loc[part.index, "p_holm"] = holm_adjust(part.p_raw.to_numpy())
    return result


def diagnostic_tables(folder, scores, losses):
    """Post-fit explanation only: these bins and yearly summaries never select a model."""
    annual = scores[scores.subset.astype(str).str.fullmatch(r"\d{4}")].pivot(
        index=["phase", "subset"], columns="model", values="qlike")
    comparisons = []
    for benchmark in PRIMARY:
        for (phase, year), row in annual.iterrows():
            delta = row[benchmark] - row.joint_full_paired
            comparisons.append({"phase": phase, "year": year, "benchmark": benchmark,
                                "qlike_gain": delta, "relative_gain": delta / row[benchmark], "win": delta > 1e-12})
    pd.DataFrame(comparisons).to_csv(folder / "annual_comparisons.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    replay = losses[losses.phase.eq("replay")]
    wide = replay.pivot(index="signal_date", columns="model", values="forecast")
    actual = replay.drop_duplicates("signal_date").set_index("signal_date").realized.reindex(wide.index)
    ratio = wide.joint_full_paired / wide.ewma
    bins = pd.qcut(ratio, 5, labels=False, duplicates="drop")
    rows = []
    for group in sorted(bins.unique()):
        mask = bins.eq(group)
        rows.append({"bin": int(group) + 1, "weeks": int(mask.sum()), "correction_min": ratio[mask].min(),
                     "correction_max": ratio[mask].max(), "mean_correction": ratio[mask].mean(),
                     "mean_actual_over_ewma": (actual[mask] / wide.ewma[mask]).mean(),
                     "mean_calibrated_over_ewma": (wide.calibrated[mask] / wide.ewma[mask]).mean()})
    quantiles = pd.DataFrame(rows)
    quantiles.to_csv(folder / "correction_quantiles.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    return quantiles


def report(folder, now, scores, tests, metrics, paths):
    from report import fonts, returns_plot
    import matplotlib.pyplot as plt
    fonts()
    main = scores.query("phase == 'replay' and subset == 'all'")
    fig, axis = plt.subplots(figsize=(10, 5))
    names = ["ewma", "calibrated", "joint_static_paired", "joint_fixed_paired", "joint_trade_paired", "joint_full_paired", "joint_full_raw"]
    table = main.set_index("model").loc[names]
    selected = table.loc["joint_full_paired"]
    raw_base, cal_base = table.loc["ewma"], table.loc["calibrated"]
    low_counts = tuple(round(row.underestimate_twice * row.weeks) for row in (raw_base, selected))
    high_counts = tuple(round(row.overestimate_twice * row.weeks) for row in (raw_base, selected))
    display = ["原始 EWMA", "平均校准 EWMA", "静态残差", "固定双记忆", "仅交易日双记忆", "完整 T-MAVE＋收缩", "完整 T-MAVE 未收缩（辅助）"]
    axis.barh(display, table.qlike, color=["#80928f"] * 5 + ["#137b70", "#aa573c"])
    axis.invert_yaxis()
    axis.set_xlabel("五日风险 QLIKE（越低越好）")
    axis.set_title("EWMA 残差校准：相同周度标签，历史回放")
    for i, value in enumerate(table.qlike):
        axis.text(value + .005, i, f"{value:.4f}", va="center")
    axis.set_xlim(0, table.qlike.max() * 1.18)
    fig.tight_layout()
    fig.savefig(folder / "forecast_comparison.png", dpi=160)
    plt.close(fig)
    compact = dict(zip(ECONOMIC, ["EWMA", "校准 EWMA", "静态残差", "固定双记忆", "完整 T-MAVE"]))
    labels = {key: key.replace("matched_", "风险匹配 ").replace("common_", "同预算 ") for _, key in paths}
    for level in (75, 90):
        for model in ECONOMIC:
            labels[f"matched_{model}_{level}"] = f"{compact[model]} · {level}"
    quantiles = pd.read_csv(folder / "correction_quantiles.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for field, label in [("mean_correction", "完整 T-MAVE 预测"), ("mean_calibrated_over_ewma", "校准 EWMA 预测"),
                         ("mean_actual_over_ewma", "后续实际平方收益")]:
        ax.plot(quantiles.bin, quantiles[field], marker="o", label=label)
    ax.axhline(1, color="#999", linewidth=.7, linestyle="--")
    ax.set(xlabel="按完整模型 / EWMA 修正倍率等频分组", ylabel="相对当周 EWMA 的均值比",
           title="回放校准诊断：分组只用于解释，不参与选型")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "correction_calibration.png", dpi=160)
    plt.close(fig)
    for level in (75, 90):
        keys = [f"matched_{model}_{level}" for model in ECONOMIC]
        returns_plot(folder, paths, keys, f"matched_{level}.png", f"开发期风险匹配 · {level}",
                     "实际回放风险允许偏离；所有模型单边 5 bp", labels=labels, period="replay")
    for key in labels:
        returns_plot(folder, paths, [key], f"returns_{key}.png", labels[key], "历史回放；零现金利率；单边 5 bp", labels=labels, period="replay")
    links = " · ".join(f'<a href="{p.name}">{p.name}</a>' for p in sorted(folder.glob("*.csv")) + sorted(folder.glob("*.json")))
    economic = metrics.query("phase == 'replay' and year == 'all' and one_way_bps == 5 and strategy.str.startswith('matched_')", engine="python")
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>T-MAVE 已兑现误差研究</title>
<style>body{{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui;color:#172235}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%;height:auto}}a{{color:#126080;overflow-wrap:anywhere}}</style>
<h1>T-MAVE 能否学会 EWMA 的持续误差？</h1><p>{now:%Y-%m-%d %H:%M} 北京时间。2015–2017 形成事件参数，2018 年起向前训练，2021–2023 开发，2024–2026 为已查看历史回放。新独立期为 0 日。</p>
<p><strong>主结果：QLIKE 从 {raw_base.qlike:.4f} 到 {selected.qlike:.4f}，相对原始 / 校准 EWMA 的降幅为 {1-selected.qlike/raw_base.qlike:.1%} / {1-selected.qlike/cal_base.qlike:.1%}。</strong>年度方向另见 annual_comparisons.csv；是否支持机制增量和经济优势，分别看下方配对检验与账户结果，不能从预测点估计直接推出。</p>
<p>第二阶段完整模型同时记住原价格压力与已经兑现的五日高估/低估误差，各含快状态与慢锚点，并保留三时钟、确认门控及先流动后写入。收缩只使用此前三年已结束的向前预测；gamma=0 精确回到校准 EWMA。首轮绝对损失 one-SE 将记忆全部收缩为零，第二阶段改为成对损失 one-SE 并结合两种状态；属于看过首轮后的探索性迭代。</p>
<p>两倍低估次数：原始 EWMA {low_counts[0]}/{int(selected.weeks)}，完整候选 {low_counts[1]}/{int(selected.weeks)}；两倍高估次数分别为 {high_counts[0]}/{int(selected.weeks)}、{high_counts[1]}/{int(selected.weeks)}。绝对对数误差及各时钟消融也一并列出，不以某一种损失改善替代全面检验。</p>
<img src="forecast_comparison.png" alt="主预测比较"><h2>预测结果</h2><div class="scroll">{main.to_html(index=False,float_format=lambda x: f'{x:.4f}')}</div>
<h2>修正倍率与后续风险</h2><p>按已生成预测的修正倍率做事后五分位展示，不是筛选情景、训练规则或独立显著性证据；后续平方收益只是方差的噪声代理。</p><img src="correction_calibration.png" alt="修正倍率分位校准诊断">
<h2>4 周区块主检验</h2><p>正差表示完整收缩模型的 QLIKE 更低。Holm 校正五个对照；全部为探索性检验。</p><div class="scroll">{tests.query("phase == 'replay' and block_weeks == 4").to_html(index=False,float_format=lambda x:f'{x:.5f}')}</div>
<h2>开发期风险匹配后的账户结果</h2><p>匹配常数只在开发期求解；回放期风险仍可能不同，不应解释成精确相同的未来风险。</p><div class="scroll">{economic[['strategy','cagr','annual_vol','max_drawdown','cdar95']].to_html(index=False,float_format=lambda x:f'{x:.4%}')}</div>
<img src="matched_75.png" alt="强档风险匹配"><img src="matched_90.png" alt="温和档风险匹配"><h2>全部证据</h2><p>{links}</p></html>'''
    (folder / "report.html").write_text(body, encoding="utf8")


def run_residual(folder, now):
    check_engine()
    check_clocks()
    check_allocation()
    market, _, hashes = load_market()
    schedule, risk, features, states, base, periods, fresh = prepare(market)
    predictions, folds, coefficients, shrinkage = forecast(schedule, risk, features)
    targets, calibration = books(market, schedule, base, predictions, periods)
    scores, losses = prediction_scores(schedule, risk, predictions)
    tests = prediction_tests(losses)
    metrics, paths = evaluate_sleeves(market, targets, {}, periods)
    metrics = metrics.merge(tail_stats(paths), how="left", on=["phase", "strategy", "year", "one_way_bps"], validate="one_to_one")
    economics = economic_tests(paths)
    dates = market.close.index
    folds["first_signal_date"] = dates[folds.first_signal_row]
    folds["last_train_label_date"] = dates[folds.last_train_label]
    assert folds.last_train_label.lt(folds.first_signal_row).all()
    evidence = pd.concat([schedule, risk, predictions.add_prefix("forecast_")], axis=1)
    # Cross-phase labels remain unavailable for scoring; training is separately constrained by end date.
    evidence.loc[~schedule.label_in_phase, "realized"] = np.nan
    folder.mkdir(parents=True, exist_ok=False)
    tables = {"weekly_forecasts": evidence, "forecast_scores": scores, "forecast_losses": losses,
              "forecast_tests": tests, "train_folds": folds, "coefficients": coefficients, "shrinkage": shrinkage,
              "risk_calibration": calibration, "tmave_error_states": states.reset_index(), "portfolio_metrics": metrics,
              "economic_tests": economics, "daily_backtest": pd.concat(paths, names=["phase", "strategy"]).reset_index()}
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    diagnostic_tables(folder, scores, losses)
    manifest = {"freeze_date": FREEZE, "fresh_sessions": int(fresh), "input_hashes": hashes, "protocol": "RESIDUAL_RESEARCH_PROTOCOL.md",
                "protocol_sha256": sha256((Path(__file__).resolve().parent.parent / "RESIDUAL_RESEARCH_PROTOCOL.md").read_text(encoding="utf8").encode()).hexdigest()}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    report(folder, now, scores, tests, metrics, paths)
    print(scores.query("phase == 'replay' and subset == 'all'")[["model", "qlike", "mean_actual_over_forecast"]].to_string(index=False))
    print(tests.query("phase == 'replay' and block_weeks == 4").to_string(index=False))
    print(f"结果目录 {folder}")
