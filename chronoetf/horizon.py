"""按 T-MAVE 状态调整股、金、债协方差的估计记忆。"""

from dataclasses import replace

import numpy as np
import pandas as pd

from backtest import BASE_COST, BOOTSTRAP_SEED, check_engine, load_market
from sleeves import FREEZE_DATE, NAMES, build_factors, evaluate_sleeves, fixed_session_states
from strategy import (
    allocations, base_portfolio, bootstrap_difference, check_allocation, check_clocks,
    circular_blocks, paired_intervals,
)
from tmave import period_ranges, weekly_signals


PRIMARY = "tmave_full"
FIXED = {"budget_120": 0., "budget_60": 1., "blend_50": .5}
CLOCKS = ("trade", "natural", "event", "full")
LABELS = {
    "budget_120": "固定 120 日",
    "budget_60": "固定 60 日",
    "blend_50": "静态各半混合",
    "vol_ratio": "波动率比自适应",
    "rolling_pressure": "同因子滚动压力",
    "fixed_memory": "同因子固定双记忆",
    "tmave_trade": "T-MAVE · 交易日",
    "tmave_natural": "T-MAVE · 加自然日",
    "tmave_event": "T-MAVE · 加事件时间",
    PRIMARY: "T-MAVE · 完整三时钟",
}
CONTROLS = (*FIXED, "vol_ratio", "rolling_pressure", "fixed_memory")
RISK_METRICS = ("covariance_qlike", "reference_qlike")
MAIN_METRICS = ("covariance_qlike", "net_sharpe")
REFERENCE = np.array([.15, .15, .70])
MATRIX_ENTRIES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def decision_scores(factors):
    def groups(field):
        return factors.frame[field].unstack("sleeve").loc[:, list(NAMES)]

    group_returns = groups("log_return")
    recent = group_returns.rolling(20, min_periods=18).std()
    long = group_returns.rolling(120, min_periods=110).std()
    volatility_ratio = np.log(recent / long).mean(axis=1, skipna=False)

    states = fixed_session_states(factors)
    fast, anchor = states[:, :, 0], states[:, :, 1]
    fixed = np.where(factors.observed, .60 * anchor + .40 * np.maximum(fast - anchor, 0), np.nan)
    scores = pd.DataFrame({
        "vol_ratio": volatility_ratio,
        "rolling_pressure": factors.scores["rolling"].mean(axis=1),
        "fixed_memory": fixed.mean(axis=1),
        **{f"tmave_{clock}": factors.scores[clock].mean(axis=1) for clock in CLOCKS},
    }, index=group_returns.index)
    confirmed = groups("anchor_confirmed").any(axis=1).rolling(5, min_periods=1).max().astype(bool)
    return scores, confirmed


def empirical_weight(known, scores):
    """使用校准分布的中位秩，读数缺失时采取静态各半混合。"""
    known = np.sort(known[np.isfinite(known)])
    left = np.searchsorted(known, scores, side="left")
    right = np.searchsorted(known, scores, side="right")
    weight = (left + right) / (2 * len(known))
    return np.where(np.isfinite(scores), weight, .5)


def second_moment(open_log, start, end, group_weights):
    active = group_weights.sum(axis=1) > 0
    values = open_log[start:end + 1, active]
    if not np.isfinite(values).all():
        return None
    changes = np.diff(values, axis=0) @ group_weights[active]
    return changes.T @ changes / len(changes)


def covariance_losses(forecast, realized):
    covariance_qlike = np.linalg.slogdet(forecast)[1] + np.trace(np.linalg.solve(forecast, realized))
    ratio = (REFERENCE @ realized @ REFERENCE) / (REFERENCE @ forecast @ REFERENCE)
    return covariance_qlike, ratio - np.log(ratio) - 1.


def matrix_fields(tag, matrix):
    return {f"{tag}_{NAMES[i]}_{NAMES[j]}": matrix[i, j] for i, j in MATRIX_ENTRIES}


def build_horizons(market, factors, periods):
    dates = market.close.index
    signals = {phase: weekly_signals(dates, *bounds) for phase, bounds in periods.items()}
    fit = weekly_signals(dates, dates.searchsorted("2016-01-01"), periods["development"][0])
    scores, confirmed = decision_scores(factors)
    weights = {key: np.full(len(dates), value) for key, value in FIXED.items()}
    for key, values in scores.items():
        weights[key] = empirical_weight(values.iloc[fit].to_numpy(), values.to_numpy())
    score_at = scores.to_dict("list")
    books = allocations(market.close, np.unique(np.concatenate(list(signals.values()))))
    mixed = {key: {} for key in LABELS}
    open_log = np.log(market.opening.to_numpy()) * 10_000
    decisions = []
    for phase, rows in signals.items():
        next_open = dict(zip(rows[:-1], rows[1:] + 1))
        for i in rows:
            low, high = books["budget_60"][i], books["budget_120"][i]
            end = next_open.get(i)
            actual = second_moment(open_log, i + 1, end, high.assets_by_group) if end is not None else None
            observation = {
                "phase": phase, "signal_date": dates[i], "execution_date": dates[i + 1],
                "label_end_open": dates[end] if end is not None else pd.NaT,
                "label_sessions": end - i - 1 if end is not None else 0,
                "confirmed_event_5d": confirmed.iloc[i],
                **(matrix_fields("realized_bp2", actual) if actual is not None else {}),
            }
            for key in LABELS:
                weight = weights[key][i]
                cov = weight * low.covariance + (1. - weight) * high.covariance
                mixed[key][i] = replace(high, covariance=cov)
                forecast = cov * 100_000_000
                losses = covariance_losses(forecast, actual) if actual is not None else (np.nan, np.nan)
                decisions.append({
                    **observation, "strategy": key, "weight_60": weight, "nominal_window": 120 - 60 * weight,
                    "decision_score": score_at[key][i] if key in score_at else np.nan,
                    **matrix_fields("forecast_bp2", forecast), **dict(zip(RISK_METRICS, losses)),
                })
    targets = {key: base_portfolio(market, series) for key, series in mixed.items()}
    calibration = {"first": dates[fit[0]], "last": dates[fit[-1]], "signals": len(fit)}
    return targets, pd.DataFrame(decisions), calibration


def summarize_risk(decisions):
    def summary(phase, year, data):
        return {
            "phase": phase, "year": year, "signals": len(data), "scored_weeks": data.covariance_qlike.count(),
            **{metric: data[metric].mean() for metric in RISK_METRICS},
            "mean_weight_60": data.weight_60.mean(), "p10_weight_60": data.weight_60.quantile(.1),
            "p90_weight_60": data.weight_60.quantile(.9), "mean_nominal_window": data.nominal_window.mean(),
        }

    evidence = []
    for (phase, strategy), rows in decisions.groupby(["phase", "strategy"], sort=False):
        evidence.append({"strategy": strategy, **summary(phase, "all", rows)})
        for year, part in rows.groupby(rows.execution_date.dt.year):
            evidence.append({"strategy": strategy, **summary(phase, str(year), part)})
    return pd.DataFrame(evidence)


def split_replay_2026(decisions):
    extra = decisions.loc[decisions.phase.eq("replay") & decisions.execution_date.dt.year.eq(2026)]
    return pd.concat([decisions, extra.assign(phase="replay_2026")], ignore_index=True)


def holm_adjust(p_values):
    values = np.asarray(p_values)
    order = values.argsort()
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(np.arange(len(values), 0, -1) * values[order]))
    return adjusted


def paired_tests(paths, decisions):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    peers = tuple(key for key in LABELS if key != PRIMARY)
    all_paths = dict(paths)
    all_paths.update({("replay_2026", key): paths[("replay", key)].loc[lambda p: p.index.year == 2026] for key in LABELS})
    phases = ("development", "replay", "replay_2026", "independent")
    daily = paired_intervals(all_paths, {"policy": PRIMARY, "core": "budget_120"}, peers, phases,
                            drawdown_interval=True, with_pvalue=True)
    cdar = daily.metric.eq("cdar95")
    daily.loc[cdar, ["difference", "lower", "upper"]] = -daily.loc[cdar, ["difference", "upper", "lower"]].to_numpy()
    daily["unit"] = "sessions"
    daily["block_observations"] = daily["block"]
    records = []
    for phase, rows in split_replay_2026(decisions).groupby("phase", sort=False):
        data = rows.pivot(index="execution_date", columns="strategy", values=list(RISK_METRICS)).dropna()
        for block in (2, 4, 8):
            indices = circular_blocks(rng, len(data), block)
            for metric in RISK_METRICS:
                for key in peers:
                    differences = data[metric][key].to_numpy() - data[metric][PRIMARY].to_numpy()
                    records.append({
                        "phase": phase, "strategy": PRIMARY, "benchmark": key, "block": block * 5,
                        "block_observations": block, "unit": "rebalance_periods", "observations": len(differences),
                        "metric": metric,
                        **bootstrap_difference(differences.mean(), differences[indices].mean(axis=1), with_pvalue=True),
                    })
    result = pd.concat([pd.DataFrame(records), daily], ignore_index=True)
    result["primary"] = result.metric.isin(MAIN_METRICS) & result.benchmark.isin(CONTROLS)
    result["p_holm"] = np.nan
    for _, rows in result.loc[result.primary].groupby(["phase", "block"]):
        result.loc[rows.index, "p_holm"] = holm_adjust(rows.p_raw)
    result["significant_gain"] = result.p_holm.lt(.05) & result.difference.gt(0.)
    result["significant_loss"] = result.p_holm.lt(.05) & result.difference.lt(0.)
    return result


def decision_diagnostics(decisions):
    rows = split_replay_2026(decisions)
    output = []
    for phase, period in rows.groupby("phase", sort=False):
        weights = period.pivot(index="execution_date", columns="strategy", values="weight_60")
        losses = period.pivot(index="execution_date", columns="strategy", values="covariance_qlike")
        confirmed = period.loc[period.strategy.eq(PRIMARY)].set_index("execution_date").confirmed_event_5d
        for key in LABELS:
            data = pd.DataFrame({"weight": weights[key], "short_gain": losses.budget_120 - losses.budget_60,
                                 "gain_vs_blend": losses.blend_50 - losses[key], "confirmed": confirmed}).dropna()
            weight, short_gain = data.weight.to_numpy(), data.short_gain.to_numpy()
            rank = data["weight"].corr(data["short_gain"], method="spearman") if data.weight.nunique() > 1 else np.nan
            high = weight > .5
            preferred_short = short_gain > 0
            output.append({
                "phase": phase, "strategy": key, "scored_weeks": len(data), "weight_gain_rank_corr": rank,
                "short_advantage_share": preferred_short.mean(),
                "direction_match": (high == preferred_short).mean() if data.weight.nunique() > 1 else np.nan,
                "confirmed_weeks": int(data.confirmed.sum()), "calm_weeks": int((~data.confirmed).sum()),
                "weight_confirmed": data.loc[data.confirmed, "weight"].mean(),
                "weight_calm": data.loc[~data.confirmed, "weight"].mean(),
                "gain_vs_blend_confirmed": data.loc[data.confirmed, "gain_vs_blend"].mean(),
                "gain_vs_blend_calm": data.loc[~data.confirmed, "gain_vs_blend"].mean(),
            })
    return pd.DataFrame(output)


def run_horizon(folder, now):
    check_engine()
    check_clocks()
    check_allocation()
    market, _, hashes = load_market()
    periods, fresh = period_ranges(market.close.index, FREEZE_DATE)
    targets, decisions, calibration = build_horizons(market, build_factors(market.close), periods)
    performance, paths = evaluate_sleeves(market, targets, {}, periods)
    risk = summarize_risk(decisions)
    comparisons = paired_tests(paths, decisions)
    diagnostics = decision_diagnostics(decisions)
    folder.mkdir(parents=True, exist_ok=False)
    outputs = {
        "portfolio_metrics": performance, "risk_metrics": risk, "weekly_decisions": decisions,
        "paired_tests": comparisons, "decision_diagnostics": diagnostics,
        "daily_backtest": pd.concat(paths, names=["phase", "strategy"]).reset_index(),
    }
    for name, frame in outputs.items():
        frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    from report import write_horizon_report

    write_horizon_report(folder, now, performance, paths, risk, decisions, comparisons, diagnostics, calibration, fresh, hashes)
    print(performance.query("phase=='replay' and year=='all' and one_way_bps==@BASE_COST")[
        ["strategy", "cagr", "sharpe", "max_drawdown", "cdar95"]
    ].to_string(index=False))
    print(f"结果目录 {folder}")
