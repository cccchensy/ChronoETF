"""用 T-MAVE 读出增强股金之间的相对收益配置。"""

import numpy as np
import pandas as pd

from backtest import BASE_COST, CODES, check_engine, load_market
from strategy import (
    account_path, allocations, base_portfolio, check_clocks, covers_objectives, development_pick,
    paired_intervals, performance_stats,
)
from sleeves import (
    FREEZE_DATE, GUARD_SCALE, build_factors, evaluate_sleeves, guard_states, scaled_targets, weekly_limits,
)
from tmave import period_ranges, scenario_results, scenario_windows, weekly_signals


ALPHA = .10
TILT_LIMIT = .20
LABELS = {
    "core": "120 日风险预算",
    "risk60": "60 日风险预算",
    "guard_fixed": "前轮固定双记忆控仓",
    "trend": "风险调整动量倾斜",
    "ridge": "静态因子岭回归",
    "ridge_fixed": "岭回归 + 固定双记忆",
    "ridge_trade": "岭回归 + T-MAVE 交易日版",
    "ridge_tmave": "岭回归 + T-MAVE 三时钟",
    "ridge_fixed_guard": "固定双记忆配置 + 分组防守",
    "ridge_tmave_guard": "T-MAVE 配置 + 分组防守",
}
MODELS = {"ridge": None, "ridge_fixed": "fixed", "ridge_trade": "trade", "ridge_tmave": "full"}
CANDIDATES = ("ridge_tmave", "ridge_tmave_guard")
BASE_FEATURES = ("momentum_15", "momentum_60", "momentum_120", "downside_relative", "shock_z")


def relative_features(frame):
    equity = frame.xs("equity", level="sleeve")
    gold = frame.xs("gold", level="sleeve")
    columns = [*BASE_FEATURES, *(f"{f}_{mode}" for f in ("short", "anchor") for mode in ("fixed", "trade", "full"))]
    return equity[columns] - gold[columns], equity.log_return - gold.log_return


def forward_spread(market, spread):
    """在信号日拟定组内权重，以两次实际调仓开盘之间的收益作标签。"""
    dates = market.close.index
    start = dates.searchsorted(pd.Timestamp("2016-01-01"))
    orders = weekly_signals(dates, start, len(dates)) + 1
    open_prices = market.opening.to_numpy()
    prices = np.log(market.close)
    ret = prices.diff()
    vol = ret.rolling(60, min_periods=55).std().to_numpy()
    risk = spread.rolling(60, min_periods=55).std().to_numpy()
    rows = []
    for enter, leave in zip(orders[:-1], orders[1:]):
        t = enter - 1
        current_vol = vol[t, :3]
        active = np.isfinite(ret.iloc[t, :3]) & np.isfinite(current_vol) & (current_vol > 0)
        w = 1 / current_vol[active]
        paths = open_prices[leave] / open_prices[enter] - 1
        if not len(w) or not np.isfinite(paths[:3][active]).all() or not np.isfinite(paths[3]):
            continue
        realized = w @ paths[:3][active] / w.sum() - paths[3]
        denominator = risk[t] * np.sqrt(leave - enter)
        if not np.isfinite(denominator) or denominator <= 0:
            continue
        rows.append({
            "signal_date": dates[t], "entry_open": dates[enter], "label_end": dates[leave],
            "realized_spread": realized, "weekly_spread_vol": denominator,
            "risk_normalized_spread": np.clip(realized / denominator, -4, 4),
        })
    return pd.DataFrame(rows).set_index("signal_date")


def ridge_prediction(history, response, new):
    """仅用训练样本的均值和波动标准化，岭惩罚不作用于截距。"""
    center = history.mean(axis=0)
    scale = np.maximum(history.std(axis=0), 1e-8)

    def design(data):
        return np.column_stack((np.ones(len(data)), np.clip((data - center) / scale, -4, 4)))

    x = design(history)
    regularized = np.diag([0, *[ALPHA] * (x.shape[1] - 1)])
    beta = np.linalg.solve(x.T @ x / len(x) + regularized, x.T @ response / len(x))
    return design(new) @ beta, beta


def fit_signals(features, labels, dates, signals):
    all_predictions = pd.DataFrame(np.nan, index=dates, columns=MODELS)
    trainable = labels.join(features, how="inner").dropna()
    groups = pd.Series(dates[signals + 1].year, index=dates[signals])
    folds = []
    for year, members in groups.groupby(groups):
        first = members.index.min()
        past = trainable.loc[trainable.label_end.lt(first)]
        now = features.loc[members.index].dropna()
        for key, mode in MODELS.items():
            cols = list(BASE_FEATURES) + ([f"short_{mode}", f"anchor_{mode}"] if mode else [])
            values, beta = ridge_prediction(
                past[cols].to_numpy(), past.risk_normalized_spread.to_numpy(), now[cols].to_numpy(),
            )
            all_predictions.loc[now.index, key] = values
            folds.append({
                "year": int(year), "model": key, "first_signal": first, "train_last_label": past.label_end.max(),
                "training_weeks": len(past), "valid_signal_weeks": len(now), "intercept": beta[0],
                **{f"coef_{name}": value for name, value in zip(cols, beta[1:])},
            })
    folds = pd.DataFrame(folds)
    assert (folds.train_last_label < folds.first_signal).all()
    return all_predictions, folds


def tilt_stock_gold(base, scores):
    """正预测提高股票组目标，负预测提高黄金组目标。"""
    equity = base[:, :3].sum(axis=1)
    gold = base[:, 3]
    shift = TILT_LIMIT * (equity + gold) * np.tanh(np.nan_to_num(scores, nan=0))
    shift = np.clip(shift, -equity, gold)
    share = np.divide(equity + shift, equity, out=np.ones_like(equity), where=equity > 0)
    result = base.copy()
    result[:, :3] *= share[:, None]
    result[:, 3] -= shift
    np.testing.assert_allclose(result.sum(axis=1), base.sum(axis=1), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(result[:, 4], base[:, 4])
    return result, shift


def design_experiment(market, factors, periods):
    dates = market.close.index
    signal_rows = np.unique(np.concatenate([weekly_signals(dates, *p) for p in periods.values()]))
    books = allocations(market.close, signal_rows)
    core = base_portfolio(market, books["budget_120"])
    targets = {"core": core, "risk60": base_portfolio(market, books["budget_60"])}
    features, spread = relative_features(factors.frame)
    labels = forward_spread(market, spread)
    signals, folds = fit_signals(features, labels, dates, signal_rows)
    signals["trend"] = features[["momentum_15", "momentum_60", "momentum_120"]].mean(axis=1) / 2

    fixed = factors.scores["fixed"]
    limits = weekly_limits(fixed, dates)
    active, actions = guard_states(fixed, dates, periods, limits)
    active[:, 2] = False
    actions[:, 2] = ""
    scale = np.where(active, GUARD_SCALE, 1.)
    targets["guard_fixed"] = scaled_targets(core, scale)
    guard = dict(mode="fixed", groups=("equity", "gold"), scores=fixed, active=active, actions=actions, limits=limits)
    controls = {key: guard for key in ("guard_fixed", "ridge_fixed_guard", "ridge_tmave_guard")}
    shifts = {}
    for key in ("trend", *MODELS):
        targets[key], shifts[key] = tilt_stock_gold(core, signals[key].to_numpy())
    for who in ("fixed", "tmave"):
        key = f"ridge_{who}"
        targets[f"{key}_guard"] = scaled_targets(targets[key], scale)
    targets = {key: targets[key] for key in LABELS}
    for key, target in targets.items():
        assert (target >= -1e-12).all() and (target.sum(axis=1) <= 1 + 1e-10).all()
        if key != "risk60":
            np.testing.assert_array_equal(target[:, 4], core[:, 4])

    dev = periods["development"]
    cost_scores = {
        bps: {key: performance_stats(account_path(market, values, key, dev, bps)) for key, values in targets.items()}
        for bps in (BASE_COST, 10)
    }
    joint = [key for key in CANDIDATES if all(covers_objectives(v[key], v["core"]) for v in cost_scores.values())]
    winner, frontier = development_pick(cost_scores[BASE_COST], joint or CANDIDATES)
    selection = pd.DataFrame([
        {"strategy": key, **values, "one_way_bps": bps, "in_tmave_selection_pool": key in CANDIDATES,
         "joint_improvement_both_costs": key in joint, "pareto_for_choice": key in frontier,
         "chosen_for_replay_comparison": key == winner,
         "retained_by_development_rule": key == (winner if joint else "core")}
        for bps, table in cost_scores.items() for key, values in table.items()
    ])
    evidence = signals.loc[dates[signal_rows]].join(labels)
    evidence = evidence.join(features.add_prefix("feature_"))
    for key, change in shifts.items():
        evidence[f"transfer_into_equity_{key}"] = change[signal_rows]
    evidence["execution_date"] = dates[signal_rows + 1]
    evidence["phase"] = ""
    for phase, (start, end) in periods.items():
        mask = evidence.execution_date.between(dates[start], dates[end - 1])
        evidence.loc[mask, "phase"] = phase
        # 评估归属按成交期。标签不能用分区之后的价格。
        outside = mask & evidence.label_end.gt(dates[end - 1])
        evidence.loc[outside, list(labels.columns)] = np.nan
    evidence.index.name = "signal_date"
    return targets, controls, selection, winner, folds, evidence


def signal_quality(evidence):
    rows = []
    for phase, frame in evidence.groupby("phase", sort=False):
        scopes = [(phase, frame)]
        if phase == "replay":
            mask = frame.execution_date.dt.year.eq(2026)
            scopes.append(("replay_2026", frame.loc[mask]))
        for label, samples in scopes:
            for key in ("trend", *MODELS):
                matched = samples[[key, "risk_normalized_spread"]].dropna()
                rank = matched.corr(method="spearman").at[key, "risk_normalized_spread"]
                signs = np.sign(matched[key]) == np.sign(matched.risk_normalized_spread)
                rows.append({"phase": label, "model": key, "nonoverlap_weeks": len(matched),
                             "rank_corr": rank, "direction_accuracy": signs.mean()})
    return pd.DataFrame(rows)


def check_tilt():
    base = np.array([[.06, .06, .06, .12, .7], [.03, .02, .01, .24, .7], [.04, .06, .10, .10, .7]])
    positions, transfer = tilt_stock_gold(base, np.array([100., -100., 0.]))
    np.testing.assert_allclose(transfer, [.06, -.06, 0.])
    np.testing.assert_allclose(positions[:, 3], [.06, .30, .1])
    np.testing.assert_array_equal(positions[:, 4], [.7, .7, .7])


def run_alpha(folder, now):
    check_engine()
    check_clocks()
    check_tilt()
    market, _, hashes = load_market()
    factors = build_factors(market.close)
    periods, fresh = period_ranges(market.close.index, FREEZE_DATE)
    targets, controls, selection, winner, folds, signals = design_experiment(market, factors, periods)
    performance, paths = evaluate_sleeves(market, targets, controls, periods)
    quality = signal_quality(signals)
    windows, candidates = scenario_windows(np.log(market.close).diff(), periods)
    scenarios = scenario_results(windows, paths, LABELS)
    compare_paths = dict(paths)
    compare_paths.update({("replay_2026", key): path.loc[path.index.year == 2026]
                          for (phase, key), path in paths.items() if phase == "replay"})
    same = "ridge_fixed_guard" if winner.endswith("_guard") else "ridge_fixed"
    intervals = paired_intervals(
        compare_paths, {"policy": winner, "core": "core"}, benchmarks=("core", same, "risk60"),
        phases=("replay", "replay_2026", "independent"), drawdown_interval=True,
    )
    folder.mkdir(parents=True, exist_ok=False)
    for name, table in {
        "portfolio_metrics": performance, "selection": selection, "train_folds": folds,
        "forecast_quality": quality, "paired_intervals": intervals, "scenarios": scenarios,
    }.items():
        table.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    signals.to_csv(folder / "relative_signals.csv", encoding="utf-8-sig", float_format="%.12g")
    pd.concat(paths, names=["phase", "strategy"]).to_csv(
        folder / "daily_backtest.csv", encoding="utf-8-sig", float_format="%.12g",
    )
    from report import write_alpha_report

    write_alpha_report(folder, now, performance, paths, selection, winner, same, quality,
                       scenarios, candidates, intervals, fresh, hashes)
    print(performance.query("phase=='replay' and year in ['all','2026'] and one_way_bps==5")[
        ["strategy", "year", "cagr", "total_return", "sharpe", "max_drawdown", "cdar95"]
    ].to_string(index=False))
    print(f"开发期锁定 {winner}；结果目录：{folder}")
