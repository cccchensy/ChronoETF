"""以 T-MAVE 对股金组合方差预测作周频波动控制。"""

import numpy as np
import pandas as pd

from backtest import BASE_COST, BOOTSTRAP_SEED, CODES, PER_YEAR, check_engine, load_market
from horizon import holm_adjust
from risk import HORIZON, VAR_FLOOR, fit_readout, qlike
from sleeves import FREEZE_DATE, NAMES, build_factors, evaluate_sleeves, fixed_session_states
from strategy import allocations, base_portfolio, bootstrap_difference, check_allocation, check_clocks, circular_blocks
from tmave import period_ranges, weekly_signals


FLOOR = .35
LEVELS = {75: ("ewma", "plain", "fixed", "trade", "full"), 90: ("ewma", "full")}
MODEL_NAMES = {"ewma": "EWMA", "plain": "同因子多窗口", "fixed": "同因子固定双记忆",
               "trade": "T-MAVE 交易日", "full": "T-MAVE 三时钟", "static": "恒定缩仓", "matched": "同仓位静态"}
MATCHED_DYNAMIC = {f"matched_{name}_75": name for name in ("ewma", "fixed", "trade")}
LABELS = {"budget_60": "固定 60 日", "budget_120": "固定 120 日", **{
    f"{name}_{level}": f"{MODEL_NAMES[name]} · {level}%"
    for level, models in LEVELS.items() for name in ("static", *models, "matched")
}, **{key: f"同仓位 {MODEL_NAMES[method]}" for key, method in MATCHED_DYNAMIC.items()}}
PRIMARY = ("budget_60", "matched_75", "ewma_75", "fixed_75")
TAIL_WINDOW = 20


def holding_schedule(dates, periods):
    sources = {"formation": (dates.searchsorted("2016-01-01"), periods["development"][0]), **periods}
    parts = []
    for phase, (start, end) in sources.items():
        rows = weekly_signals(dates, start, end)
        end_label = rows + HORIZON + 1
        parts.append(pd.DataFrame({
            "phase": phase, "signal_row": rows, "signal_date": dates[rows], "execution_date": dates[rows + 1],
            "days_held": np.diff(np.r_[rows + 1, end]), "label_row": end_label,
            "label_in_phase": end_label < end,
        }))
    return pd.concat(parts, ignore_index=True)


def risk_inputs(market, schedule, book):
    """把所有历史与标签投影到同一信号日的股金参考权重。"""
    close_moves = np.log(market.close.iloc[:, :4]).diff().to_numpy()
    open_moves = np.log(market.opening.iloc[:, :4]).diff().to_numpy()
    ratio = book[:, :4] / book[:, :4].sum(axis=1, keepdims=True)
    values = []
    for item, weights in zip(schedule.itertuples(), ratio):
        t = item.signal_row
        use = weights > 0
        daily = close_moves[:t + 1, use] @ weights[use]
        fast = np.nanmean(daily[-5:] ** 2)
        middle = np.nanmean(daily[-20:] ** 2)
        level = max(np.nanmean(daily[-60:] ** 2), VAR_FLOOR)
        ewma = pd.Series(daily ** 2).ewm(alpha=.06, adjust=False, min_periods=20, ignore_na=True).mean().iat[-1]
        downside = np.nanmean(np.minimum(daily[-20:], 0) ** 2)
        equities = weights[:3] > 0
        stock = close_moves[t - 59:t + 1, :3][:, equities] @ (weights[:3][equities] / weights[:3].sum())
        gold = close_moves[t - 59:t + 1, 3]
        joined = pd.DataFrame({"stock": stock, "gold": gold})
        correlation_gap = joined.iloc[-20:].corr().iat[0, 1] - joined.corr().iat[0, 1]
        following = open_moves[t + 2:t + 2 + HORIZON, use] @ weights[use]
        realized = float(np.mean(following ** 2)) if len(following) == HORIZON else np.nan
        values.append((level, fast, middle, ewma, downside, correlation_gap, realized))
    return pd.DataFrame(values, columns=("anchor", "var5", "var20", "ewma", "down20", "correlation_gap", "realized"))


def fit_sleeve_forecasts(schedule, risk, book, factors):
    level = risk.anchor.to_numpy()
    common = np.column_stack((np.ones(len(risk)), np.log(risk[["var5", "var20", "ewma"]].to_numpy() / level[:, None]),
                              np.log(risk.down20 / level + .05), risk.correlation_gap.to_numpy()))
    common = np.clip(common, -3., 3.)
    two = np.column_stack((book[:, :3].sum(axis=1), book[:, 3]))
    two /= two.sum(axis=1, keepdims=True)
    fields = factors.frame
    state_bank = {"fixed": fixed_session_states(factors), **{
        key: np.stack((fields[f"short_{key}"].to_numpy().reshape(-1, len(NAMES)),
                       fields[f"anchor_{key}"].to_numpy().reshape(-1, len(NAMES))), axis=-1)
        for key in ("trade", "full")
    }}
    states = {name: np.einsum("sg,sgk->sk", two, collection[schedule.signal_row, :2])
              for name, collection in state_bank.items()}
    features = {"plain": common, **{name: np.column_stack((common, state)) for name, state in states.items()}}
    forecasts = pd.DataFrame({"ewma": risk.ewma}, index=risk.index)
    for key in features:
        forecasts[key] = np.nan
    folds = []
    grouped = schedule.loc[schedule.phase.ne("formation")].groupby(schedule.execution_date.dt.year)
    for year, section in grouped:
        first = section.signal_row.iloc[0]
        training = schedule.label_row.lt(first).to_numpy()
        for name, design in features.items():
            values, count = fit_readout(design[:, None, :], level[:, None], risk.realized.to_numpy()[:, None],
                                        training, section.index.to_numpy())
            forecasts.loc[section.index, name] = values.ravel()
            used = schedule.loc[training, "label_row"].max()
            folds.append({"execution_year": year, "model": name, "first_signal": first,
                          "last_allowed_label_row": used, "training_weeks": count})
    return forecasts, states, pd.DataFrame(folds)


def risk_multiplier(goal, variance):
    return np.clip(goal / np.sqrt(np.maximum(variance, VAR_FLOOR)), FLOOR, 1.)


def calibrate_goal(variance, weights, level):
    low, high = 0., np.sqrt(variance.max())
    for _ in range(40):
        guess = (low + high) / 2
        if np.average(risk_multiplier(guess, variance), weights=weights) < level:
            low = guess
        else:
            high = guess
    return (low + high) / 2


def risk_books(schedule, forecasts, base):
    positions = schedule.signal_row.to_numpy()
    exposure = base[positions, :4].sum(axis=1) * schedule.days_held.to_numpy()
    development = schedule.phase.eq("development").to_numpy()
    goals, targets, scales = [], {}, {}
    investments = np.column_stack((base[positions, :3].sum(axis=1), base[positions, 3]))
    capital_days = investments * schedule.days_held.to_numpy()[:, None]

    def weights(equity, gold):
        book = base.copy()
        book[positions, :3] *= np.asarray(equity)[:, None]
        book[positions, 3] *= gold
        return book

    for level, methods in LEVELS.items():
        nominal = level / 100
        constant = np.full(len(positions), nominal)
        targets[f"static_{level}"] = weights(constant, constant)
        scales[f"static_{level}"] = constant
        for method in methods:
            variance = forecasts[method].to_numpy()
            goal = calibrate_goal(variance[development], exposure[development], nominal)
            scale = risk_multiplier(goal, variance)
            np.testing.assert_allclose(np.average(scale[development], weights=exposure[development]), nominal, atol=1e-10)
            goals.append({"level": level, "model": method, "goal_annualized": goal * np.sqrt(PER_YEAR),
                          "development_weighted_scale": np.average(scale[development], weights=exposure[development])})
            key = f"{method}_{level}"
            targets[key] = weights(scale, scale)
            scales[key] = scale
        full = scales[f"full_{level}"]
        constant_equivalent = np.zeros_like(investments)
        for _, segment in schedule.groupby("phase", sort=False):
            idx = segment.index.to_numpy()
            actual = capital_days[idx] * full[idx, None]
            match = actual.sum(axis=0) / capital_days[idx].sum(axis=0)
            constant_equivalent[idx] = match
        targets[f"matched_{level}"] = weights(*constant_equivalent.T)
        scales[f"matched_{level}_equity"], scales[f"matched_{level}_gold"] = constant_equivalent.T
        if level == 75:
            for key, method in MATCHED_DYNAMIC.items():
                variance = forecasts[method].to_numpy()
                multipliers = np.zeros_like(investments)
                for _, segment in schedule.groupby("phase", sort=False):
                    idx = segment.index.to_numpy()
                    for j in range(2):
                        goal = calibrate_goal(variance[idx], capital_days[idx, j], constant_equivalent[idx[0], j])
                        multipliers[idx, j] = risk_multiplier(goal, variance[idx])
                targets[key] = weights(*multipliers.T)
                scales[f"{key}_equity"], scales[f"{key}_gold"] = multipliers.T
    for value in targets.values():
        np.testing.assert_array_equal(value[:, 4], base[:, 4])
        assert np.all(value >= 0.) and np.all(value.sum(axis=1) <= 1 + 1e-10)
    return targets, pd.DataFrame(goals), pd.DataFrame(scales)


def forecast_scores(evidence, forward, forecasts):
    rows = []
    for phase, frame in evidence.groupby("phase", sort=False):
        if phase == "formation":
            continue
        periods = [(phase, frame)]
        if phase == "replay":
            periods.append(("replay_2026", frame.loc[frame.execution_date.dt.year.eq(2026)]))
        for period, part in periods:
            eligible = part.index[part.label_in_phase & forward.loc[part.index].notna() & forecasts.loc[part.index].notna().all(axis=1)]
            actual = forward.loc[eligible]
            for name, pred in forecasts.loc[eligible].items():
                rows.append({"phase": period, "forecast": name, "weeks": len(actual), "qlike": qlike(actual, pred).mean(),
                             "mean_forecast_over_realized": pred.mean() / actual.mean(),
                             "risk_rank_corr": pred.corr(actual, method="spearman")})
    return pd.DataFrame(rows)


def forecast_tests(detail):
    """同一调仓日期、同一标签的方差预测损失差。"""
    subset = detail.loc[detail.phase.eq("replay") & detail.execution_date.dt.year.eq(2026)]
    samples = pd.concat([detail, subset.assign(phase="replay_2026")], ignore_index=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for phase, section in samples.groupby("phase", sort=False):
        columns = ["realized", *(f"forecast_{key}" for key in ("full", "ewma", "fixed"))]
        eligible = section[columns].dropna()
        truth = eligible.realized
        loss = qlike(truth, eligible.forecast_full)
        for block in (2, 4, 8):
            index = circular_blocks(rng, len(eligible), block)
            individual = []
            for peer in ("ewma", "fixed"):
                delta = (qlike(truth, eligible[f"forecast_{peer}"]) - loss).to_numpy()
                individual.append({
                    "phase": phase, "strategy": "full", "benchmark": peer, "block": block, "block_unit": "weeks",
                    "metric": "forecast_qlike", "observations": len(eligible), "primary": False, "analysis": "forecast",
                    **bootstrap_difference(delta.mean(), delta[index].mean(axis=1), with_pvalue=True),
                })
            for row, adjusted in zip(individual, holm_adjust([x["p_raw"] for x in individual])):
                rows.append({**row, "p_holm": adjusted})
    return pd.DataFrame(rows)


def rolling_volatility(path):
    return path.net_return.rolling(TAIL_WINDOW).std() * np.sqrt(PER_YEAR)


def tail_stats(paths):
    rows = []
    for (phase, key), path in paths.items():
        for year, segment in [("all", path), *((str(y), p) for y, p in path.groupby(path.index.year))]:
            rows.append({"phase": phase, "strategy": key, "year": year, "one_way_bps": BASE_COST,
                         "vol20_p90": rolling_volatility(segment).quantile(.90)})
    return pd.DataFrame(rows)


def risk_tests(paths):
    peers = {75: (*PRIMARY, "static_75", "plain_75", "trade_75", *MATCHED_DYNAMIC),
             90: ("budget_60", "matched_90", "ewma_90")}
    around_2026 = {("replay_2026", key): p.loc[p.index.year == 2026] for (phase, key), p in paths.items() if phase == "replay"}
    sample = {**paths, **around_2026}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = []

    def statistics(x, vol):
        risk = x.std(axis=-1, ddof=1) * np.sqrt(PER_YEAR)
        annual = np.expm1(PER_YEAR * np.log1p(x).mean(axis=-1))
        return {"annual_vol": risk, "vol20_p90": np.quantile(vol, .90, axis=-1), "cagr": annual}

    for phase in ("development", "replay", "replay_2026", "independent"):
        if (phase, "budget_60") not in sample:
            continue
        for level, names in peers.items():
            key = f"full_{level}"
            bins = (key, *names)
            net = {name: sample[(phase, name)].net_return.to_numpy() for name in bins}
            rolling = {name: rolling_volatility(sample[(phase, name)]).dropna().to_numpy() for name in bins}
            size, tails = len(net[key]), len(rolling[key])
            for block in (10, 20, 40):
                i = circular_blocks(rng, size, block)
                j = circular_blocks(rng, tails, block)
                a = statistics(net[key], rolling[key])
                boot_a = statistics(net[key][i], rolling[key][j])
                for ref in names:
                    b = statistics(net[ref], rolling[ref])
                    boot_b = statistics(net[ref][i], rolling[ref][j])
                    for metric in a:
                        if metric == "cagr":
                            point = a[metric] - b[metric]
                            change = boot_a[metric] - boot_b[metric]
                        else:
                            point = 1 - a[metric] / b[metric]
                            change = 1 - boot_a[metric] / boot_b[metric]
                        estimates.append({
                            "phase": phase, "strategy": key, "benchmark": ref, "block": block, "metric": metric,
                            "observations": tails if metric == "vol20_p90" else size,
                            "primary": level == 75 and metric == "annual_vol" and ref in PRIMARY,
                            "analysis": "same_exposure" if ref in MATCHED_DYNAMIC else "prespecified", "block_unit": "sessions",
                            **bootstrap_difference(point, change, with_pvalue=True),
                        })
    tests = pd.DataFrame(estimates)
    tests["p_holm"] = np.nan
    matched = tests.analysis.eq("same_exposure") & tests.metric.eq("annual_vol")
    for _, rows in tests[tests.primary | matched].groupby(["analysis", "phase", "block"]):
        tests.loc[rows.index, "p_holm"] = holm_adjust(rows.p_raw.to_numpy())
    return tests


def run_volatility_policy(folder, now):
    check_engine()
    check_clocks()
    check_allocation()
    market, _, hashes = load_market()
    dates = market.close.index
    periods, fresh = period_ranges(dates, FREEZE_DATE)
    schedule = holding_schedule(dates, periods)
    ids = schedule.signal_row.to_numpy()
    books = allocations(market.close, ids)
    base = {key: base_portfolio(market, books[key]) for key in ("budget_60", "budget_120")}
    book = base["budget_60"][ids]
    inputs = risk_inputs(market, schedule, book)
    predictions, states, folds = fit_sleeve_forecasts(schedule, inputs, book, build_factors(market.close))
    folds["first_signal_date"] = dates[folds.first_signal]
    folds["last_label_end_open"] = dates[folds.last_allowed_label_row]
    assert folds.last_label_end_open.lt(folds.first_signal_date).all()
    use = schedule.phase.ne("formation")
    overlays, goals, scale = risk_books(schedule.loc[use].reset_index(drop=True), predictions.loc[use].reset_index(drop=True), base["budget_60"])
    performance, paths = evaluate_sleeves(market, {**base, **overlays}, {}, periods)
    metrics = performance.merge(tail_stats(paths), how="left", on=["phase", "strategy", "year", "one_way_bps"], validate="one_to_one")
    scores = forecast_scores(schedule, inputs.realized, predictions)
    base_departure = schedule.loc[use].assign(label_end_open=dates.take(np.minimum(schedule.loc[use, "label_row"], len(dates) - 1)))
    safe_risk = inputs.loc[use].copy()
    safe_risk.loc[~schedule.label_in_phase, "realized"] = np.nan
    detail = pd.concat([base_departure, safe_risk, predictions.loc[use].add_prefix("forecast_"),
                        pd.DataFrame({f"{mode}_{field}": state[:, j] for mode, state in states.items()
                                      for j, field in enumerate(("fast", "anchor"))}).loc[use],
                        scale.set_index(schedule.index[use]).add_prefix("scale_")], axis=1)
    detail.loc[~detail.label_in_phase, "label_end_open"] = pd.NaT
    tests = pd.concat([risk_tests(paths), forecast_tests(detail)], ignore_index=True)
    daily = pd.concat(paths, names=["phase", "strategy"]).reset_index()
    daily = daily.drop(columns=[c for c in daily if c == "guarded" or c.startswith(("pressure_", "enter_", "exit_", "action_", "guard_"))])
    folder.mkdir(parents=True, exist_ok=False)
    for name, data in {
        "portfolio_metrics": metrics, "daily_backtest": daily, "weekly_forecasts": detail,
        "forecast_scores": scores, "paired_tests": tests, "calibration": goals, "train_folds": folds,
    }.items():
        data.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    from report import write_volatility_report

    write_volatility_report(folder, now, metrics, paths, scores, tests, goals, fresh, hashes)
    print(metrics.query("phase=='replay' and year=='all' and one_way_bps==5")[
        ["strategy", "annual_vol", "vol20_p90", "cagr", "sharpe", "max_drawdown"]
    ].to_string(index=False))
    print(f"结果目录 {folder}")
