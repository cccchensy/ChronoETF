"""预设的条件记忆、分段慢写和下行约束实验。"""

import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit

from backtest import BASE_COST, CODES, Market, PER_YEAR, simulate
from residual_policy import forecast, prediction_scores, prepare
from risk import HORIZON
from sleeves import build_factors, evaluate_sleeves
from strategy import bootstrap_difference, circular_blocks, flow_jump
from tmave import scenario_results, scenario_windows
from transfer_research import transfer_data
from volatility_policy import calibrate_goal, fit_sleeve_forecasts, risk_multiplier


ROOT = Path(__file__).resolve().parent
FREEZE = pd.Timestamp("2026-11-21")
READOUTS = ("joint_full", "short", "conditional", "mean", "ema", "donor", "capped")
CANDIDATES = ("conditional", "capped", "tail", "capped_tail")
SLOW = [8, 10]
ARMS = {name: (name, None) for name in ("calibrated", "full", *READOUTS[1:])} | {
    "tail_short": ("conditional", "short"),
    "tail": ("conditional", "conditional"),
    "capped_tail": ("capped", "capped"),
}
METRICS = ("cagr", "annual_vol", "max_drawdown", "cdar95")


def capped_writes(gates, directions):
    """逐组给同一事件段留出至少一半的段首慢状态写入系数。"""
    limited = np.zeros_like(gates)
    retained = np.ones_like(gates)
    last_row = np.full(gates.shape[1], -100)
    last_sign = np.zeros(gates.shape[1])
    remaining = np.ones(gates.shape[1])
    for i, sign in enumerate(directions):
        event = sign != 0
        new = event & ((sign != last_sign) | (i - last_row > 4))
        remaining[new] = 1
        headroom = np.maximum(1 - .5 / remaining, 0)
        limited[i] = np.minimum(gates[i], headroom)
        remaining *= 1 - limited[i]
        retained[i] = remaining
        last_row[event], last_sign[event] = i, sign[event]
    return limited, retained


def memory_inputs(context):
    schedule, risk, old, history, base = context[:5]
    signals = schedule.signal_row.to_numpy()
    group_weights = np.column_stack((base[signals, :3].sum(axis=1), base[signals, 3]))
    group_weights /= group_weights.sum(axis=1, keepdims=True)
    read = schedule.regime.eq("confirmed_under").to_numpy()
    full = old["joint_full"]
    features = {"joint_full": full, **{name: full.copy() for name in READOUTS[1:]}}
    features["short"][:, SLOW] = 0
    features["conditional"][:, SLOW] *= read[:, None]
    write_evidence = pd.DataFrame(index=history.index)

    def values(column):
        return history[column].fillna(0).to_numpy().reshape(-1, 3)

    def project(states):
        return (states[signals, :2] * group_weights).sum(axis=1)

    for name, columns, fast_col, slow_col in (
        ("error", ("candidate", "direction", "fast_gate", "slow_gate", "elapsed_full"), 7, 8),
        ("price", ("price_pressure_candidate", "price_event_direction", "price_fast_gate",
                   "price_anchor_gate", "price_elapsed_full"), 9, 10),
    ):
        candidate, direction, fast, slow, elapsed = (values(col) for col in columns)
        limited, retained = capped_writes(slow, direction)
        np.testing.assert_array_less(.5 - 1e-12, retained)
        write_evidence[f"{name}_original_gate"] = slow.ravel()
        write_evidence[f"{name}_capped_gate"] = limited.ravel()
        write_evidence[f"{name}_retained"] = retained.ravel()
        state = flow_jump(candidate, elapsed, fast, limited)
        features["capped"][:, fast_col] = project(state[:, :, 0])
        features["capped"][:, slow_col] = project(state[:, :, 1]) * read
        candidates = pd.DataFrame(candidate)
        features["mean"][:, slow_col] = project(candidates.rolling(30, min_periods=1).mean().to_numpy()) * read
        features["ema"][:, slow_col] = project(candidates.ewm(halflife=30, adjust=False).mean().to_numpy()) * read

    bins = np.digitize(risk.var20 / risk.anchor, [.75, 1.25])
    rng = np.random.default_rng(42)
    source = np.full(len(signals), -1)
    features["donor"][:, SLOW] = 0
    for i, t in enumerate(signals):
        choices = np.flatnonzero((signals <= t - 60) & (bins == bins[i]))
        if len(choices):
            donor = int(rng.choice(choices))
            source[i] = signals[donor]
            if read[i]:
                features["donor"][i, SLOW] = full[donor, SLOW]
    return features, source, write_evidence


def future_depth(market, schedule, base):
    moves = np.diff(np.log(market.opening.iloc[:, :4].to_numpy()), axis=0)
    out = np.full(len(schedule), np.nan)
    for row in schedule.itertuples():
        start = row.signal_row + 1
        if len(moves[start:start + HORIZON]) < HORIZON:
            continue
        weights = base[row.signal_row, :4]
        eligible = weights > 0
        following = moves[start:start + HORIZON, eligible] @ (weights[eligible] / weights.sum())
        nav = np.exp(np.r_[0., following.cumsum()])
        out[row.Index] = np.max(1 - nav / np.maximum.accumulate(nav))
    return out


def fit_tail(x, actual, train, test):
    mean = x[train].mean(axis=0)
    scale = np.maximum(x[train].std(axis=0), 1e-6)
    design = np.column_stack((np.ones(len(x)), np.clip((x - mean) / scale, -3, 3)))
    xx, y = design[train], actual[train]

    def objective(beta):
        logit = xx @ beta
        penalty = beta.copy()
        penalty[0] = 0
        loss = np.mean(np.logaddexp(0, logit) - y * logit) + .05 * (penalty @ penalty)
        grad = xx.T @ (expit(logit) - y) / len(y) + .10 * penalty
        return loss, grad

    opt = minimize(objective, np.zeros(xx.shape[1]), jac=True, method="L-BFGS-B",
                   bounds=[(-6, 6), *[(-2, 2)] * (xx.shape[1] - 1)])
    if not opt.success:
        raise RuntimeError(opt.message)
    return expit(design[test] @ opt.x)


def tail_predictions(schedule, features, depth):
    methods = ("short", "conditional", "capped")
    prob = pd.DataFrame(np.nan, index=schedule.index, columns=methods)
    truth_level = pd.Series(np.nan, index=schedule.index, name="tail_truth_level")
    folds = []
    year = schedule.execution_date.dt.year
    for execution_year, section in schedule[year.ge(2018)].groupby(year):
        first = int(section.signal_row.iloc[0])
        train = schedule.label_row.lt(first).to_numpy() & np.isfinite(depth)
        cut = np.quantile(depth[train], .90)
        test = section.index.to_numpy()
        truth_level.loc[test] = cut
        folds.append({"year": int(execution_year), "first_signal_row": first,
                      "last_train_label": int(schedule.label_row[train].max()), "training_weeks": int(train.sum()),
                      "truth_threshold": cut, "tail_training_events": int((depth[train] > cut).sum())})
        for method in methods:
            prob.loc[test, method] = fit_tail(features[method], depth > cut, train, test)
    dev = schedule.phase.eq("development") & schedule.label_in_phase
    cuts = prob[dev].quantile(.90)
    alerts = prob.gt(cuts, axis="columns")
    return prob, truth_level, cuts, alerts, pd.DataFrame(folds)


def risk_forecasts(market):
    ctx = prepare(market, FREEZE)
    schedule, risk = ctx[:2]
    features, donor, gates = memory_inputs(ctx)
    predicted, folds, _, shrinkage = forecast(schedule, risk, features, READOUTS)
    point = predicted[["calibrated"]].copy()
    for method in READOUTS:
        point["full" if method == "joint_full" else method] = predicted[f"{method}_paired"]
    depth = future_depth(market, schedule, ctx[4])
    tail = tail_predictions(schedule, features, depth)
    return ctx, point, depth, tail, folds, shrinkage, donor, gates


def target_books(market, context, predicted, alerts, frozen):
    schedule, risk, _, _, base, periods = context[:6]
    active = schedule.phase.ne("formation")
    signals = schedule.signal_row[active].to_numpy()
    dev = schedule.phase[active].eq("development").to_numpy()
    days = schedule.days_held[active].to_numpy()
    capital = base[signals, :4].sum(axis=1) * days
    targets, calibrations = {}, []
    if frozen:
        factors = build_factors(market.close)
        legacy, _, _ = fit_sleeve_forecasts(schedule, risk, base[schedule.signal_row], factors)
        equity_gold = np.column_stack((base[signals, :3].sum(axis=1), base[signals, 3]))
        group_days = (equity_gold * days[:, None])[dev]

    def as_book(multiplier):
        book = base.copy()
        book[signals, :4] *= multiplier
        return book

    def dev_vol(book):
        p = simulate(market, book, "risk", *periods["development"], BASE_COST)
        return p.net_return.std() * np.sqrt(PER_YEAR)

    for level in (75, 90):
        reference = predicted.loc[active, "calibrated"].to_numpy()
        common_goal = calibrate_goal(reference[dev], capital[dev], level / 100)
        vol_target = dev_vol(as_book(risk_multiplier(common_goal, reference)[:, None]))
        for method, (var_model, tail_model) in ARMS.items():
            variance = predicted.loc[active, var_model].to_numpy()
            upper = np.where(alerts.loc[active, tail_model], .60, 1.) if tail_model else np.ones(len(variance))

            def book_at(goal):
                scale = np.minimum(risk_multiplier(goal, variance), upper)
                return as_book(scale[:, None])

            def gap(goal):
                return dev_vol(book_at(goal)) - vol_target

            common_book = book_at(common_goal)
            common_vol = dev_vol(common_book)
            if abs(common_vol - vol_target) <= 1e-12:
                matched_goal = common_goal
            else:
                hi = np.sqrt(variance[dev].max())
                low_gap, high_gap = gap(0.), gap(hi)
                if low_gap * high_gap <= 0:
                    matched_goal = brentq(gap, 0., hi, xtol=1e-14)
                else:
                    matched_goal = 0. if abs(low_gap) < abs(high_gap) else hi
            for lane, goal in (("common", common_goal), ("matched", matched_goal)):
                key = f"{lane}_{method}_{level}"
                targets[key] = common_book if goal == common_goal else book_at(goal)
                calibrated_vol = common_vol if goal == common_goal else dev_vol(targets[key])
                calibrations.append({"strategy": key, "goal_daily_sd": goal, "development_vol": calibrated_vol,
                                     "matching_gap": calibrated_vol - vol_target, "target_development_vol": vol_target})
        if frozen:
            legacy_variance = legacy.loc[active, "full"].to_numpy()
            goal = calibrate_goal(legacy_variance[dev], capital[dev], level / 100)
            scale = risk_multiplier(goal, legacy_variance)
            equity, gold = np.average(np.broadcast_to(scale[dev, None], group_days.shape), axis=0, weights=group_days)
            targets[f"frozen_{level}"] = as_book(scale[:, None])
            targets[f"static_dev_{level}"] = as_book(np.array([equity, equity, equity, gold]))
            calibrations.append({"strategy": f"static_dev_{level}", "equity_scale": equity, "gold_scale": gold})
            assert np.allclose((group_days * scale[dev, None]).sum(axis=0), group_days.sum(axis=0) * [equity, gold])
    for book in targets.values():
        assert np.isfinite(book).all() and (book >= 0).all() and (book.sum(axis=1) <= 1 + 1e-10).all()
        np.testing.assert_array_equal(book[:, 4], base[:, 4])
    return targets, pd.DataFrame(calibrations)


def score_tail(schedule, depth, tail):
    prob, truth_level, thresholds, alerts, _ = tail
    truth = depth > truth_level
    rows = []
    for phase in ("development", "replay", "independent"):
        valid = schedule.phase.eq(phase) & schedule.label_in_phase & np.isfinite(depth)
        if not valid.any():
            continue
        for model in prob:
            y, p, a = truth[valid], prob.loc[valid, model], alerts.loc[valid, model]
            rows.append({"phase": phase, "model": model, "weeks": len(y), "events": int(y.sum()),
                         "signals": int(a.sum()), "hits": int((a & y).sum()),
                         "false_positive": int((a & ~y).sum()), "misses": int((~a & y).sum()),
                         "brier": float(((p - y) ** 2).mean()), "probability_threshold": float(thresholds[model])})
    return pd.DataFrame(rows)


def scenario_evidence(market, context, paths):
    # 压力路径分类仅用于事后评价，观察目标股金倍率持续低于或等于 60% 的天数。
    windows, _ = scenario_windows(np.log(market.close).diff(), context[5])
    keys = [f"matched_{model}_75" for model in ("calibrated", *CANDIDATES)]
    for key in keys:
        path = paths["replay", key]
        signals = market.close.index.get_indexer(path.signal_date.ffill())
        desired = path.target_equity + path.target_gold
        base = context[4][signals, :4].sum(axis=1)
        path["guarded"] = desired.div(base).le(.60 + 1e-12)
    return scenario_results(windows, paths, keys)


def gate_results(performance):
    data = performance.query("year == 'all' and one_way_bps in [5, 10]")
    rows = []
    for (basket, phase, fee), part in data.groupby(["basket", "phase", "one_way_bps"]):
        part = part.set_index("strategy")
        base = part.loc["matched_calibrated_75"]
        for name in CANDIDATES:
            row = part.loc[f"matched_{name}_75"]
            checks = {"cdar": row.cdar95 <= .95 * base.cdar95,
                      "drawdown": row.max_drawdown >= base.max_drawdown - 1e-12,
                      "volatility": row.annual_vol <= base.annual_vol + .0005,
                      "return": row.cagr >= base.cagr - .002}
            rows.append({"basket": basket, "phase": phase, "fee": int(fee), "model": name,
                         **{metric: bool(ok) for metric, ok in checks.items()}, "passed": all(checks.values()),
                         "cdar95": float(row.cdar95)})
    gates = pd.DataFrame(rows)
    dev = gates.query("basket == 'original' and phase == 'development'")
    candidates = [name for name in CANDIDATES if dev.loc[dev.model.eq(name), "passed"].all()]
    level = dev.query("fee == 5").set_index("model").cdar95
    choice = min(candidates, key=level.__getitem__) if candidates else None
    replay_ok = bool(choice and gates.query("phase == 'replay' and model == @choice").passed.all())
    return gates, {"development_choice": choice, "all_replay_gates_pass": replay_ok}


def risk_stats(returns):
    path = np.cumprod(1 + returns, axis=-1)
    depth = 1 - path / np.maximum.accumulate(np.maximum(path, 1), axis=-1)
    k = int(np.ceil(depth.shape[-1] * .05))
    tail = np.partition(depth, depth.shape[-1] - k, axis=-1)[..., -k:]
    return np.stack((np.expm1(PER_YEAR * np.log1p(returns).mean(axis=-1)),
                     returns.std(axis=-1, ddof=1) * np.sqrt(PER_YEAR),
                     -depth.max(axis=-1), tail.mean(axis=-1)), axis=-1)


def economic_intervals(paths):
    pairs = [("conditional", "calibrated"), ("capped", "conditional"), ("tail", "conditional"),
             ("tail", "tail_short"), ("capped_tail", "capped"), ("full", "calibrated")]
    keys = [(f"matched_{a}_75", f"matched_{b}_75") for a, b in pairs]
    keys.append(("frozen_75", "static_dev_75"))
    arrays = {name: {k: p.net_return.to_numpy() for (phase, k), p in book.items() if phase == "replay"}
              for name, book in paths.items()}
    dates = [book["replay", "matched_calibrated_75"].index for book in paths.values()]
    assert all(x.equals(dates[0]) for x in dates)
    rows = []
    for block in (10, 20, 40):
        indices = circular_blocks(np.random.default_rng(42 + block), len(dates[0]), block)
        for a, b in keys:
            changes = []
            for basket, returns in arrays.items():
                if a not in returns:
                    continue
                left, right = returns[a], returns[b]
                changes.append((basket, risk_stats(left) - risk_stats(right),
                                risk_stats(left[indices]) - risk_stats(right[indices])))
            if len(changes) > 1:
                changes.append(("pooled", np.mean([x[1] for x in changes], axis=0),
                                np.mean([x[2] for x in changes], axis=0)))
            for scope, point, draws in changes:
                for j, metric in enumerate(METRICS):
                    rows.append({"scope": scope, "strategy": a, "benchmark": b, "metric": metric,
                                 "block": block, "observations": len(dates[0]),
                                 **bootstrap_difference(point[j], draws[:, j])})
    return pd.DataFrame(rows)


def forecast_intervals(losses):
    wide = losses.query("phase == 'replay'").pivot(index="signal_date", columns=["basket", "model"], values="qlike")
    assert wide.notna().all().all()
    rows = []
    for block in (2, 4, 8):
        ix = circular_blocks(np.random.default_rng(42 + block), len(wide), block)
        variants = {**{b: wide[b] for b in wide.columns.get_level_values(0).unique()},
                    "pooled": wide.T.groupby(level="model").mean().T}
        for scope, panel in variants.items():
            for benchmark in ("calibrated", "short", "full", "mean", "ema", "donor"):
                delta = (panel.conditional - panel[benchmark]).to_numpy()
                rows.append({"scope": scope, "strategy": "conditional", "benchmark": benchmark,
                             "metric": "qlike", "block": block, "observations": len(wide),
                             **bootstrap_difference(delta.mean(), delta[ix].mean(axis=1))})
    return pd.DataFrame(rows)


def independent_checks(basket, market, experiment, portfolio, paths, calibration):
    context, predicted, depth, tail, folds, shrinkage, donor, gates = experiment
    schedule, risk, _, _, base = context[:5]
    assert (folds.last_train_label < folds.first_signal_row).all()
    assert (tail[-1].last_train_label < tail[-1].first_signal_row).all()
    previous = shrinkage.groupby(["year", "model"]).last_validation_label.max().reset_index()
    assert previous.merge(folds, on=["year", "model"]).eval("last_validation_label < first_signal_row").all()
    t = schedule.signal_row.to_numpy()
    assert np.all(donor[donor >= 0] <= t[donor >= 0] - 60)
    assert schedule.label_row.eq(schedule.signal_row + HORIZON + 1).all()
    for channel in ("error", "price"):
        assert gates[f"{channel}_capped_gate"].le(gates[f"{channel}_original_gate"]).all()
        assert gates[f"{channel}_retained"].ge(.5 - 1e-12).all()
    count = 0
    for row in schedule.itertuples():
        if row.label_row >= len(market.opening):
            continue
        prices = market.opening.iloc[row.signal_row + 1:row.label_row + 1, :4].to_numpy()
        weights = base[row.signal_row, :4]
        use = weights > 0
        growth = np.prod((prices[1:, use] / prices[:-1, use]) ** (weights[use] / weights.sum()), axis=1)
        wealth = np.r_[1, growth.cumprod()]
        max_depth = np.max(1 - wealth / np.maximum.accumulate(wealth))
        np.testing.assert_allclose(depth[row.Index], max_depth, atol=1e-13, rtol=0)
        np.testing.assert_allclose(risk.realized[row.Index], np.mean(np.log(growth) ** 2), atol=1e-13, rtol=0)
        count += 1
    lookup = portfolio.query("year == 'all' and one_way_bps == 5").set_index(["phase", "strategy"])
    for pair, path in paths.items():
        net = path.nav.to_numpy()
        dd = 1 - net / np.maximum.accumulate(np.maximum(net, 1))
        truth = [net[-1] ** (PER_YEAR / len(net)) - 1, path.net_return.std() * np.sqrt(PER_YEAR),
                 -max(dd), np.sort(dd)[-int(np.ceil(.05 * len(dd))):].mean()]
        np.testing.assert_allclose(lookup.loc[pair, list(METRICS)].astype(float), truth, atol=1e-12, rtol=0)
    matched = calibration[calibration.strategy.str.startswith("matched_")]
    assert matched.matching_gap.abs().max() < 1e-6
    archive = pd.read_csv(ROOT / "results/20261116-195751/weekly_forecasts.csv", parse_dates=["signal_date"])
    current = schedule.assign(calibrated=predicted.calibrated, full=predicted.full)
    merged = current.merge(archive.query("basket == @basket and phase in ['development','replay']"),
                           on=["phase", "signal_date"], validate="one_to_one")
    assert len(merged) == schedule.phase.isin(["development", "replay"]).sum()
    np.testing.assert_allclose(merged[["calibrated", "full"]],
                               merged[["forecast_calibrated", "forecast_joint_full_paired"]], rtol=1e-10, atol=1e-14)
    return {"labels_each_type": count, "account_days_5bp": sum(len(p) for p in paths.values()),
            "strategy_phase_metric_checks": len(paths), "forecast_fold_checks": len(folds) + len(tail[-1]),
            "archived_forecasts_checked": 2 * len(merged), "max_development_vol_gap": float(matched.matching_gap.abs().max()),
            "limited_writes": {c: int((gates[f"{c}_capped_gate"] < gates[f"{c}_original_gate"]).sum())
                               for c in ("error", "price")}}


def check_future(market, original):
    cut = pd.Timestamp("2025-06-30")
    suffix = market.close.index > cut
    impact = 1 + .015 * np.sin(np.arange(suffix.sum())[:, None] + np.arange(3))
    opening, close = market.opening.copy(), market.close.copy()
    opening.loc[suffix, CODES[:3]] *= impact
    close.loc[suffix, CODES[:3]] *= impact
    changed = risk_forecasts(Market(opening, close, market.volume))
    keep = original[0][0].signal_date.le(cut)
    np.testing.assert_allclose(original[1][keep], changed[1][keep], rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(original[3][0][keep], changed[3][0][keep], rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(original[3][2], changed[3][2], rtol=0, atol=1e-12)
    before = original[0][0].signal_row[keep].to_numpy()
    np.testing.assert_array_equal(original[0][4][before], changed[0][4][before])
    np.testing.assert_array_equal(original[6][keep], changed[6][keep])
    return {"cutoff": cut.date().isoformat(), "checked_signals": int(keep.sum())}


def plot_risk(folder, performance):
    from report_figures import BLUE, ORANGE, PRIMARY, clean, fonts
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fonts()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    data = performance.query("phase == 'replay' and year == 'all' and one_way_bps == 5")
    data = data.set_index(["basket", "strategy"])
    baskets = ["original", "style", "sector"]
    labels = ["原组合", "宽基／红利", "行业组"]
    models = [("calibrated", "校准 EWMA", ORANGE), ("conditional", "条件读取", PRIMARY),
              ("capped_tail", "限幅＋下行", BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4))
    fig.subplots_adjust(left=.115, right=.98, top=.83, bottom=.22, wspace=.40)
    for ax, metric, title in zip(axes, ("cdar95", "max_drawdown", "cagr"),
                                 ("CDaR95", "最大回撤深度", "净年化收益")):
        all_values = []
        for offset, (method, name, color) in zip((-.24, 0, .24), models):
            values = np.array([data.loc[(b, f"matched_{method}_75"), metric] for b in baskets]) * 100
            if metric == "max_drawdown":
                values = -values
            all_values.extend(values)
            bars = ax.barh(np.arange(3) + offset, values, height=.19, color=color, label=name)
            ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=3, fontsize=9)
        ax.set(yticks=np.arange(3), yticklabels=labels,
               xlim=(min(0., 1.35 * min(all_values)), 1.31 * max(all_values)))
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        clean(ax, title)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=3, frameon=False,
               bbox_to_anchor=(.5, .045))
    fig.savefig(folder / "risk_control.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_drawdown_intervals(folder, intervals):
    from report_figures import BLUE, GRAY, PRIMARY, clean, fonts
    import matplotlib.pyplot as plt

    fonts()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    data = intervals.query("block == 20 and strategy == 'matched_conditional_75' "
                           "and benchmark == 'matched_calibrated_75'")
    scopes = ["original", "style", "sector", "pooled"]
    labels = ["原组合", "宽基／红利", "行业组", "同期篮子均值"]
    fig, ax = plt.subplots(figsize=(11.8, 2.75))
    fig.subplots_adjust(left=.20, right=.95, top=.79, bottom=.26)
    rows = data.query("metric == 'cdar95'").set_index("scope").loc[scopes]
    for y, row in enumerate(rows.itertuples()):
        color = BLUE if row.Index == "pooled" else PRIMARY
        ax.hlines(y, 100 * row.lower, 100 * row.upper, color=color, lw=1.6)
        ax.plot(100 * row.difference, y, "o", color=color, ms=5)
    ax.axvline(0, color=GRAY, lw=1, ls="--")
    ax.set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels) - .4, -.6),
           xlabel="条件读取 − 校准 EWMA 的 CDaR95 差（百分点，负值表示回撤较浅）")
    clean(ax, "条件读取的尾部回撤差与 95% 区块区间")
    fig.savefig(folder / "drawdown_intervals.png", dpi=200, facecolor="white")
    plt.close(fig)


def run(folder):
    markets, quality, mapping, hashes = transfer_data()
    if set(markets) != {"original", "style", "sector"}:
        raise ValueError("本轮需要三个共同日期篮子")
    collected, paths, audits = {}, {}, {}
    folder.mkdir(parents=True, exist_ok=False)

    def record(name, table, basket):
        collected.setdefault(name, []).append(table.assign(basket=basket))

    for basket, market in markets.items():
        print(f"{basket} | 年度预测、风险校准与账户回放", flush=True)
        experiment = risk_forecasts(market)
        context, point, depth, tail, folds, shrinkage, donor, gates = experiment
        schedule, risk, _, _, _, periods, _ = context
        probability, truth_level, _, alerts, tail_folds = tail
        targets, calibration = target_books(market, context, point, alerts, frozen=basket == "original")
        metrics, books = evaluate_sleeves(market, targets, {}, periods)
        audits[basket] = independent_checks(basket, market, experiment, metrics, books, calibration)
        if basket == "original":
            frozen = pd.read_csv(ROOT / "results/20261116-163605/portfolio_metrics.csv")
            base = frozen.query("year == 'all' and one_way_bps == 5 and strategy in ['full_75','full_90']")
            original = metrics.query("year == 'all' and one_way_bps == 5 and strategy in ['frozen_75','frozen_90']")
            np.testing.assert_allclose(base.sort_values(["phase", "strategy"])[list(METRICS)],
                                       original.sort_values(["phase", "strategy"])[list(METRICS)], atol=1e-11, rtol=0)
            audits[basket]["frozen_metrics_checked"] = len(original)
        if basket == "sector":
            audits[basket]["future_perturbation"] = check_future(market, experiment)
        paths[basket] = books
        scores, losses = prediction_scores(schedule, risk, point)
        record("forecast_scores", scores, basket)
        record("forecast_losses", losses, basket)
        record("portfolio_metrics", metrics, basket)
        record("risk_calibration", calibration, basket)
        record("tail_scores", score_tail(schedule, depth, tail), basket)
        record("shock_windows", scenario_evidence(market, context, books), basket)
        record("folds", folds, basket)
        record("tail_folds", tail_folds, basket)
        record("shrinkage", shrinkage, basket)
        weekly = pd.concat([schedule, risk, point.add_prefix("forecast_"), probability.add_prefix("prob_")], axis=1)
        weekly = weekly.assign(drawdown_label=depth, tail_threshold=truth_level, donor_signal_row=donor)
        weekly.to_csv(folder / f"weekly_{basket}.csv.gz", index=False)
        gates.to_csv(folder / f"slow_writes_{basket}.csv.gz")
        pd.concat(books, names=["phase", "strategy"]).to_csv(folder / f"accounts_{basket}.csv.gz")
        print(f"{basket} | 核对完成，{audits[basket]['labels_each_type']} 个标签时点", flush=True)

    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
    performance = tables["portfolio_metrics"]
    print("按共同日期计算配对区间", flush=True)
    tables["intervals"] = pd.concat([forecast_intervals(tables["forecast_losses"]), economic_intervals(paths)], ignore_index=True)
    tables["gates"], decision = gate_results(performance)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False, float_format="%.12g")

    economics = performance.query("year == 'all'")[["basket", "phase", "strategy", "one_way_bps", "trading_days", *METRICS,
                                                    "annual_turnover_two_way", "mean_cash_weight", "mean_equity_weight",
                                                    "mean_gold_weight", "fee_over_period_start_nav"]]
    economics.to_csv(folder / "economics.csv", index=False, float_format="%.12g")
    main_scores = tables["forecast_scores"].query("subset in ['all','confirmed_under','quiet']")
    intervals = tables["intervals"]
    intervals = intervals[(intervals.metric.eq("qlike") & intervals.block.eq(4))
                          | (intervals.metric.ne("qlike") & intervals.block.eq(20))]
    parts = tables["shock_windows"]
    scenarios = parts.groupby(["basket", "scenario", "strategy"], as_index=False).agg(
        windows=("start", "size"), mean_return=("net_return", "mean"), mean_drawdown=("max_drawdown", "mean"),
        mean_low_exposure_days=("guarded_days", "mean"))
    protocol = (ROOT.parent / "RESEARCH_GOAL.md").read_text(encoding="utf-8")
    registered = protocol.split("## 2026-11-21 持续目标：用条件记忆控制深回撤", 1)[1].split("\n## ", 1)[0].strip()

    def rows(frame):
        return json.loads(frame.to_json(orient="records", double_precision=12))

    evidence = {"protocol_sha256": sha256(registered.encode()).hexdigest(), "freeze_after": str(FREEZE.date()),
                "input_hashes": hashes | dict(zip(quality.symbol, quality.sha256)), "baskets": mapping,
                "input_first": str(markets["original"].close.index[0].date()),
                "input_last": str(markets["original"].close.index[-1].date()),
                "prospective_sessions": int((markets["original"].close.index > FREEZE).sum()),
                "decision": decision, "audit": audits, "gates": rows(tables["gates"]),
                "forecast_scores": rows(main_scores[["basket", "phase", "subset", "model", "weeks", "qlike"]]),
                "tail_scores": rows(tables["tail_scores"]), "scenarios": rows(scenarios),
                "risk_calibration": rows(tables["risk_calibration"].query("strategy.str.endswith('_75')", engine="python")),
                "intervals": rows(intervals)}
    (folder / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_risk(folder, performance)
    plot_drawdown_intervals(folder, tables["intervals"])
    print(f"结果目录 {folder}\n{decision}", flush=True)


if __name__ == "__main__":
    now = datetime.now(timezone(timedelta(hours=8)))
    run(ROOT / "results" / now.strftime("%Y%m%d-%H%M%S"))
