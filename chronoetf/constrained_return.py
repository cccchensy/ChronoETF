"""在冻结强控波的风险上限内检验分组、现金工具和分步回补。"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BOOTSTRAP_REPS, BOOTSTRAP_SEED, CODES, drawdown, simulate
from cash_sleeve import CAPITAL, LOT, MIN_COMMISSION, load_cash, simulate_cash
from drawdown_research import FREEZE as CONDITIONAL_FREEZE
from drawdown_research import METRICS, memory_inputs, risk_stats
from enhancement import allocate_groups, covariance_pair, group_inputs
from horizon import holm_adjust
from residual_policy import forecast, prepare
from sleeves import build_factors
from strategy import bootstrap_difference, circular_blocks, performance_stats
from tmave import FEES, period_ranges, scenario_windows
from transfer_research import transfer_data
from volatility_policy import calibrate_goal, fit_sleeve_forecasts, risk_multiplier


ROOT = Path(__file__).resolve().parent
FREEZE = pd.Timestamp("2026-11-22")
METHODS = ("full", "ewma", "calibrated", "plain", "fixed", "trade")
REFERENCES = ("frozen_75", "static_dev_75", "common_conditional_75")
CASH_RATES = {"idle": 0, "etf": 2, "etf_stress": 5}
DEVELOPMENT_CASES = tuple((cash, fee) for cash in ("idle", "etf") for fee in (5, 10))
BLOCKS = (10, 20, 40)


def project(origin, proposal, covariances, caps):
    """裁剪到可行线段的最远点。起点可行，各二次型在该线段上凸。"""
    def feasible(w):
        return np.all(np.einsum("i,kij,j->k", w, covariances, w) <= caps * (1 + 1e-12))

    assert feasible(origin)
    if feasible(proposal):
        return proposal
    delta = proposal - origin
    low, high = 0., 1.
    for _ in range(40):
        mid = (low + high) / 2
        if feasible(origin + mid * delta):
            low = mid
        else:
            high = mid
    return origin + low * delta


def forecast_books(market):
    context = prepare(market, FREEZE)
    schedule, risk, _, _, base, periods, fresh = context
    factors = build_factors(market.close)
    book = base[schedule.signal_row]
    active = schedule.phase.ne("formation")
    ids = schedule.loc[active, "signal_row"].to_numpy()
    dev = schedule.loc[active, "phase"].eq("development").to_numpy()
    rows = book[active]
    grouped = np.column_stack((rows[:, :3].sum(axis=1), rows[:, 3:]))
    days = schedule.loc[active, "days_held"].to_numpy()
    capital_days = grouped[:, :2] * days[:, None]
    forecasts, _, joint_folds = fit_sleeve_forecasts(schedule, risk, book, factors)

    def scaled(variance, budget_variance):
        goal = calibrate_goal(budget_variance[dev], capital_days[dev].sum(axis=1), .75)
        return risk_multiplier(goal, variance)

    variance = forecasts.loc[active, "full"].to_numpy()
    frozen_scale = scaled(variance, variance)
    reference = grouped.copy()
    reference[:, :2] *= frozen_scale[:, None]
    static = grouped.copy()
    static[:, :2] *= np.average(np.broadcast_to(frozen_scale[dev, None], capital_days[dev].shape),
                                axis=0, weights=capital_days[dev])
    conditional_context = context
    if "independent" in periods:
        conditional_context = prepare(market, CONDITIONAL_FREEZE)
    conditional_features, _, _ = memory_inputs(conditional_context)
    prior_schedule, prior_risk = conditional_context[:2]
    prior, prior_folds, _, prior_shrinkage = forecast(prior_schedule, prior_risk, conditional_features,
                                                     readouts=("conditional",))
    assert (prior_folds.last_train_label < prior_folds.first_signal_row).all()
    boundary = prior_folds.set_index("year").first_signal_row
    assert (prior_shrinkage.last_validation_label < prior_shrinkage.year.map(boundary)).all()
    prior_by_signal = pd.concat([prior_schedule.signal_row, prior], axis=1).set_index("signal_row").loc[ids]
    conditional = grouped.copy()
    conditional[:, :2] *= scaled(prior_by_signal.conditional_paired.to_numpy(),
                                 prior_by_signal.calibrated.to_numpy())[:, None]

    inputs, predictions = [], []
    folds = [joint_folds.assign(group="joint")]
    for g, name in enumerate(("equity", "gold")):
        source, projection = group_inputs(market, schedule, book, risk, g)
        pred, _, annual = fit_sleeve_forecasts(schedule, source, projection, factors)
        folds.append(annual.assign(group=name))
        for year, year_rows in schedule[active].groupby(schedule.execution_date.dt.year):
            first = int(year_rows.signal_row.iat[0])
            mature = schedule.label_row.lt(first) & source[["ewma", "realized"]].notna().all(axis=1)
            ratio = (source.realized[mature] / source.ewma[mature]).mean()
            pred.loc[year_rows.index, "calibrated"] = source.ewma[year_rows.index] * ratio
            folds.append(pd.DataFrame([{"execution_year": year, "model": "calibrated", "first_signal": first,
                                        "last_allowed_label_row": schedule.label_row[mature].max(),
                                        "training_weeks": int(mature.sum()), "group": name, "ewma_scale": ratio}]))
        inputs.append(source.loc[active].reset_index(drop=True))
        predictions.append(pred.loc[active].reset_index(drop=True))
    folds = pd.concat(folds, ignore_index=True)
    assert (folds.last_allowed_label_row < folds.first_signal).all()
    assert all(np.isfinite(pred[list(METHODS)]).all().all() and (pred[list(METHODS)] > 0).all().all()
               for pred in predictions)
    return (schedule[active].reset_index(drop=True), base, periods, fresh, grouped, reference, static,
            conditional, inputs, predictions, folds)


def split_and_recover(half, grouped, covariances, caps, multiplier, eligible=None):
    result = half.copy()
    previous = np.zeros(2)
    for i, (w, s, c) in enumerate(zip(half, covariances, caps)):
        upper = grouped[i, :2]
        proposal = w.copy()
        if eligible is None:
            proposal[:2] *= multiplier
        else:
            add = np.minimum(np.minimum(.10 * upper, previous + .05 * upper), upper - w[:2])
            proposal[:2] += np.where(eligible[i], add, 0.)
        result[i] = project(w, proposal, s, c)
        previous = result[i, :2] - w[:2]
    return result


def risk_ok(candidate, baseline):
    return bool(candidate[1] <= baseline[1] + 1e-12 and candidate[2] >= baseline[2] - 1e-12
                and candidate[3] <= baseline[3] + 1e-12)


def simulate_book(market, target, period, fee, cash, quotes, events):
    if cash == "idle":
        return simulate(market, target, "weekly", *period, fee)
    return simulate_cash(market, target, period, fee, quotes, events, CASH_RATES[cash])


def design(market, quotes, events):
    (schedule, base, periods, fresh, grouped, reference, static, conditional, inputs,
     predictions, folds) = forecast_books(market)
    ids = schedule.signal_row.to_numpy()
    upper = grouped[:, :2]
    ratio = base[ids, :3] / upper[:, :1]
    anchors = np.column_stack([x.anchor for x in inputs])
    quiet = np.column_stack([x.var5 <= x.var20 for x in inputs])
    common = [covariance_pair(market.close, t, share, levels) for t, share, levels in zip(ids, ratio, anchors)]

    def as_target(w):
        assert (w[:, :2] <= upper + 1e-12).all()
        np.testing.assert_array_equal(w[:, 2], grouped[:, 2])
        target = base.copy()
        target[ids, :3] = ratio * w[:, :1]
        target[ids, 3] = w[:, 1]
        assert np.isfinite(target).all() and (target >= 0).all() and (target.sum(axis=1) <= 1 + 1e-12).all()
        return target

    def development(target):
        return {(cash, fee): risk_stats(simulate_book(market, target, periods["development"], fee, cash,
                                                       quotes, events).net_return.to_numpy())
                for cash, fee in DEVELOPMENT_CASES}

    targets = {key: as_target(w) for key, w in zip(REFERENCES, (reference, static, conditional))}
    limits = development(targets["frozen_75"])
    trials, choices, diagnostics, decisions = [], {}, {}, []

    def assess(target):
        scores = development(target)
        return all(risk_ok(score, limits[case]) for case, score in scores.items()), scores

    for method in METHODS:
        predicted = np.column_stack([x[method] for x in predictions])
        covariances = np.array([covariance_pair(market.close, t, share, v) + pair
                                for t, share, v, pair in zip(ids, ratio, predicted, common)])
        caps = np.einsum("ni,nkij,nj->nk", reference, covariances, reference)
        half, previous, solver_failures = reference.copy(), np.zeros(2), 0
        for i, (original, u, s, c) in enumerate(zip(reference, upper, covariances, caps)):
            preferred, success = allocate_groups(original, u, s, previous)
            previous = project(original, preferred, s, c)[:2]
            half[i, :2] = (original[:2] + previous) / 2
            solver_failures += not success

        def record(target, multiplier, arm):
            passed, scores = assess(target)
            for (cash, fee), values in scores.items():
                trials.append({"model": method, "arm": arm, "multiplier": multiplier, "cash_mode": cash,
                               "one_way_bps": fee, "passes_risk": risk_ok(values, limits[cash, fee]),
                               **dict(zip(METRICS, values))})
            return passed, scores["etf", 5][0]

        best = None
        for multiplier in np.linspace(.85, 1., 7):
            candidate = split_and_recover(half, grouped, covariances, caps, multiplier)
            target = as_target(candidate)
            passed, cagr = record(target, multiplier, "half")
            # 七档都不可行时，保留最小倍率供诊断。
            rank = (passed, cagr, -multiplier) if passed else (passed, 0., -multiplier)
            if best is None or rank > best[0]:
                best = (rank, multiplier, candidate, target, passed, cagr)
        _, multiplier, candidate, target, half_pass, half_cagr = best
        below = predicted <= anchors
        prior_below = np.vstack((np.zeros((1, 2), dtype=bool), below[:-1]))
        eligible = quiet & below & prior_below
        recovery = split_and_recover(candidate, grouped, covariances, caps, multiplier, eligible)
        recovery_target = as_target(recovery)
        recovery_pass, recovery_cagr = record(recovery_target, multiplier, "recovery")
        half_key, recovery_key = f"{method}_half", f"{method}_recovery"
        targets[half_key], targets[recovery_key] = target, recovery_target
        chosen = recovery_key if recovery_pass and (not half_pass or recovery_cagr > half_cagr) else half_key
        choices[method] = {"selected": chosen, "multiplier": multiplier, "half_development_risk": half_pass,
                           "recovery_development_risk": recovery_pass}
        increase = recovery[:, :2] - candidate[:, :2]
        applied = increase > 1e-9
        previous = np.vstack((np.zeros((1, 2)), increase[:-1]))
        assert (increase >= 0).all() and (increase <= .10 * upper + 1e-12).all()
        assert (increase <= previous + .05 * upper + 1e-12).all() and (increase[~eligible] == 0).all()
        max_risk = {key: float(np.sqrt(np.einsum("ni,nkij,nj->nk", w, covariances, w) / caps).max())
                    for key, w in (("half", candidate), ("recovery", recovery))}
        assert max(max_risk.values()) <= 1 + 1e-11
        diagnostics[method] = {"solver_failures": solver_failures, "max_risk_ratio": max_risk,
                               "recovery_weeks": {phase: {"total": int(mask.sum()),
                                                  "eligible": int(eligible[mask].any(axis=1).sum()),
                                                  "delivered": int(applied[mask].any(axis=1).sum())}
                                                  for phase in periods for mask in [schedule.phase.eq(phase).to_numpy()]}}
        decisions.append(schedule.assign(model=method, equity_forecast=predicted[:, 0],
                                         gold_forecast=predicted[:, 1], equity_anchor=anchors[:, 0],
                                         gold_anchor=anchors[:, 1], half_equity=candidate[:, 0],
                                         half_gold=candidate[:, 1], recovery_equity=recovery[:, 0],
                                         recovery_gold=recovery[:, 1], equity_eligible=eligible[:, 0],
                                         gold_eligible=eligible[:, 1]))
    return targets, periods, fresh, choices, diagnostics, pd.DataFrame(trials), pd.concat(decisions), folds


def cash_checks(path, quotes, events, fee):
    """独立按成交手数、登记日存量和前日净值复算现金账本。"""
    shares = path.cash_etf_shares
    traded = path.cash_etf_trade_shares
    np.testing.assert_array_equal(shares, traded.cumsum())
    assert (shares % LOT == 0).all() and (path.cash_weight >= -1e-12).all()
    cost = (traded.abs() * quotes.open.loc[path.index] / CAPITAL * fee / 10_000).clip(lower=MIN_COMMISSION)
    np.testing.assert_allclose(path.cash_etf_cost, cost.where(traded.ne(0), 0.), atol=1e-12, rtol=0)
    np.testing.assert_allclose(path.cost_nav, path.cash_etf_cost + path.risk_cost, atol=1e-12, rtol=0)
    accrual, payment = pd.Series(0., index=path.index), pd.Series(0., index=path.index)
    for item in events:
        record, ex, paid = (pd.Timestamp(item[key]) for key in ("record", "ex", "paid"))
        if record in path.index and ex in path.index:
            accrual.loc[ex] = shares.loc[record] * item["per_share"] / CAPITAL
            if paid in path.index:
                payment.loc[paid] = accrual.loc[ex]
    np.testing.assert_allclose(path.dividend_accrual, accrual, atol=1e-12, rtol=0)
    np.testing.assert_allclose(path.dividend_paid, payment, atol=1e-12, rtol=0)
    np.testing.assert_allclose(path.dividend_receivable, (accrual - payment).cumsum(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(path.nav, (1 + path.net_return).cumprod(), atol=1e-11, rtol=0)
    allocation = path[[f"w_{c}" for c in CODES]].sum(axis=1) + path.cash_etf_weight + path.cash_weight
    np.testing.assert_allclose(allocation + path.dividend_receivable / path.nav, 1., atol=1e-11, rtol=0)


def evaluate(market, targets, periods, quotes, events):
    records, paths, audits = [], {}, {"paths": 0, "rows": 0, "cash_paths": 0, "funding_sales": 0}
    for phase, period in periods.items():
        for method, target in targets.items():
            modes = ("idle", "etf", "etf_stress") if method in ("frozen_75", "full_half", "full_recovery") else ("idle", "etf")
            for cash in modes:
                for fee in FEES:
                    path = simulate_book(market, target, period, fee, cash, quotes, events)
                    key = (phase, method, cash, fee)
                    paths[key] = path
                    score = performance_stats(path)
                    np.testing.assert_allclose([score[m] for m in METRICS], risk_stats(path.net_return.to_numpy()),
                                               atol=1e-11, rtol=0)
                    audits["paths"] += 1
                    audits["rows"] += len(path)
                    if cash != "idle":
                        cash_checks(path, quotes, events, CASH_RATES[cash])
                        audits["cash_paths"] += 1
                        audits["funding_sales"] += int(path.cash_etf_trade_shares.lt(0).sum())
                    records.append({"phase": phase, "strategy": method, "cash_mode": cash, "one_way_bps": fee,
                                    "cash_etf_bps": CASH_RATES[cash],
                                    **{k: score[k] for k in ("trading_days", *METRICS, "annual_turnover_two_way",
                                                            "fee_over_period_start_nav", "mean_cash_weight",
                                                            "mean_equity_weight", "mean_gold_weight", "mean_bond_weight")},
                                    "mean_cash_etf_weight": path["cash_etf_weight"].mean() if cash != "idle" else 0.,
                                    "cash_etf_net_pnl_over_start_nav": float((path.cash_etf_pnl - path.cash_etf_cost).sum())
                                    if cash != "idle" else 0.})
    return pd.DataFrame(records), paths, audits


def paired_evidence(paths, scores, choices):
    selected = {method: item["selected"] for method, item in choices.items()}
    candidate = selected["full"]
    peers = [selected[method] for method in METHODS[1:]]
    benchmarks = (*peers, *REFERENCES)
    compiled = {}
    for phase in scores.phase.unique():
        if phase == "development":
            continue
        panel = scores[scores.phase.eq(phase)].set_index(["strategy", "cash_mode", "one_way_bps"])
        table = []
        returns = {key: paths[phase, key, "etf", 5].net_return.to_numpy() for key in (candidate, *benchmarks)}
        n = len(returns[candidate])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for block in BLOCKS:
            indices = circular_blocks(rng, n, block)
            sampled = {key: risk_stats(values[indices]) for key, values in returns.items()}
            block_rows = []
            for key in benchmarks:
                delta = risk_stats(returns[candidate]) - risk_stats(returns[key])
                differences = sampled[candidate] - sampled[key]
                tests = {m: bootstrap_difference(delta[j], differences[:, j], with_pvalue=j == 0)
                         for j, m in enumerate(METRICS)}
                ten = panel.loc[(candidate, "etf", 10), "cagr"] - panel.loc[(key, "etf", 10), "cagr"]
                block_rows.append({"benchmark": key, "primary_family": key in peers, "block": block,
                                   "cagr_delta_10bps": ten, **tests})
            for row, p in zip(block_rows[:len(peers)], holm_adjust([x["cagr"]["p_raw"] for x in block_rows[:len(peers)]])):
                row["cagr"]["p_holm"] = p
                row["increment_supported"] = bool(row["cagr"]["lower"] > 0 and p < .05 and row["cagr_delta_10bps"] > 0)
            table.extend(block_rows)
        risk_gates = []
        lookup = scores.set_index(["phase", "strategy", "cash_mode", "one_way_bps"])
        for arm in (candidate, *peers):
            for section in ("development", phase):
                for cash, fee in DEVELOPMENT_CASES:
                    values = lookup.loc[(section, arm, cash, fee), list(METRICS)].to_numpy()
                    cap = lookup.loc[(section, "frozen_75", cash, fee), list(METRICS)].to_numpy()
                    risk_gates.append({"strategy": arm, "phase": section, "cash_mode": cash, "one_way_bps": fee,
                                       "risk_ok": risk_ok(values, cap), "vol_delta": values[1] - cap[1],
                                       "drawdown_depth_delta": cap[2] - values[2], "cdar_delta": values[3] - cap[3]})
        gross_delta = panel.loc[(candidate, "etf", 5), "cagr"] - panel.loc[("frozen_75", "idle", 5), "cagr"]
        same_cash = panel.loc[(candidate, "etf", 5), "cagr"] - panel.loc[("frozen_75", "etf", 5), "cagr"]
        candidate_risk = all(x["risk_ok"] for x in risk_gates if x["strategy"] == candidate)
        compiled[phase] = {"comparisons": table, "risk_gates": risk_gates, "candidate": candidate,
                           "candidate_risk_ok": candidate_risk, "total_gain": gross_delta,
                           "same_cash_gain": same_cash,
                           "system_gain_gate": bool(candidate_risk and gross_delta >= .002 and same_cash > 0),
                           "algorithm_supported": bool(candidate_risk and all(x["increment_supported"]
                                                      for x in table if x["primary_family"] and x["block"] == 20))}
    return compiled


def scenarios(market, paths, periods):
    windows, _ = scenario_windows(np.log(market.close).diff(), periods)
    rows = []
    for window in windows:
        for key in ("full_half", "full_recovery"):
            path = paths["replay", key, "etf", 5].loc[window["start"]:window["end"]]
            rows.append({"scenario": window["scenario"], "strategy": key, "start": str(path.index[0].date()),
                         "net_return": (1 + path.net_return).prod() - 1, "max_drawdown": drawdown(path.net_return).min(),
                         "mean_equity_gold": path[[f"w_{c}" for c in CODES[:4]]].sum(axis=1).mean()})
    table = pd.DataFrame(rows)
    return table.groupby(["scenario", "strategy"]).agg(count=("start", "size"), net_return=("net_return", "mean"),
                                                        max_drawdown=("max_drawdown", "mean"),
                                                        mean_equity_gold=("mean_equity_gold", "mean")).reset_index()


def check_archive(scores):
    old = pd.read_csv(ROOT / "results/20261121-235427/economics.csv")
    old = old.query("basket == 'original' and strategy in ['frozen_75', 'static_dev_75', 'common_conditional_75']")
    keys = ["phase", "strategy", "one_way_bps"]
    new = scores.query("cash_mode == 'idle'").set_index(keys).loc[old.set_index(keys).index]
    np.testing.assert_allclose(new[list(METRICS)], old[list(METRICS)], atol=1e-11, rtol=0)
    return len(old)


def plot_evidence(folder, scores, evidence):
    from report_figures import BLUE, GRAY, INK, ORANGE, PRIMARY, clean, fonts
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, PercentFormatter

    fonts()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    sample = scores.query("basket == 'original' and phase == 'replay' and one_way_bps == 5")
    sample = sample.set_index(["strategy", "cash_mode"])
    arms = [("frozen_75", "idle"), ("frozen_75", "etf"), ("full_half", "etf"), ("full_recovery", "etf")]
    labels = ["冻结 · 零现金收益", "冻结 · 现金 ETF", "半幅分组 · 现金 ETF", "分步回补 · 现金 ETF"]
    colors = [GRAY, BLUE, ORANGE, PRIMARY]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 2.8))
    fig.subplots_adjust(left=.20, right=.98, top=.79, bottom=.20, wspace=.27)
    for ax, metric, title in zip(axes, ("cagr", "annual_vol", "max_drawdown"),
                                 ("净年化收益", "年化波动率", "最大回撤深度")):
        values = sample.loc[arms, metric].abs().to_numpy() * 100
        bars = ax.barh(np.arange(len(arms)), values, color=colors, height=.53)
        ax.bar_label(bars, labels=[f"{v:.3f}%" for v in values], padding=5, fontsize=10, color=INK)
        ax.set(xlim=(0, 1.35 * max(values)), yticks=np.arange(len(arms)))
        ax.set_yticklabels(labels if ax is axes[0] else [""] * len(arms))
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0 if metric == "cagr" else 1))
        clean(ax, title)
    fig.savefig(folder / "return_risk.png", dpi=200, facecolor="white")
    plt.close(fig)

    replay = evidence["panels"]["original"]["tests"]["replay"]
    checks = [x["cagr"] for x in replay["comparisons"] if x["primary_family"] and x["block"] == 20]
    labels = ["原始 EWMA", "校准 EWMA", "同输入静态读出", "固定双记忆", "仅交易日"]
    fig, ax = plt.subplots(figsize=(11.8, 2.75))
    fig.subplots_adjust(left=.19, right=.96, top=.82, bottom=.27)
    for i, row in enumerate(checks):
        ax.hlines(i, 100 * row["lower"], 100 * row["upper"], color=PRIMARY, linewidth=2.)
        ax.plot(100 * row["difference"], i, "o", color=BLUE, markersize=6)
    ax.axvline(0., color=ORANGE, linestyle="--", linewidth=1.2)
    ax.set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels) - .5, -.5),
           xlabel="完整三时钟 T-MAVE − 对照的净年化差（百分点）")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    clean(ax, "同现金规则下的五组对照 · 配对 95% 区间")
    fig.savefig(folder / "paired_return.png", dpi=200, facecolor="white")
    plt.close(fig)


def run(folder):
    markets, quality, mapping, price_hashes = transfer_data()
    assert set(markets) == {"original", "style", "sector"}
    dates = markets["original"].close.index
    assert all(m.close.index.equals(dates) for m in markets.values())
    quotes, cash = load_cash(dates)
    protocol = (ROOT.parent / "RESEARCH_GOAL.md").read_text(encoding="utf8")
    preregistration = protocol.split("## 2026-11-22 持续目标：冻结风险上限内增收", 1)[1]
    preregistration = preregistration.split("## 2026-11-22 增收实验结果", 1)[0].rstrip()
    folder.mkdir(parents=True)
    results, panels = [], {}
    for basket, market in markets.items():
        print(f"{basket} | 分组预测、开发期选型及逐周风险限制", flush=True)
        targets, periods, fresh, choices, caps, trials, decisions, folds = design(market, quotes, cash["announcements"])
        print(f"{basket} | 真实现金撮合与配对回放", flush=True)
        scores, paths, audits = evaluate(market, targets, periods, quotes, cash["announcements"])
        if basket == "original":
            audits["frozen_and_conditional_archive_rows"] = check_archive(scores)
        for table, name in ((trials, "development"), (decisions, "decisions"), (folds, "folds")):
            table.to_csv(folder / f"{name}_{basket}.csv", index=False, float_format="%.12g")
        pd.concat({key: path for key, path in paths.items() if key[-1] == 5},
                  names=["phase", "strategy", "cash_mode", "one_way_bps"]).to_csv(folder / f"accounts_{basket}.csv.gz")
        panels[basket] = {"development_selection": choices, "target_checks": caps, "audit": audits,
                          "tests": paired_evidence(paths, scores, choices),
                          "scenarios": scenarios(market, paths, periods).to_dict("records")}
        results.append(scores.assign(basket=basket))
        print(f"{basket} | 已核对 {audits['rows']:,} 行账户", flush=True)
    scores = pd.concat(results, ignore_index=True)
    scores.to_csv(folder / "economics.csv", index=False, float_format="%.12g")
    evidence = {"freeze": FREEZE.date().isoformat(), "post_freeze_sessions": int(fresh),
                "periods": {key: {"first": str(dates[start].date()), "last": str(dates[stop - 1].date()),
                                  "sessions": int(stop - start)} for key, (start, stop) in periods.items()},
                "protocol_sha256": sha256(preregistration.encode("utf8")).hexdigest(),
                "price_hashes": price_hashes, "transfer_quality": quality.to_dict("records"), "asset_mapping": mapping,
                "cash": cash, "bootstrap": {"seed": BOOTSTRAP_SEED, "reps": BOOTSTRAP_REPS, "blocks": BLOCKS},
                "panels": panels}
    (folder / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    plot_evidence(folder, scores, evidence)
    print(folder, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    cli.add_argument("--output", type=Path, default=ROOT / "results" / now)
    run(cli.parse_args().output)
