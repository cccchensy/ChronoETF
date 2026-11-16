"""预设小规模实验：完整 T-MAVE、分组风险约束、回补与现金情景。"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backtest import CODES, PER_YEAR, check_engine, drawdown, load_market, simulate
from horizon import holm_adjust
from risk import VAR_FLOOR
from sleeves import build_factors, evaluate_sleeves
from strategy import (allocations, annotate_account, base_portfolio, check_allocation,
                      check_clocks, circular_blocks, bootstrap_difference, performance_stats)
from tmave import cdar, period_ranges
from volatility_policy import (holding_schedule, risk_inputs, fit_sleeve_forecasts,
                               risk_books, risk_multiplier, forecast_scores, tail_stats)

FREEZE_DATE = "2026-11-16"
METHODS = ("full", "ewma", "fixed", "trade")
LEVELS = (75, 90)
VARIANTS = ("split_half", "split", "recovery", "combined")
EVALUATED = (*VARIANTS, "riskcal")


def group_inputs(market, schedule, book, original, group):
    """股票组采用信号日固定权重；黄金组单独投影。不使用未来组内权重。"""
    columns = slice(0, 3) if group == 0 else slice(3, 4)
    close = np.log(market.close.iloc[:, columns]).diff().to_numpy()
    opening = np.log(market.opening.iloc[:, columns]).diff().to_numpy()
    projected = book.copy()
    projected[:, :4] = 0
    projected[:, columns] = book[:, columns]
    rows = []
    for item, w in zip(schedule.itertuples(), book[:, columns]):
        w = w / w.sum()
        t = item.signal_row
        daily = close[:t + 1] @ w
        anchor = max(np.nanmean(daily[-60:] ** 2), VAR_FLOOR)
        ewma = pd.Series(daily ** 2).ewm(alpha=.06, adjust=False, min_periods=20, ignore_na=True).mean().iat[-1]
        following = opening[t + 2:t + 7] @ w
        realized = float(np.mean(following ** 2)) if len(following) == 5 else np.nan
        rows.append((anchor, np.nanmean(daily[-5:] ** 2), np.nanmean(daily[-20:] ** 2),
                     ewma, np.nanmean(np.minimum(daily[-20:], 0) ** 2),
                     original.correlation_gap.iloc[item.Index], realized))
    return pd.DataFrame(rows, columns=original.columns), projected


def covariance_pair(close, row, equity_ratio, variances):
    history = np.log(close.iloc[max(0, row - 60):row + 1]).diff().iloc[1:].to_numpy()
    returns = np.column_stack((history[:, :3] @ equity_ratio, history[:, 3], history[:, 4]))
    pair = []
    for window in (60, 20):
        sample = returns[-window:]
        sample = sample[np.isfinite(sample).all(axis=1)]
        corr = np.corrcoef(sample, rowvar=False)
        corr = .9 * corr + .1 * np.eye(3)
        # Both risk estimates use the same T-MAVE group variances, but different correlations/bond variance.
        vol = np.sqrt(np.r_[variances, max(np.mean(sample[:, 2] ** 2), VAR_FLOOR)])
        cov = corr * np.outer(vol, vol)
        assert np.linalg.eigvalsh(cov).min() > -1e-12
        pair.append(cov)
    return pair


def allocate_groups(original, upper, covariance, previous):
    """近似原基础配置，两个协方差口径均不超过共同缩仓的预测风险。"""
    cap = np.array([original @ s @ original for s in covariance])
    bond = original[2]

    def objective(x):
        return np.sum(((x - upper) / .3) ** 2) + .05 * np.sum(np.abs(x - previous)) / .3

    def constraints(x):
        w = np.r_[x, bond]
        return np.array([1 - (w @ s @ w) / max(c, VAR_FLOOR) for s, c in zip(covariance, cap)])

    result = minimize(objective, original[:2], method="SLSQP", bounds=list(zip(np.zeros(2), upper)),
                      constraints={"type": "ineq", "fun": constraints},
                      options={"maxiter": 100, "ftol": 1e-11})
    x = result.x if result.success and constraints(result.x).min() >= -1e-7 else original[:2]
    assert constraints(x).min() >= -1e-7
    return np.r_[x, bond], bool(result.success)


def build_experiment(market):
    dates = market.close.index
    periods, fresh = period_ranges(dates, FREEZE_DATE)
    schedule = holding_schedule(dates, periods)
    ids = schedule.signal_row.to_numpy()
    raw = allocations(market.close, ids)
    base = base_portfolio(market, raw["budget_60"])
    book = base[ids]
    factors = build_factors(market.close)
    inputs = risk_inputs(market, schedule, book)
    forecasts, states, folds = fit_sleeve_forecasts(schedule, inputs, book, factors)
    folds["basket"] = "joint"
    all_folds, all_scores = [folds], [forecast_scores(schedule, inputs.realized, forecasts).assign(basket="joint")]
    groups = []
    for g in range(2):
        group_risk, group_book = group_inputs(market, schedule, book, inputs, g)
        prediction, _, group_folds = fit_sleeve_forecasts(schedule, group_risk, group_book, factors)
        groups.append(prediction)
        all_folds.append(group_folds.assign(basket=("equity", "gold")[g]))
        all_scores.append(forecast_scores(schedule, group_risk.realized, prediction).assign(basket=("equity", "gold")[g]))
    use = schedule.phase.ne("formation")
    active = schedule.loc[use].reset_index(drop=True)
    overlays, calibration, scales = risk_books(active, forecasts.loc[use].reset_index(drop=True), base)
    targets = {f"full_{level}": overlays[f"full_{level}"] for level in LEVELS}
    targets["budget_60"] = base
    decision_rows = []
    valid = schedule.index[use]
    for level in LEVELS:
        for method in METHODS:
            # Same original full-policy benchmark for all controls; only group forecast source changes.
            variants = VARIANTS if method == "full" else ("split",)
            for variant in variants:
                targets[f"{method}_{variant}_{level}"] = base.copy()
            previous = {v: np.zeros(2) for v in variants}
            prior_recovery = None
            goal = calibration.query("level == @level and model == 'full'").goal_annualized.iloc[0] / np.sqrt(PER_YEAR)
            for k, i in enumerate(valid):
                t = ids[i]
                upper = np.array([book[i, :3].sum(), book[i, 3]])
                baseline = targets[f"full_{level}"][t]
                original = np.array([baseline[:3].sum(), baseline[3], baseline[4]])
                ratio = book[i, :3] / upper[0]
                covariances = covariance_pair(market.close, t, ratio, [groups[g].loc[i, method] for g in range(2)])
                split, success = allocate_groups(original, upper, covariances, previous.get("split", np.zeros(2)))
                full_scale = float(scales.loc[k, f"full_{level}"])
                trend = float(np.log(market.close.iloc[t, :4] / market.close.iloc[t - 60, :4]) @ (book[i, :4] / book[i, :4].sum()))
                forecast = forecasts.loc[i, "full"]
                blend = .75 * forecast + .25 * min(forecast, inputs.loc[i, "var20"]) if trend > 0 else forecast
                recovery = float(risk_multiplier(goal, blend))
                if prior_recovery is not None and recovery > full_scale:
                    recovery = max(full_scale, min(recovery, prior_recovery + .10))
                prior_recovery = recovery
                for variant in variants:
                    w = original.copy()
                    if variant in ("split", "combined"):
                        w = split.copy()
                    elif variant == "split_half":
                        w = .5 * (original + split)
                    if variant == "recovery":
                        w[:2] = upper * recovery
                    elif variant == "combined":
                        w[:2] = np.minimum(upper, w[:2] * recovery / full_scale)
                    key = f"{method}_{variant}_{level}"
                    targets[key][t, :3] = ratio * w[0]
                    targets[key][t, 3] = w[1]
                    previous[variant] = w[:2]
                    decision_rows.append({"phase": active.phase.iloc[k], "strategy": key, "signal_date": dates[t],
                                          "equity": w[0], "gold": w[1], "bond": w[2], "full_scale": full_scale,
                                          "recovery_scale": recovery, "trend60": trend, "solver_success": success,
                                          "risk_ratio60": np.sqrt(w @ covariances[0] @ w / (original @ covariances[0] @ original)),
                                          "risk_ratio20": np.sqrt(w @ covariances[1] @ w / (original @ covariances[1] @ original))})
    calibration_rows = []
    start, end = periods["development"]
    for level in LEVELS:
        benchmark, candidate = f"full_{level}", f"full_split_{level}"
        limits = {fee: performance_stats(simulate(market, targets[benchmark], benchmark, start, end, fee)) for fee in (5, 10)}
        chosen, selected = targets[benchmark].copy(), np.nan
        for multiplier in np.linspace(.6, 1., 17):
            trial = targets[candidate].copy()
            trial[:, :4] *= multiplier
            passes = True
            for fee in (5, 10):
                actual = performance_stats(simulate(market, trial, candidate, start, end, fee))
                reference = limits[fee]
                ok = (actual["annual_vol"] <= .98 * reference["annual_vol"]
                      and -actual["max_drawdown"] <= -.98 * reference["max_drawdown"]
                      and actual["cdar95"] <= .98 * reference["cdar95"])
                passes &= ok
                calibration_rows.append({"level": level, "model": "riskcal", "multiplier": multiplier,
                                         "one_way_bps": fee, "risk_passes": ok, "development_cagr": actual["cagr"]})
            if passes:
                chosen, selected = trial, multiplier
        targets[f"full_riskcal_{level}"] = chosen
        calibration_rows.append({"level": level, "model": "riskcal_selected", "multiplier": selected})
    calibration = pd.concat([calibration, pd.DataFrame(calibration_rows)], ignore_index=True)
    for weights in targets.values():
        assert np.isfinite(weights).all() and (weights >= 0).all() and (weights.sum(axis=1) <= 1 + 1e-10).all()
        np.testing.assert_array_equal(weights[:, 4], base[:, 4])
    all_folds = pd.concat(all_folds, ignore_index=True)
    all_folds["first_signal_date"] = dates[all_folds.first_signal]
    all_folds["last_label_end_open"] = dates[all_folds.last_allowed_label_row]
    assert all_folds.last_label_end_open.lt(all_folds.first_signal_date).all()
    evidence = schedule.copy()
    for name, prediction in zip(("joint", "equity", "gold"), (forecasts, *groups)):
        for method in METHODS:
            evidence[f"{name}_{method}_variance"] = prediction[method]
    for mode, values in states.items():
        evidence[f"{mode}_fast"] = values[:, 0]
        evidence[f"{mode}_anchor"] = values[:, 1]
    return targets, periods, fresh, pd.DataFrame(decision_rows), all_folds, pd.concat(all_scores), evidence, calibration


def compare_gate(metrics, phase, candidate, benchmark):
    rows = []
    for fee in (5, 10):
        scope = metrics.query("phase == @phase and year == 'all' and one_way_bps == @fee").set_index("strategy")
        a, b = scope.loc[candidate], scope.loc[benchmark]
        rows.append({"phase": phase, "strategy": candidate, "benchmark": benchmark, "one_way_bps": fee,
                     "cagr_gain": a.cagr - b.cagr, "vol_change": a.annual_vol - b.annual_vol,
                     "drawdown_depth_change": b.max_drawdown - a.max_drawdown, "cdar_change": a.cdar95 - b.cdar95,
                     "passes": bool(a.cagr >= b.cagr + .003 and a.annual_vol <= b.annual_vol + 1e-12
                                    and a.max_drawdown >= b.max_drawdown - 1e-12 and a.cdar95 <= b.cdar95 + 1e-12)})
    return rows


def select(metrics):
    gates, decisions = [], []
    for level in LEVELS:
        benchmark = f"full_{level}"
        eligible = []
        for variant in EVALUATED:
            key = f"full_{variant}_{level}"
            development = compare_gate(metrics, "development", key, benchmark)
            replay = compare_gate(metrics, "replay", key, benchmark)
            gates.extend(development + replay)
            if all(r["passes"] for r in development):
                eligible.append((development[0]["cagr_gain"], key, all(r["passes"] for r in replay)))
        chosen = max(eligible, default=(0, benchmark, True))
        decisions.append({"level": level, "development_choice": chosen[1], "retained_policy": chosen[1] if chosen[2] else benchmark,
                          "candidate_count": len(EVALUATED), "development_pass_count": len(eligible),
                          "status": "retain_baseline" if not eligible or not chosen[2] else "historical_candidate_only"})
    return pd.DataFrame(gates), pd.DataFrame(decisions)


def paired_tests(paths):
    def stats(x):
        nav = np.cumprod(1 + x, axis=-1)
        return {"cagr": np.expm1(PER_YEAR * np.log1p(x).mean(axis=-1)),
                "annual_vol": x.std(axis=-1, ddof=1) * np.sqrt(PER_YEAR),
                "max_drawdown": (nav / np.maximum.accumulate(np.maximum(nav, 1), axis=-1) - 1).min(axis=-1),
                "cdar95": cdar(x)}
    rows = []
    rng = np.random.default_rng(42)
    for phase in ("development", "replay"):
        for level in LEVELS:
            benchmark = f"full_{level}"
            b = paths[phase, benchmark].net_return.to_numpy()
            for block in (10, 20, 40):
                ix = circular_blocks(rng, len(b), block)
                bs, br = stats(b), stats(b[ix])
                for variant in EVALUATED:
                    key = f"full_{variant}_{level}"
                    a = paths[phase, key].net_return.to_numpy()
                    aa, ar = stats(a), stats(a[ix])
                    for metric in aa:
                        rows.append({"phase": phase, "level": level, "strategy": key, "benchmark": benchmark,
                                     "block": block, "metric": metric,
                                     **bootstrap_difference(aa[metric] - bs[metric], ar[metric] - br[metric], True)})
    result = pd.DataFrame(rows)
    for _, part in result.groupby(["phase", "block"]):
        result.loc[part.index, "p_holm"] = holm_adjust(part.p_raw.to_numpy())
    return result


def cash_scenarios(market, targets, periods):
    rows, accounts = [], []
    for phase, (start, end) in periods.items():
        for level in LEVELS:
            for rate in (0., .01, .02):
                for fee in (5, 10, 25):
                    key = f"full_{level}"
                    path = simulate(market, targets[key], key, start, end, fee, cash_annual_rate=rate)
                    annotate_account(market, path)
                    rows.append({"phase": phase, "strategy": key, "cash_assumed_rate": rate, "one_way_bps": fee,
                                 "status": "hypothetical_constant_net_rate", **performance_stats(path)})
                    if fee == 5:
                        accounts.append(path.assign(phase=phase, strategy=key, cash_assumed_rate=rate).reset_index())
    return pd.DataFrame(rows), pd.concat(accounts, ignore_index=True)


def write_report(folder, now, metrics, choices, gates, cash, paths):
    from report import returns_plot, fonts
    fonts()
    names = {"full": "完整 T-MAVE", "ewma": "EWMA", "fixed": "固定双记忆", "trade": "仅交易日"}
    variants = {"split_half": "半幅分组", "split": "分组配置", "recovery": "分步回补", "combined": "分组＋回补", "riskcal": "风险校准"}
    labels = {}
    for _, key in paths:
        if key == "budget_60":
            labels[key] = "固定 60 日风险预算"
        else:
            method, rest = key.split("_", 1)
            variant, level = rest.rsplit("_", 1) if "_" in rest else ("", rest)
            labels[key] = f'{names[method]} · {variants.get(variant, "原版")} {level}'
    for level in LEVELS:
        keys = [f"full_{level}", *[f"full_{v}_{level}" for v in ("split", "recovery", "combined", "riskcal")]]
        returns_plot(folder, paths, keys, f"comparison_{level}.png", f"T-MAVE 收益增强 · {level}",
                     "历史回放；单边 5 bp；保留未达标候选", labels=labels, period="replay")
    for key in labels:
        returns_plot(folder, paths, [key], f"returns_{key}.png", key, "历史回放；单边 5 bp", labels=labels, period="replay")
    view = metrics.query("phase == 'replay' and year == 'all' and one_way_bps == 5")
    table = view[["strategy", "cagr", "annual_vol", "max_drawdown", "cdar95", "mean_cash_weight"]].copy()
    for col in table.columns[1:]:
        table[col] = table[col].map(lambda x: f"{x:.2%}")
    links = " · ".join(f'<a href="{p.name}">{p.name}</a>' for p in sorted(folder.glob("*.csv")) + sorted(folder.glob("*.json")))
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChronoETF 收益增强研究</title><style>body{{max-width:1120px;margin:36px auto;padding:0 20px;font:16px/1.7 system-ui;color:#172235}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow:auto}}img{{width:100%;height:auto}}a{{color:#126080}}</style>
<h1>完整 T-MAVE 下的收益增强</h1><p>{now:%Y-%m-%d %H:%M} 北京时间。行情截止 2026-10-30。2024–2026 是已查看历史回放，独立期为 0 个交易日。</p>
<p>在开发期选择候选，回放只作复核。收益提高 0.3 个百分点且波动、最大回撤深度和 CDaR 均不恶化，5/10 bp 都需满足；25 bp 为成本压力。现金利息仅为假设情景。</p>
<h2>最终保留策略</h2><div class="scroll">{choices.to_html(index=False)}</div><h2>历史回放</h2><div class="scroll">{table.to_html(index=False)}</div>
<img src="comparison_75.png" alt="强控波与候选收益回撤"><img src="comparison_90.png" alt="温和控波与候选收益回撤">
<h2>完整机制与边界</h2><p>因子经有界变换形成状态输入；快状态响应冲击，长期锚点在重复冲击或权益共振确认时写入。交易日、自然日和滞后一日事件时钟共同推进先流动后写入。逐年训练五日方差读出；新分组风险矩阵保持半正定，周末信号下一开盘执行，不加杠杆，国债目标不变。</p>
<p>约束预测风险不能保证实际风险。EWMA、固定双记忆、仅交易日对照使用相同配置算法。分组全幅、半幅、回补、组合及第二阶段风险校准候选均报告，未通过者不替换冻结版。风险校准仅用开发期选中 0.90 / 0.95 股金倍率；其回放收益差分别约为 +0.05 / +0.20 个百分点，但开发期收益均略低于原版。全部统计为探索性证据。</p>
<h2>全部证据</h2><p>{links}</p></html>'''
    (folder / "report.html").write_text(body, encoding="utf-8")


def run_enhancement(folder, now):
    check_engine()
    check_clocks()
    check_allocation()
    market, _, hashes = load_market()
    targets, periods, fresh, decisions, folds, scores, evidence, calibration = build_experiment(market)
    metrics, paths = evaluate_sleeves(market, targets, {}, periods)
    metrics = metrics.merge(tail_stats(paths), how="left", on=["phase", "strategy", "year", "one_way_bps"], validate="one_to_one")
    gates, choices = select(metrics)
    tests = paired_tests(paths)
    cash, cash_daily = cash_scenarios(market, targets, periods)
    windows = []
    for (phase, key), path in paths.items():
        rolling = np.expm1(np.log1p(path.net_return).rolling(63).sum())
        windows.append({"phase": phase, "strategy": key, "worst_63d": rolling.min(), "worst_63d_end": rolling.idxmin()})
    folder.mkdir(parents=True, exist_ok=False)
    for name, table in {"portfolio_metrics": metrics, "selection": choices, "acceptance_gates": gates,
                        "paired_tests": tests, "cash_scenarios": cash, "cash_daily": cash_daily,
                        "decisions": decisions, "train_folds": folds, "forecast_scores": scores,
                        "weekly_forecasts": evidence, "calibration": calibration, "rolling_windows": pd.DataFrame(windows),
                        "daily_backtest": pd.concat(paths, names=["phase", "strategy"]).reset_index()}.items():
        table.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    build_factors(market.close).frame.drop(columns=["forward_return_5", "forward_downside_5"]).reset_index().to_csv(folder / "tmave_states.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    (folder / "manifest.json").write_text(json.dumps({"freeze_date": FREEZE_DATE, "fresh_sessions": int(fresh),
                                                    "input_hashes": hashes, "protocol": "RESEARCH_GOAL.md",
                                                    "protocol_sha256": sha256((Path(__file__).resolve().parent.parent / "RESEARCH_GOAL.md").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
                                                    "selection": choices.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(folder, now, metrics, choices, gates, cash, paths)
    print(choices.to_string(index=False))
    print(metrics.query("phase == 'replay' and year == 'all' and one_way_bps == 5")[["strategy", "cagr", "annual_vol", "max_drawdown"]].to_string(index=False))
    print(f"结果目录 {folder}")
