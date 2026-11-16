"""检验 T-MAVE 对重复冲击的方差预测及下一开盘风控。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import BASE_COST, BOOTSTRAP_SEED, CODES, EPS, check_engine, load_market
from horizon import holm_adjust
from risk import EVENT_QUANTILE, HORIZON, VAR_FLOOR, fit_readout, qlike, readout_inputs
from sleeves import FIT_END, FREEZE_DATE, NAMES, evaluate_sleeves, group_returns
from strategy import (
    CAPS, GROUPS, allocations, base_portfolio, bootstrap_difference, check_allocation, check_clocks,
    circular_blocks, covers_objectives, development_pick, event_evidence, flow_jump, paired_intervals, time_flow,
)
from tmave import period_ranges, weekly_signals


FLOOR = .75
ROUTES = {"cash": "留现金", "rotate": "组间轮换"}
MODELS = {"ewma": "EWMA", "fixed": "固定双记忆", "trade": "T-MAVE 交易日", "full": "T-MAVE 三时钟"}
WEEKLY_CONTROLS = {f"weekly_{name}_cash": f"{MODELS[name]} · 周频留现金" for name in ("ewma", "fixed", "trade")}
LABELS = {
    "budget_120": "固定 120 日",
    "budget_60": "固定 60 日",
    "event_rebalance": "固定 60 日 · 同事件日再平衡",
    **{f"{model}_{route}": f"{label} · {description}"
       for route, description in ROUTES.items() for model, label in {**MODELS, "weekly": "T-MAVE 周频"}.items()},
    **WEEKLY_CONTROLS,
}
CANDIDATES = tuple(f"full_{route}" for route in ROUTES)


@dataclass
class EventRisk:
    anchor: pd.DataFrame
    realized: pd.DataFrame
    forecasts: dict
    confirmed: np.ndarray
    hits: np.ndarray
    fields: dict
    folds: pd.DataFrame
    threshold: float


def fit_event_risk(market):
    dates = market.close.index
    returns = np.log(market.close).diff()
    squared = returns.pow(2)
    anchor = squared.rolling(60, min_periods=55).mean().clip(lower=VAR_FLOOR)
    short_var = squared.rolling(20, min_periods=18).mean().clip(lower=VAR_FLOOR)
    surprise = np.log(squared / anchor + .05).clip(-3, 3)
    volume = np.log(market.volume / market.volume.rolling(20, min_periods=18).median().shift())
    realized = np.log(market.opening).diff().pow(2).rolling(HORIZON).mean().shift(-HORIZON - 1)

    standardized = returns / np.sqrt(anchor.shift())
    history = dates < FIT_END
    threshold = float(np.nanquantile(np.abs(standardized.loc[history].to_numpy()), EVENT_QUANTILE))
    hit = standardized.abs().ge(threshold)
    downside = standardized.le(-threshold)
    confirmed = (downside & downside.rolling(HORIZON, min_periods=1).sum().ge(2)).to_numpy()
    hits = hit.rolling(HORIZON, min_periods=1).sum().to_numpy(dtype=int)
    shock = hit.to_numpy()
    observed = standardized.notna().to_numpy() & surprise.notna().to_numpy()
    supported = shock & volume.gt(0).to_numpy()
    repeated = shock & (hits >= 2)
    fast_gate = (.12 + .28 * shock + .10 * supported) * observed
    slow_gate = (.035 + .06 * supported + .10 * repeated) * observed
    mask = history[:, None] & observed
    fixed_fast, fixed_slow = fast_gate[mask].mean(), slow_gate[mask].mean()

    ticks = event_evidence(standardized, threshold)["event_increment"]
    average_ticks = ticks[history].sum(axis=0) / observed[history].sum(axis=0)
    elapsed = time_flow(dates, ticks, average_ticks, True, True)
    trade_elapsed = time_flow(dates, ticks, average_ticks, False, False)
    candidate = surprise.fillna(0).to_numpy()
    states = {
        "fixed": flow_jump(candidate, trade_elapsed, fixed_fast * observed, fixed_slow * observed),
        "trade": flow_jump(candidate, trade_elapsed, fast_gate, slow_gate),
        "full": flow_jump(candidate, elapsed, fast_gate, slow_gate),
    }
    common = readout_inputs(short_var, anchor, surprise, volume)
    features = {name: np.concatenate((common, state), axis=2) for name, state in states.items()}
    forecasts = {"ewma": squared.ewm(alpha=.06, adjust=False, min_periods=20, ignore_na=True).mean()}
    forecasts.update({name: pd.DataFrame(np.nan, index=dates, columns=CODES) for name in states})
    folds = []
    positions = np.arange(len(dates))
    values = realized.to_numpy()
    level = anchor.to_numpy()
    for year in np.unique(dates[~history].year):
        start = dates.searchsorted(pd.Timestamp(year, 1, 1)) - 1
        stop = dates.searchsorted(pd.Timestamp(year + 1, 1, 1)) - 1
        train = positions + HORIZON + 1 < start
        for name, x in features.items():
            output, count = fit_readout(x, level, values, train, slice(start, stop))
            forecasts[name].iloc[start:stop] = output
            folds.append({
                "execution_year": int(year), "model": name, "first_signal": dates[start],
                "last_label_end_open": dates[np.flatnonzero(train)[-1] + HORIZON + 1], "training_asset_days": count,
            })
    fields = {
        "standardized_shock": standardized.to_numpy(), "current_shock": shock,
        "short_gate": fast_gate, "long_gate": slow_gate, "event_increment": ticks,
        "elapsed_full": elapsed, "full_fast": states["full"][:, :, 0], "full_anchor": states["full"][:, :, 1],
        "fixed_fast": states["fixed"][:, :, 0], "fixed_anchor": states["fixed"][:, :, 1],
    }
    return EventRisk(anchor, realized, forecasts, confirmed, hits, fields, pd.DataFrame(folds), threshold)


def pressure_scales(book, anchor, forecast, active):
    estimate = np.where(np.isfinite(forecast), forecast, anchor)
    scale = np.ones_like(active, dtype=float)
    for j, group in enumerate(GROUPS[:2]):
        before = np.nansum(anchor[:, group] * book[:, group], axis=1)
        after = np.nansum(estimate[:, group] * book[:, group], axis=1)
        multiple = np.divide(after, before, out=np.ones_like(before), where=before > 0)
        scale[:, j] = np.where(active[:, j], np.clip(multiple ** -.5, FLOOR, 1), 1)
    return scale


def overlay_positions(book, scale, active, positive_trend, rotate):
    adjusted = book.copy()
    for j, group in enumerate(GROUPS[:2]):
        adjusted[:, group] *= scale[:, j, None]
    if rotate:
        for source, destination in ((0, 1), (1, 0)):
            original, other = GROUPS[source], GROUPS[destination]
            released = (book[:, original] - adjusted[:, original]).sum(axis=1)
            amount = book[:, other].sum(axis=1)
            room = np.maximum(0., CAPS[destination] - amount)
            permitted = ~active[:, destination] & positive_trend[:, destination] & (amount > 0)
            transfer = np.where(permitted, np.minimum(released, room), 0.)
            multiple = np.divide(transfer, amount, out=np.zeros_like(amount), where=amount > 0)
            adjusted[:, other] += book[:, other] * multiple[:, None]
    return adjusted


def design_policy(market, risk, periods):
    dates = market.close.index
    weekly = np.unique(np.concatenate([weekly_signals(dates, *bounds) for bounds in periods.values()]))
    books = allocations(market.close, weekly)
    targets = {key: base_portfolio(market, books[key]) for key in ("budget_120", "budget_60")}
    base = targets["budget_60"]
    previous = weekly[0]
    locations = np.arange(previous, len(dates))
    carry = np.zeros_like(base)
    carry[previous:] = base[weekly[np.searchsorted(weekly, locations, side="right") - 1]]

    confirmed = np.column_stack([risk.confirmed[:, group].any(axis=1) for group in GROUPS[:2]])
    active = pd.DataFrame(confirmed).rolling(HORIZON, min_periods=1).max().to_numpy(dtype=bool)
    starts = active & ~np.vstack((np.zeros((1, 2), dtype=bool), active[:-1]))
    changed = np.vstack((active[:1], active[1:] != active[:-1])).any(axis=1)
    event_orders = np.r_[False, (changed | confirmed.any(axis=1))[:-1]]
    event_orders[weekly + 1] = True
    targets["event_rebalance"] = carry
    order_days = {"event_rebalance": event_orders}
    group, _ = group_returns(market.close)
    positive_trend = group.rolling(60, min_periods=55).sum().gt(0).to_numpy()
    scales = {name: pressure_scales(carry, risk.anchor.to_numpy(), forecast.to_numpy(), active)
              for name, forecast in risk.forecasts.items()}
    for route in ROUTES:
        for name, scale in scales.items():
            key = f"{name}_{route}"
            targets[key] = overlay_positions(carry, scale, active, positive_trend, route == "rotate")
            order_days[key] = event_orders
        targets[f"weekly_{route}"] = targets[f"full_{route}"]
    for name in ("ewma", "fixed", "trade"):
        targets[f"weekly_{name}_cash"] = targets[f"{name}_cash"]

    for key, weights in targets.items():
        if key in ("budget_60", "budget_120"):
            continue
        np.testing.assert_array_equal(weights[:, 4], carry[:, 4])
        assert np.all(weights >= 0) and np.all(weights.sum(axis=1) <= 1 + 1e-10)
    return targets, order_days, carry, weekly, confirmed, active, starts, scales


def choose_policy(performance):
    development = performance.query("phase=='development' and year=='all' and one_way_bps in [5, 10]")
    dev = {bps: tab.set_index("strategy").to_dict("index") for bps, tab in development.groupby("one_way_bps")}
    outcomes = []
    for choice in CANDIDATES:
        route = choice.split("_")[1]
        for bps, scores in dev.items():
            for peer in ("budget_60", "budget_120", f"fixed_{route}"):
                outcomes.append({"candidate": choice, "cost": bps, "benchmark": peer,
                                 "joint_improvement": covers_objectives(scores[choice], scores[peer])})
    comparisons = pd.DataFrame(outcomes)
    qualified = comparisons.groupby("candidate").joint_improvement.all()
    pool = tuple(qualified[qualified].index) or CANDIDATES
    selected, frontier = development_pick(dev[BASE_COST], pool)
    comparisons["on_frontier"] = comparisons.candidate.isin(frontier)
    comparisons["selected"] = comparisons.candidate.eq(selected)
    return selected, bool(qualified[selected]), comparisons


def forecast_evidence(market, risk, periods):
    dates = market.close.index
    stage = []
    values = {name: table.to_numpy() for name, table in risk.forecasts.items()}
    realized = risk.realized.to_numpy()
    for phase, (start, end) in periods.items():
        signals = np.arange(start - 1, end - 1)
        exits = signals + HORIZON + 1
        valid = exits < end
        target = realized[signals].copy()
        target[~valid] = np.nan
        end_date = dates.take(np.minimum(exits, len(dates) - 1)).where(valid)
        rows = pd.DataFrame({
            "phase": phase, "signal_date": dates[signals].repeat(len(CODES)),
            "execution_date": dates[signals + 1].repeat(len(CODES)), "label_end_open": end_date.repeat(len(CODES)),
            "symbol": np.tile(CODES, len(signals)), "realized_var": target.ravel(),
            "confirmed_downside": risk.confirmed[signals].ravel(), "shocks_5d": risk.hits[signals].ravel(),
            "anchor_var": risk.anchor.to_numpy()[signals].ravel(),
            **{f"forecast_{k}": a[signals].ravel() for k, a in values.items()},
            **{k: a[signals].ravel() for k, a in risk.fields.items()},
        })
        matched = rows[["realized_var", *(f"forecast_{k}" for k in values)]].notna().all(axis=1)
        for name in values:
            rows[f"loss_{name}"] = qlike(rows.realized_var, rows[f"forecast_{name}"]).where(matched)
        stage.append(rows)
    return pd.concat(stage, ignore_index=True)


def forecast_comparison(evidence):
    replay_2026 = evidence.loc[evidence.phase.eq("replay") & evidence.execution_date.dt.year.eq(2026)]
    samples = pd.concat([evidence, replay_2026.assign(phase="replay_2026")])
    summary, tests = [], []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for phase, frame in samples.loc[samples.symbol.ne(CODES[4])].groupby("phase", sort=False):
        dates = pd.DatetimeIndex(frame.signal_date.unique())
        matched = frame.dropna(subset=["loss_full"])
        cases = {"all": matched, "confirmed": matched.loc[matched.confirmed_downside],
                 "quiet": matched.loc[matched.shocks_5d.eq(0)]}
        for case, group in cases.items():
            summary.extend({"phase": phase, "case": case, "forecast": name, "asset_days": len(group),
                            "sessions": group.signal_date.nunique(), "qlike": group[f"loss_{name}"].mean()}
                           for name in MODELS)
        events = cases["confirmed"]
        count = events.groupby("signal_date").size().reindex(dates, fill_value=0).to_numpy()
        for block in (10, 20, 40):
            if not len(events):
                continue
            indices = circular_blocks(rng, len(dates), block)
            counts = count[indices].sum(axis=1)
            valid = counts > 0
            for name in ("ewma", "fixed", "trade"):
                gain = events[f"loss_{name}"] - events.loss_full
                daily = gain.groupby(events.signal_date).sum().reindex(dates, fill_value=0).to_numpy()
                draws = daily[indices[valid]].sum(axis=1) / counts[valid]
                tests.append({
                    "phase": phase, "strategy": "full", "benchmark": name, "block": block, "metric": "event_qlike",
                    "observations": len(events), **bootstrap_difference(gain.mean(), draws, with_pvalue=True),
                })
    return pd.DataFrame(summary), pd.DataFrame(tests)


def compare_policies(paths, risk_tests, selected):
    route = selected.split("_")[1]
    benchmarks = ("budget_60", "budget_120", "event_rebalance", *(f"{name}_{route}" for name in ("ewma", "fixed", "trade", "weekly")))
    all_paths = dict(paths)
    for key in (selected, *benchmarks):
        all_paths[("replay_2026", key)] = paths[("replay", key)].loc[lambda p: p.index.year == 2026]
    daily = paired_intervals(all_paths, {"policy": selected, "core": "budget_60"}, benchmarks,
                            ("replay", "replay_2026", "development", "independent"),
                            drawdown_interval=True, with_pvalue=True)
    cdar = daily.metric.eq("cdar95")
    daily.loc[cdar, ["difference", "lower", "upper"]] = -daily.loc[cdar, ["difference", "upper", "lower"]].to_numpy()
    combined = pd.concat([risk_tests, daily], ignore_index=True)
    combined["primary"] = ((combined.metric.eq("event_qlike") & combined.benchmark.isin(("ewma", "fixed"))) |
                           (combined.metric.eq("net_sharpe") & combined.benchmark.isin(("budget_60", "budget_120", f"fixed_{route}"))))
    combined["p_holm"] = np.nan
    for _, rows in combined.loc[combined.primary].groupby(["phase", "block"]):
        combined.loc[rows.index, "p_holm"] = holm_adjust(rows.p_raw.to_numpy())
    return combined.assign(analysis="prespecified")


def compare_weekly(paths):
    period = ("replay", "development", "replay_2026", "independent")
    arms = ("weekly_cash", *WEEKLY_CONTROLS, "budget_60")
    data = dict(paths)
    for arm in arms:
        replay = data[("replay", arm)]
        data[("replay_2026", arm)] = replay.loc[replay.index.year == 2026]
    stats = pd.concat([
        paired_intervals(data, {"policy": "weekly_cash", "core": "budget_60"},
                         ("budget_60", *WEEKLY_CONTROLS), period, drawdown_interval=True, with_pvalue=True),
        paired_intervals(data, {"policy": "weekly_trade_cash", "core": "budget_60"},
                         ("budget_60",), period, drawdown_interval=True, with_pvalue=True),
    ], ignore_index=True)
    cdar = stats.metric.eq("cdar95")
    stats.loc[cdar, ["difference", "lower", "upper"]] = -stats.loc[cdar, ["difference", "upper", "lower"]].to_numpy()
    stats["p_holm"] = np.nan
    for _, rows in stats.loc[stats.metric.eq("net_sharpe")].groupby(["phase", "block"]):
        stats.loc[rows.index, "p_holm"] = holm_adjust(rows.p_raw.to_numpy())
    return stats.assign(primary=False, analysis="weekly_diagnostic")


def window_results(market, paths, periods, book, weekly, starts):
    dates = market.close.index
    opening = market.opening.to_numpy()
    rows = []
    for phase, (start, end) in periods.items():
        valid = np.arange(start - 1, end - HORIZON - 1)
        for offset, g in zip(*np.where(starts[valid])):
            signal = valid[offset]
            enter, exit_ = signal + 1, signal + HORIZON + 1
            if exit_ >= end:
                continue
            assets = GROUPS[g]
            weight = book[signal, assets]
            tradeable = weight > 0
            returns = opening[exit_, assets][tradeable] / opening[enter, assets][tradeable] - 1
            if not np.isfinite(returns).all():
                continue
            group_return = weight[tradeable] @ returns / weight.sum()
            next_week = weekly[np.searchsorted(weekly, signal)]
            for (stage, key), path in paths.items():
                if stage != phase:
                    continue
                initial = path.at[dates[enter], "open_nav_before_trade"]
                final = path.at[dates[exit_], "open_nav_before_trade"]
                wealth = np.r_[1., path.loc[dates[enter]:dates[exit_ - 1], "nav"].to_numpy() / initial, final / initial]
                rows.append({
                    "phase": phase, "strategy": key, "signal_date": dates[signal], "event_open": dates[enter],
                    "label_end_open": dates[exit_], "group": NAMES[g], "rebounded": group_return > 0,
                    "group_return": group_return, "weekly_delay_sessions": int(next_week - signal),
                    "net_return": final / initial - 1,
                    "drawdown": (wealth / np.maximum.accumulate(wealth) - 1.).min(),
                })
    return pd.DataFrame(rows)


def account_evidence(paths):
    accounts = []
    for (phase, key), path in paths.items():
        frame = path.copy()
        if key not in ("budget_120", "budget_60", "event_rebalance"):
            base = paths[(phase, "budget_60")]
            for group in NAMES[:2]:
                target = f"target_{group}"
                frame[f"guard_{group}"] = base[target] - frame[target] > EPS
            frame["guarded"] = frame[["guard_equity", "guard_gold"]].any(axis=1)
        empty = [col for col in frame if col.startswith(("pressure_", "enter_", "exit_", "action_"))]
        accounts.append(frame.drop(columns=empty).assign(phase=phase, strategy=key).reset_index())
    combined = pd.concat(accounts, ignore_index=True)
    return combined.loc[:, ["phase", "strategy", *(col for col in combined if col not in ("phase", "strategy"))]]


def run_event_policy(folder, now):
    check_engine()
    check_clocks()
    check_allocation()
    market, _, hashes = load_market()
    periods, fresh = period_ranges(market.close.index, FREEZE_DATE)
    risk = fit_event_risk(market)
    targets, orders, carry, weekly, confirmed, active, starts, scales = design_policy(market, risk, periods)
    performance, paths = evaluate_sleeves(market, targets, {}, periods, orders)
    selected, qualifies, selection = choose_policy(performance)
    evidence = forecast_evidence(market, risk, periods)
    summary, risk_tests = forecast_comparison(evidence)
    tests = pd.concat([compare_policies(paths, risk_tests, selected), compare_weekly(paths)], ignore_index=True)
    episodes = window_results(market, paths, periods, carry, weekly, starts)
    dates = market.close.index
    decisions = []
    for phase, (start, end) in periods.items():
        signal = np.arange(start - 1, end - 1)
        fields = {"phase": phase, "signal_date": dates[signal], "execution_date": dates[signal + 1],
                  "weekly_due": np.isin(signal, weekly), "event_book_due": orders["event_rebalance"][signal + 1]}
        for j, name in enumerate(NAMES[:2]):
            fields[f"confirm_{name}"] = confirmed[signal, j]
            fields[f"active_{name}"] = active[signal, j]
            for key, scale in scales.items():
                fields[f"scale_{key}_{name}"] = scale[signal, j]
        decisions.append(pd.DataFrame(fields))
    folder.mkdir(parents=True, exist_ok=False)
    files = {
        "portfolio_metrics": performance, "selection": selection, "forecast_scores": summary, "forecast_detail": evidence,
        "event_windows": episodes, "paired_tests": tests, "decisions": pd.concat(decisions, ignore_index=True),
        "train_folds": risk.folds, "daily_backtest": account_evidence(paths),
    }
    for key, frame in files.items():
        frame.to_csv(folder / f"{key}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    from report import write_event_policy_report

    write_event_policy_report(folder, now, performance, paths, summary, tests, selection, selected, qualifies,
                              episodes, risk.threshold, fresh, hashes)
    print(performance.query("phase=='replay' and year=='all' and one_way_bps==5")[
        ["strategy", "cagr", "sharpe", "max_drawdown", "cdar95"]].to_string(index=False))
    print(f"开发期诊断 {selected}，联合改善 {qualifies}。结果目录 {folder}")
