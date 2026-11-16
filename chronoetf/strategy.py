"""在分组风险分配与趋势主干上比较 T-MAVE 的三类时间。"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from backtest import BASE_COST, BOOTSTRAP_REPS, BOOTSTRAP_SEED, CODES, PER_YEAR, build_targets, check_engine, drawdown, load_market, simulate
from risk import EVENT_QUANTILE, LONG_HALF_LIFE, SHORT_HALF_LIFE, VALIDATION_YEAR, fit_risk
from tmave import FEES, cdar, fit_memory, period_ranges, scenario_results, scenario_windows, summarize, targets_and_controls, weekly_signals


MODES = {
    "trade": (False, False),
    "natural": (True, False),
    "event": (False, True),
    "full": (True, True),
    "fixed": (True, True),
}
FAST_GATE = .50
SLOW_GATE = .18
EVENT_MIX = .25
GROUPS = (slice(0, 3), slice(3, 4), slice(4, 5))
CAPS = np.array([.65, .45, .70])
CORE_SPECS = {
    "inverse_capped": (60, None),
    "budget_60": (60, 0.0),
    "budget_120": (120, 0.0),
    "trend_035": (60, .35),
    "trend_070": (60, .70),
}
POLICY_GRID = {"tmave_light": (.15, .90), "tmave_strong": (.30, .85)}
LABELS = {
    "inverse": "原 60 日逆波动", "bond": "5 年国债单持", "equal": "五 ETF 等权",
    "momentum_ewma": "动量 + EWMA", "v2_read": "前轮原记忆 + 创新读取",
    "inverse_capped": "60 日逆波动（分组上限）", "budget_60": "60 日分组风险预算",
    "budget_120": "120 日分组风险预算", "trend_035": "风险预算 + 多周期趋势 0.35",
    "trend_070": "风险预算 + 多周期趋势 0.70",
    "tmave_light": "三时钟完整模块 · 温和", "tmave_strong": "三时钟完整模块 · 较强",
    "trade": "交易日时钟", "natural": "交易日 + 自然日", "event": "交易日 + 事件时间",
    "fixed": "三时钟 · 固定写入", "tilt_only": "三时钟 · 仅风险预算倾斜",
    "guard_only": "三时钟 · 仅风险仓位控制", "vol_match": "同主干 · 开发期波动匹配",
}


@dataclass
class Allocation:
    assets_by_group: np.ndarray
    covariance: np.ndarray
    budget: np.ndarray
    inverse: bool

    def weights(self, tilt):
        desired = self.budget * np.exp(tilt)
        group = desired if self.inverse else allocate_budget(self.covariance, desired / desired.sum())
        group = cap_groups(group)
        return self.assets_by_group @ group


@dataclass
class ClockStudy:
    states: dict
    evidence: dict
    folds: pd.DataFrame


def time_flow(dates, ticks, event_mean, use_natural, use_event):
    session_gap = np.r_[0, np.ones(len(dates) - 1)]
    calendar_gap = np.r_[0, np.diff(dates.to_numpy(dtype="datetime64[D]")).astype(int)]
    trade_time = session_gap[:, None]
    physical_time = (trade_time + calendar_gap[:, None] * (5 / 7)) / 2 if use_natural else trade_time
    if use_event:
        # 上一交易日收盘后确认的信息活动进入下一个流动区间。
        event_time = np.vstack((np.zeros(ticks.shape[1]), ticks[:-1])) / event_mean
        return physical_time * (1 - EVENT_MIX) + event_time * EVENT_MIX
    return np.broadcast_to(physical_time, ticks.shape)


def flow_jump(candidate, elapsed, fast_gate, slow_gate):
    fast_scale = np.exp2(-elapsed / SHORT_HALF_LIFE)
    slow_scale = np.exp2(-elapsed / LONG_HALF_LIFE)
    series = np.empty((*candidate.shape, 2))
    fast = np.zeros(candidate.shape[1])
    anchor = fast.copy()
    for i, signal in enumerate(candidate):
        fast_prior = anchor + fast_scale[i] * (fast - anchor)
        anchor_prior = anchor * slow_scale[i]
        fast = fast_prior + fast_gate[i] * (signal - fast_prior)
        anchor = anchor_prior + slow_gate[i] * (signal - anchor_prior)
        series[i, :, 0] = fast
        series[i, :, 1] = anchor
    return series


def event_evidence(standardized, threshold):
    positive = standardized.ge(threshold)
    negative = standardized.le(-threshold)
    pos, neg = positive.to_numpy(), negative.to_numpy()
    hits = pos | neg
    pos_count = positive.rolling(5, min_periods=1).sum().to_numpy()
    neg_count = negative.rolling(5, min_periods=1).sum().to_numpy()
    repeated = (pos & (pos_count >= 2)) | (neg & (neg_count >= 2))
    co_move = np.zeros_like(hits)
    co_move[:, :3] = (
        (pos[:, :3] & (pos[:, :3].sum(axis=1)[:, None] >= 2))
        | (neg[:, :3] & (neg[:, :3].sum(axis=1)[:, None] >= 2))
    )
    active_group = np.column_stack([hits[:, group].any(axis=1) for group in GROUPS])
    others = np.empty_like(hits, dtype=int)
    for j, group in enumerate(GROUPS):
        others[:, group] = (active_group.sum(axis=1) - active_group[:, j])[:, None]
    confirmed = repeated | co_move
    return {
        "signed_shock": pos.astype(int) - neg.astype(int),
        "repeat_same_direction": repeated,
        "equity_co_move": co_move,
        "other_groups_active": others,
        "event_increment": hits + .50 * confirmed + .25 * others,
        "fast_gate": FAST_GATE * hits,
        "slow_gate": SLOW_GATE * confirmed,
    }


def fit_clocks(close):
    """事件阈值和活动尺度每年只用上年末已观测收益拟合。"""
    dates = close.index
    log_returns = np.log(close).diff()
    prior_vol = log_returns.rolling(60, min_periods=55).std().shift(1)
    standardized = log_returns / prior_vol.replace(0, np.nan)
    z = standardized.to_numpy()
    observed = np.isfinite(z)
    candidate = np.tanh(np.nan_to_num(z, nan=0.0) / 2)
    states = {key: np.full((*z.shape, 2), np.nan) for key in MODES}
    fields = (
        "signed_shock", "repeat_same_direction", "equity_co_move", "other_groups_active",
        "event_increment", "flow_event_lag", "fast_gate", "slow_gate", "fixed_fast_gate", "fixed_slow_gate",
        "threshold", "past_event_mean", "session_increment", "calendar_days", "calendar_equiv",
    )
    evidence = {key: np.full(z.shape, np.nan) for key in (*fields, *(f"elapsed_{mode}" for mode in MODES))}
    calendar_days = np.r_[0, np.diff(dates.to_numpy(dtype="datetime64[D]")).astype(int)]
    folds = []
    for year in range(VALIDATION_YEAR, dates[-1].year + 1):
        start = dates.searchsorted(pd.Timestamp(year, 1, 1))
        end = dates.searchsorted(pd.Timestamp(year + 1, 1, 1))
        threshold = float(np.nanquantile(np.abs(z[:start]), EVENT_QUANTILE))
        event = event_evidence(standardized.iloc[:end], threshold)
        ticks = event["event_increment"]
        event_mean = np.maximum(ticks[:start].sum(axis=0) / observed[:start].sum(axis=0), .01)
        fixed_fast = event["fast_gate"][:start].sum(axis=0) / observed[:start].sum(axis=0)
        fixed_slow = event["slow_gate"][:start].sum(axis=0) / observed[:start].sum(axis=0)
        selected = slice(start - int(year == VALIDATION_YEAR), end)
        for mode, (natural, use_event) in MODES.items():
            elapsed = time_flow(dates[:end], ticks, event_mean, natural, use_event)
            fast = fixed_fast * observed[:end] if mode == "fixed" else event["fast_gate"]
            slow = fixed_slow * observed[:end] if mode == "fixed" else event["slow_gate"]
            result = flow_jump(candidate[:end], elapsed, fast, slow)
            states[mode][selected] = result[selected]
            evidence[f"elapsed_{mode}"][selected] = elapsed[selected]
        for key, values in event.items():
            evidence[key][selected] = values[selected]
        evidence["flow_event_lag"][selected] = np.vstack((np.zeros(len(CODES)), ticks[:-1]))[selected]
        evidence["fixed_fast_gate"][selected] = observed[selected] * fixed_fast
        evidence["fixed_slow_gate"][selected] = observed[selected] * fixed_slow
        evidence["threshold"][selected] = threshold
        evidence["past_event_mean"][selected] = event_mean
        evidence["session_increment"][selected] = 1
        evidence["calendar_days"][selected] = calendar_days[selected, None]
        evidence["calendar_equiv"][selected] = calendar_days[selected, None] * (5 / 7)
        for i, code in enumerate(CODES):
            folds.append({
                "year": year, "symbol": code, "last_calibration_day": dates[start - 1],
                "first_day": dates[start], "absolute_z_threshold": threshold,
                "event_mean": event_mean[i], "fixed_fast_gate": fixed_fast[i], "fixed_slow_gate": fixed_slow[i],
            })
    return ClockStudy(states, evidence, pd.DataFrame(folds))


def check_clocks():
    dates = pd.to_datetime(["2024-02-08", "2024-02-19", "2024-02-20", "2024-02-21"])
    activity = np.array([[0.], [0.], [1.5], [0.]])
    means = np.array([.5])
    trading = time_flow(dates, activity, means, False, False)
    natural = time_flow(dates, activity, means, True, False)
    informative = time_flow(dates, activity, means, False, True)
    np.testing.assert_allclose(trading[:, 0], [0, 1, 1, 1])
    np.testing.assert_allclose(natural[:, 0], [0, (1 + 11 * 5 / 7) / 2, 6 / 7, 6 / 7])
    np.testing.assert_allclose(informative[:, 0], [0, .75, .75, 1.50])
    observed = np.ones_like(activity)
    gates = np.array([[.5], [0], [.5], [0]])
    quiet_trade = flow_jump(observed, trading, gates, np.zeros_like(gates))
    quiet_calendar = flow_jump(observed, natural, gates, np.zeros_like(gates))
    assert quiet_trade[1, 0, 0] > quiet_calendar[1, 0, 0]


def cap_groups(weights):
    """按剩余相对权重重分配，权益、黄金和国债分别受组上限约束。"""
    positions = np.zeros(len(weights))
    free = weights > 0
    remaining = min(1.0, CAPS[free].sum())
    while free.any():
        proposal = remaining * weights[free] / weights[free].sum()
        capped = np.flatnonzero(free)[proposal > CAPS[free]]
        if not len(capped):
            positions[free] = proposal
            break
        positions[capped] = CAPS[capped]
        free[capped] = False
        remaining -= positions[capped].sum()
    return positions


def allocate_budget(cov, budget):
    """坐标下降求解正风险预算，再由调用方应用组资本上限。"""
    x = np.sqrt(budget / np.diag(cov))
    for _ in range(100):
        prior = x.copy()
        for j in range(len(budget)):
            variance = cov[j, j]
            other = cov[j] @ x - variance * x[j]
            root = np.sqrt(other ** 2 + 4 * variance * budget[j])
            x[j] = (root - other) / (2 * variance)
        if np.max(np.abs(x / prior - 1)) < 1e-11:
            break
    return x / x.sum()


def allocations(close, signal_indices):
    """在需下单的信号收盘日构造五种预注册的传统主干。"""
    returns = np.log(close).diff()
    r = returns.to_numpy()
    log_price = np.log(close)
    vol = returns.rolling(60, min_periods=55).std().to_numpy()
    trend = sum((log_price - log_price.shift(period)).to_numpy() / (vol * np.sqrt(period))
                for period in (15, 60, 120)) / 3
    books = {key: {} for key in CORE_SPECS}
    for i in signal_indices:
        inverse = np.where(np.isfinite(r[i]) & np.isfinite(vol[i]) & (vol[i] > 0), 1 / vol[i], 0)
        proportions = np.zeros((len(CODES), len(GROUPS)))
        for j, group in enumerate(GROUPS):
            weights = inverse[group]
            if weights.sum() == 0:
                raise ValueError(f"{close.index[i]:%Y-%m-%d} 的第 {j+1} 组缺少风险估计")
            proportions[group, j] = weights / weights.sum()
        active = inverse > 0
        risk_group = np.array([inverse[group].sum() for group in GROUPS])
        signal_group = proportions.T @ np.clip(np.nan_to_num(trend[i], nan=0), -2, 2)
        covariances = {}
        for length in (60, 120):
            window = r[i + 1 - length:i + 1, active]
            valid = window[np.isfinite(window).all(axis=1)] @ proportions[active]
            empirical = np.cov(valid, rowvar=False)
            covariances[length] = empirical * .9 + np.diag(np.diag(empirical)) * .1
        for key, (length, trend_weight) in CORE_SPECS.items():
            budgets = risk_group if trend_weight is None else np.exp(trend_weight * signal_group)
            books[key][i] = Allocation(proportions, covariances[length], budgets, trend_weight is None)
    return books


def check_allocation():
    cov = np.array([[4, .5, -.1], [.5, 2, 0], [-.1, 0, 1.]]) / 1000
    budget = np.array([.2, .3, .5])
    w = allocate_budget(cov, budget)
    contributions = w * (cov @ w)
    np.testing.assert_allclose(contributions / contributions.sum(), budget, atol=1e-9)
    capped = cap_groups(np.array([.01, .02, .97]))
    np.testing.assert_allclose(capped, [.10, .20, .70])


def account_path(market, targets, key, interval, fee=BASE_COST, order_days=None):
    return simulate(market, targets, "hold_300" if key == "bond" else key, *interval, fee, order_days)


def development_pick(scores, include=None):
    """保留四维 Pareto 前沿，再依据预定的排序选择。"""
    keys = list(scores) if include is None else list(include)
    measures = np.array([
        [scores[key]["sharpe"], scores[key]["cagr"], scores[key]["max_drawdown"], -scores[key]["cdar95"]]
        for key in keys
    ])
    front = []
    for i, key in enumerate(keys):
        dominating = (measures >= measures[i]).all(axis=1) & (measures > measures[i]).any(axis=1)
        if not dominating.any():
            front.append(key)
    winner = min(front, key=lambda k: (-scores[k]["sharpe"], -scores[k]["cagr"], scores[k]["cdar95"], k))
    return winner, front


def covers_objectives(post, initial):
    return (
        post["sharpe"] > initial["sharpe"]
        and post["cagr"] > initial["cagr"]
        and post["max_drawdown"] > initial["max_drawdown"]
        and post["cdar95"] <= initial["cdar95"]
    )


def base_portfolio(market, books):
    target = np.zeros(market.close.shape)
    for i, book in books.items():
        target[i] = book.weights(np.zeros(3))
    return target


def overlay(market, books, state, periods, tilt_amount, guard_scale, frozen_thresholds=None):
    dates = market.close.index
    signal = np.full(len(dates), np.nan)
    tilt = {}
    for i, book in books.items():
        groups = book.assets_by_group
        fast, slow = (groups.T @ state[i]).T
        # 状态矩阵以资产为行，转成三组的快慢读出。
        tilt[i] = .4 * fast + .6 * slow
        prior = book.weights(np.zeros(3))
        group_weights = np.array([prior[g].sum() for g in GROUPS])
        contribution = np.abs(group_weights * (book.covariance @ group_weights))
        contribution /= contribution.sum()
        signal[i] = contribution @ np.maximum(slow - fast, 0)
    dev = weekly_signals(dates, *periods["development"])
    if frozen_thresholds is None:
        entry = max(.02, float(np.quantile(signal[dev], .80)))
        exit_ = min(entry * .75, float(np.quantile(signal[dev], .45)))
    else:
        entry, exit_ = frozen_thresholds
    alarms = np.zeros(len(dates), dtype=bool)
    actions = np.full(len(dates), "", dtype="<U10")
    targets = np.zeros(market.close.shape)
    for interval in periods.values():
        alert = False
        for i in weekly_signals(dates, *interval):
            before = alert
            alert = bool(signal[i] > exit_ if alert else signal[i] >= entry)
            alarms[i] = alert
            transition = "reduce" if alert and not before else "restore" if before and not alert else "hold"
            actions[i] = "watch" if guard_scale == 1 and transition != "hold" else transition
            targets[i] = books[i].weights(tilt_amount * tilt[i]) * (guard_scale if alert else 1)
    return targets, {"score": signal, "alarm": alarms, "action": actions, "entry": entry, "exit": exit_}


def match_volatility(market, target, development, reference_vol):
    base_vol = summarize(account_path(market, target, "vol_match", development))["annual_vol"]
    if reference_vol >= base_vol:
        return target, 1.0
    low, high = 0.0, 1.0
    for _ in range(20):
        risk = (low + high) / 2
        vol = summarize(account_path(market, risk * target, "vol_match", development))["annual_vol"]
        if vol > reference_vol:
            high = risk
        else:
            low = risk
    return target * low, low


def build_experiment(market, clocks, periods):
    initial = weekly_signals(market.close.index, *periods["development"])
    other = [weekly_signals(market.close.index, *p) for k, p in periods.items() if k != "development"]
    books = allocations(market.close, np.unique(np.concatenate([initial, *other])))
    targets = {key: base_portfolio(market, spine) for key, spine in books.items()}
    development = periods["development"]

    def dev_stats(key):
        return summarize(account_path(market, targets[key], key, development))

    score = {key: dev_stats(key) for key in CORE_SPECS}
    core, core_frontier = development_pick(score)
    details = {}
    for key, (tilt, guard) in POLICY_GRID.items():
        targets[key], details[key] = overlay(market, books[core], clocks.states["full"], periods, tilt, guard)
        score[key] = dev_stats(key)
    qualifiers = [key for key in POLICY_GRID if covers_objectives(score[key], score[core])]
    final, policy_frontier = development_pick(score, qualifiers or list(POLICY_GRID))
    tilt, guard = POLICY_GRID[final]
    limits = details[final]["entry"], details[final]["exit"]
    for key in ("trade", "natural", "event", "fixed"):
        targets[key], details[key] = overlay(
            market, books[core], clocks.states[key], periods, tilt, guard, limits,
        )
    targets["tilt_only"], details["tilt_only"] = overlay(
        market, books[core], clocks.states["full"], periods, tilt, 1.0, limits,
    )
    targets["guard_only"], details["guard_only"] = overlay(
        market, books[core], clocks.states["full"], periods, 0, guard, limits,
    )
    targets["vol_match"], vol_scale = match_volatility(market, targets[core], development, score[final]["annual_vol"])
    previous = fit_risk(market.close, market.volume)
    baseline_targets = build_targets(market, previous)
    v2, _, _ = targets_and_controls(market, previous, fit_memory(market, previous), periods)
    targets.update({
        "inverse": baseline_targets["inv_vol"], "bond": np.tile([0, 0, 0, 0, 1.0], (len(market.close), 1)),
        "equal": baseline_targets["equal_5"], "momentum_ewma": baseline_targets["mom_ewma"],
        "v2_read": v2["old_new"],
    })
    choice = {
        "core": core, "policy": final, "core_frontier": core_frontier, "policy_frontier": policy_frontier,
        "qualifiers": qualifiers, "vol_scale": vol_scale, "entry": limits[0], "exit": limits[1],
    }
    return targets, details, choice


def annotate_account(market, path):
    """按资产组加总隔夜和日内损益，保持日净值的可对账性。"""
    cols = [f"w_{symbol}" for symbol in CODES]
    weights = path[cols].to_numpy()
    nav = path["nav"].to_numpy()
    prev_values = np.vstack([np.zeros(len(CODES)), (weights * nav[:, None])[:-1]])
    opening = market.opening.loc[path.index].to_numpy()
    closing = market.close.loc[path.index].to_numpy()
    previous_marks = market.close.ffill().shift().loc[path.index].to_numpy()
    overnight = np.where(np.isfinite(opening / previous_marks), opening / previous_marks, 1)
    intraday = np.where(np.isfinite(closing / opening), closing / opening, 1)
    end_values = weights * nav[:, None]
    pnl = prev_values * (overnight - 1) + end_values * (1 - 1 / intraday)
    labels = ("equity", "gold", "bond")
    for group, name in zip(GROUPS, labels):
        path[f"pnl_{name}"] = pnl[:, group].sum(axis=1)
    path["pnl_fee"] = -path["cost_nav"]
    np.testing.assert_allclose(path[[f"pnl_{x}" for x in (*labels, "fee")]].sum(axis=1),
                               np.diff(np.r_[1, nav]), atol=1e-9, rtol=0)


def performance_stats(path):
    summary = summarize(path)
    nav = np.r_[1.0, (1 + path["net_return"]).cumprod().to_numpy()]
    depth = nav / np.maximum.accumulate(nav) - 1
    trough = depth.argmin()
    peak = nav[:trough + 1].argmax()
    recoveries = np.flatnonzero(nav[trough + 1:] >= nav[peak] - 1e-12)
    recovered = trough + 1 + recoveries[0] if len(recoveries) else None
    summary.update({
        "mean_gold_weight": path["w_518880"].mean(),
        "mean_bond_weight": path["w_511010"].mean(),
        "worst_drawdown_peak": path.index[peak - 1] if peak else pd.NaT,
        "worst_drawdown_trough": path.index[trough - 1] if trough else pd.NaT,
        "recovery_date": path.index[recovered - 1] if recovered is not None else pd.NaT,
        "recovery_sessions": recovered - trough if recovered is not None else np.nan,
    })
    return summary


def evaluate_strategy(market, targets, details, periods):
    records, paths = [], {}
    for phase, dates in periods.items():
        for key, target in targets.items():
            for fee in FEES:
                path = account_path(market, target, key, dates, fee)
                ids = {"phase": phase, "strategy": key, "one_way_bps": fee}
                records.append({**ids, "year": "all", **performance_stats(path)})
                if fee != BASE_COST:
                    continue
                annotate_account(market, path)
                path["drawdown"] = drawdown(path["net_return"])
                path["guarded"] = False
                path["controller_action"] = ""
                path["innovation"] = np.nan
                path["enter_threshold"] = np.nan
                path["exit_threshold"] = np.nan
                if key in details:
                    timeline = details[key]
                    i = market.close.index.get_indexer(path["signal_date"].ffill())
                    path["guarded"] = (key != "tilt_only") & timeline["alarm"][i]
                    path["controller_action"] = np.where(path["scheduled"], timeline["action"][i], "")
                    path["innovation"] = timeline["score"][i]
                    path["enter_threshold"] = timeline["entry"]
                    path["exit_threshold"] = timeline["exit"]
                paths[(phase, key)] = path
                for year, segment in path.groupby(path.index.year):
                    records.append({**ids, "year": str(year), **performance_stats(segment)})
    check_engine(paths)
    return pd.DataFrame(records), paths


def circular_blocks(rng, observations, block):
    starts = rng.integers(observations, size=(BOOTSTRAP_REPS, math.ceil(observations / block)))
    blocks = (starts[:, :, None] + np.arange(block)) % observations
    return blocks.reshape(BOOTSTRAP_REPS, -1)[:, :observations]


def bootstrap_difference(difference, draws, with_pvalue=False):
    lower, upper = np.quantile(draws, [.025, .975])
    result = {"difference": difference, "lower": lower, "upper": upper}
    if with_pvalue:
        count = np.count_nonzero(np.abs(draws - difference) >= abs(difference))
        result["p_raw"] = (1 + count) / (1 + len(draws))
    return result


def paired_intervals(paths, choice, benchmarks=None, phases=("replay", "independent"),
                     drawdown_interval=False, with_pvalue=False):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary, core = choice["policy"], choice["core"]
    results = []

    def measures(data):
        risks = cdar(data)
        annual = np.expm1(PER_YEAR * np.log1p(data).mean(axis=-1))
        sharpe = np.sqrt(PER_YEAR) * data.mean(axis=-1) / data.std(axis=-1, ddof=1)
        values = {"net_sharpe": sharpe, "cagr": annual, "cdar95": risks}
        if drawdown_interval:
            wealth = np.cumprod(1 + data, axis=-1)
            peak = np.maximum.accumulate(np.maximum(wealth, 1), axis=-1)
            values["max_drawdown"] = (wealth / peak - 1).min(axis=-1)
        return values

    compare = benchmarks if benchmarks is not None else (core, "inverse", "vol_match")
    for phase in phases:
        if (phase, primary) not in paths:
            continue
        a = paths[(phase, primary)]["net_return"].to_numpy()
        for block in (10, 20, 40):
            indices = circular_blocks(rng, len(a), block)
            a_point, a_repeat = measures(a), measures(a[indices])
            for key in compare:
                b = paths[(phase, key)]["net_return"].to_numpy()
                b_point, b_repeat = measures(b), measures(b[indices])
                for metric in a_point:
                    inference = bootstrap_difference(
                        a_point[metric] - b_point[metric], a_repeat[metric] - b_repeat[metric], with_pvalue,
                    )
                    row = {
                        "phase": phase, "strategy": primary, "benchmark": key, "block": block, "metric": metric,
                        **inference,
                    }
                    if with_pvalue:
                        row["observations"] = len(a)
                    results.append(row)
    return pd.DataFrame(results)


def export_clock_states(clocks, dates, periods):
    phase = np.full(len(dates), "", dtype="<U12")
    for label, (start, end) in periods.items():
        phase[start:end] = label
    use = phase != ""
    fields = dict(clocks.evidence)
    for key, value in clocks.states.items():
        fields[f"fast_{key}"] = value[:, :, 0]
        fields[f"anchor_{key}"] = value[:, :, 1]
    evidence = clocks.evidence
    fields["cross_recess"] = (evidence["calendar_days"] > 1).astype(int)
    fields["event_class"] = np.select(
        [evidence["equity_co_move"] == 1, evidence["repeat_same_direction"] == 1, evidence["signed_shock"] != 0,
         evidence["other_groups_active"] > 0],
        ["co_move", "repeat", "single", "other_active"], default="quiet",
    )
    index = pd.MultiIndex.from_product((dates[use], CODES), names=["date", "symbol"])
    return pd.DataFrame(
        {"phase": np.repeat(phase[use], len(CODES)), **{name: matrix[use].ravel() for name, matrix in fields.items()}},
        index=index,
    )


def selection_table(performance, choice):
    data = performance.query("phase == 'development' and year == 'all' and one_way_bps == @BASE_COST")
    picks = data.loc[data["strategy"].isin([*CORE_SPECS, *POLICY_GRID])].copy()
    picks["family"] = np.where(picks["strategy"].isin(CORE_SPECS), "spine", "enhancement")
    picks["pareto"] = picks["strategy"].isin(choice["core_frontier"] + choice["policy_frontier"])
    picks["joint_improvement"] = picks["strategy"].isin(choice["qualifiers"])
    picks["locked"] = picks["strategy"].isin([choice["core"], choice["policy"]])
    return picks


def run_strategy(folder, now):
    check_engine()
    check_clocks()
    check_allocation()
    market, _, hashes = load_market()
    clocks = fit_clocks(market.close)
    periods, fresh = period_ranges(market.close.index)
    targets, details, choice = build_experiment(market, clocks, periods)
    performance, paths = evaluate_strategy(market, targets, details, periods)
    scenarios, candidates = scenario_windows(np.log(market.close).diff(), periods)
    scenarios = scenario_results(scenarios, paths, LABELS)
    confidence = paired_intervals(paths, choice)
    selection = selection_table(performance, choice)
    states = export_clock_states(clocks, market.close.index, periods)
    folder.mkdir(parents=True, exist_ok=False)
    for name, frame, include_index in (
        ("portfolio_metrics", performance, False), ("selection", selection, False),
        ("daily_backtest", pd.concat(paths, names=["phase", "strategy"]), True),
        ("time_states", states, True), ("scenarios", scenarios, False),
        ("paired_intervals", confidence, False), ("clock_folds", clocks.folds, False),
    ):
        frame.to_csv(folder / f"{name}.csv", index=include_index, encoding="utf-8-sig", float_format="%.12g")
    from report import write_strategy_report

    write_strategy_report(folder, now, performance, paths, confidence, selection, states, scenarios,
                          candidates, choice, fresh, hashes)
    sample = performance.query("phase == 'replay' and year == 'all' and one_way_bps == @BASE_COST")
    print(sample[["strategy", "cagr", "sharpe", "max_drawdown", "cdar95", "mean_bond_weight"]].to_string(index=False))
    print(f"锁定策略：{choice['core']} / {choice['policy']}，结果目录：{folder}")
