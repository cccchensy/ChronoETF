"""运行 ChronoETF 的回测、向前验证与研报生成。"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_daily import load_history
from risk import HORIZON, RISK_NAMES, TEST_YEAR, VALIDATION_YEAR, fit_risk, qlike


ROOT = Path(__file__).resolve().parent
CODES = ("510300", "510500", "159915", "518880", "511010")
ASSETS = ("沪深300", "中证500", "创业板", "黄金", "5年国债")
STRATEGIES = {
    "hold_300": "沪深300买入持有",
    "equal_5": "五ETF等权",
    "stock_bond": "60/40股债组合",
    "inv_vol": "60日逆波动",
    "momentum": "60日动量前二",
    "mom_ewma": "动量 + EWMA",
    "mom_single": "动量 + 单记忆",
    "mom_fixed": "动量 + 双记忆固定写入",
    "chrono": "ChronoETF",
    "chrono_calendar": "ChronoETF 自然日",
}
FORECAST_WEIGHTS = {
    "mom_ewma": "ewma", "mom_single": "single", "mom_fixed": "fixed",
    "chrono": "chrono", "chrono_calendar": "calendar",
}
TRADITIONAL = ("hold_300", "equal_5", "stock_bond", "inv_vol", "momentum", "mom_ewma")
PER_YEAR = 252
LOOKBACK = 60
VOL_TARGET = 0.10
BASE_COST = 5
COSTS = (0, BASE_COST, 2 * BASE_COST, 5 * BASE_COST)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42
TZ_SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")
EPS = 1e-10


@dataclass
class Market:
    opening: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame


def load_market():
    reference = json.loads((ROOT / "market_reference.json").read_bytes())
    start, end = (pd.Timestamp(reference["coverage"][key]) for key in ("start", "end"))
    suffix = f"{start:%y%m%d}-{end:%y%m%d}"
    metadata = {item["symbol"]: item for item in reference["securities"]}
    frames = {}
    hashes = {}
    calendar = None
    for code in CODES:
        path = ROOT / "data" / f"{code}_{suffix}.csv"
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        if list(frame.columns) != ["open", "high", "low", "close", "volume_lots", "status"]:
            raise ValueError(f"{path.name} 表头不符合后复权行情口径")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{path.name} 交易日重复或乱序")
        if frame.index[0] != start or frame.index[-1] != end:
            raise ValueError(f"{path.name} 未覆盖约定的日期区间")
        if calendar is not None and not calendar.equals(frame.index):
            raise ValueError(f"{code} 交易日历与其他 ETF 不一致")
        calendar = frame.index
        expected = pd.Series("traded", index=calendar)
        expected.loc[calendar < pd.Timestamp(metadata[code]["listing_date"])] = "pre_listing"
        for halt in reference["suspensions"]:
            if halt["symbol"] == code:
                expected.loc[pd.Timestamp(halt["date"])] = "suspended"
        if not frame["status"].equals(expected):
            raise ValueError(f"{code} 交易状态与参考表不一致")
        bars = frame.drop(columns="status")
        traded = frame["status"].eq("traded")
        b = bars.loc[traded]
        valid = (
            np.isfinite(b).all().all() and b.gt(0).all().all()
            and b["low"].le(b[["open", "close"]].min(axis=1)).all()
            and b["high"].ge(b[["open", "close"]].max(axis=1)).all()
            and bars.loc[~traded].isna().all().all()
        )
        if not valid:
            raise ValueError(f"{code} 存在异常 OHLC、成交量或停牌行情")
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        frames[code] = frame

    def assemble(field):
        return pd.DataFrame({code: frame[field] for code, frame in frames.items()})

    return Market(assemble("open"), assemble("close"), assemble("volume_lots")), reference, hashes


def audit_prices(market, reference):
    """以税前现金在除权日开盘再投资构造敏感性序列，原始行情只进内存。"""
    dates = market.close.index
    audit_open = market.opening.copy() * np.nan
    audit_close = market.close.copy() * np.nan
    by_code = {item["symbol"]: item for item in reference["securities"]}
    events = {}
    for item in reference["cash_dividends"] + reference["splits"]:
        key = (item["symbol"], pd.Timestamp(item["price_effective_date"]))
        if key in events:
            raise ValueError(f"复权核对中同日存在多项权益事件：{key}")
        events[key] = item

    for code in CODES:
        prefix = "sh" if by_code[code]["exchange"] == "SSE" else "sz"
        data = load_history(prefix + code, dates[0].date(), dates[-1].date(), adjustment="")
        raw = pd.DataFrame.from_dict(
            data, orient="index", columns=["open", "high", "low", "close", "volume"],
        ).astype(float)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.sort_index()
        listed = market.close[code].dropna().index
        if not raw.index.equals(listed) or not np.array_equal(raw["volume"], market.volume.loc[listed, code]):
            raise ValueError(f"{code} 对账时交易日期或成交量与后复权样本不一致")

        previous_close = None
        adjusted_close = 1.0
        for day, row in raw.iterrows():
            if previous_close is None:
                adjusted_open = row["open"] / row["close"]
            else:
                event = events.get((code, day), {})
                split = float(event.get("shares_per_old_share", 1))
                dividend = float(event.get("cash_per_old_share_cny") or 0)
                adjusted_open = adjusted_close * (split * row["open"] + dividend) / previous_close
            adjusted_close = adjusted_open * row["close"] / row["open"]
            audit_open.at[day, code], audit_close.at[day, code] = adjusted_open, adjusted_close
            previous_close = row["close"]
    if any(pd.isna(audit_close.at[day, code]) for code, day in events):
        raise ValueError("公司行动生效日缺少行情")

    hfq_returns = market.close.ffill().pct_change(fill_method=None).where(market.close.notna())
    audit_returns = audit_close.ffill().pct_change(fill_method=None).where(market.close.notna())
    difference = (hfq_returns - audit_returns).abs()
    differences = pd.DataFrame([
        {
            "symbol": code, "median_bps": difference[code].median() * 10000,
            "p95_bps": difference[code].quantile(0.95) * 10000,
            "max_bps": difference[code].max() * 10000,
            "max_date": difference[code].idxmax().date().isoformat(),
        }
        for code in CODES
    ])
    sample = [(code, day, event.get("kind", "dividend")) for (code, day), event in events.items()]
    sample.append(("510300", pd.Timestamp("2024-09-30"), "no_action"))
    actions = pd.DataFrame([
        {"symbol": code, "date": day.date().isoformat(), "event": event,
         "hfq_return": hfq_returns.at[day, code], "audit_return": audit_returns.at[day, code]}
        for code, day, event in sample if day in dates
    ]).sort_values(["date", "symbol"])
    return Market(audit_open, audit_close, market.volume), differences, actions


def build_targets(market, risk, forecast_weights=FORECAST_WEIGHTS):
    dates = market.close.index
    inverse = risk.log_returns.rolling(LOOKBACK, min_periods=55).std().replace(0, np.nan).rdiv(1)
    inverse = inverse.where(market.close.notna(), 0)
    inverse = inverse.div(inverse.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    strength = market.close.div(market.close.shift(LOOKBACK)) - 1
    rank = strength.rank(axis=1, method="first", ascending=False)
    momentum = rank.le(2).astype(float)
    momentum = momentum.div(momentum.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    targets = {
        "hold_300": np.tile([1, 0, 0, 0, 0], (len(dates), 1)),
        "equal_5": np.full(market.close.shape, 0.2),
        "stock_bond": np.tile([0.6, 0, 0, 0, 0.4], (len(dates), 1)),
        "inv_vol": inverse.to_numpy(),
        "momentum": momentum.to_numpy(),
    }
    for key in forecast_weights:
        targets[key] = np.zeros(market.close.shape)
    forecast_arrays = {key: risk.forecasts[ref].to_numpy() for key, ref in forecast_weights.items()}
    returns = risk.log_returns.to_numpy()
    for i in range(LOOKBACK, len(dates)):
        basket = targets["momentum"][i]
        active = basket > 0
        if not active.any():
            continue
        history = returns[i + 1 - LOOKBACK:i + 1, active]
        history = history[np.isfinite(history).all(axis=1)]
        corr = np.atleast_2d(np.corrcoef(history, rowvar=False)) if active.sum() > 1 else np.ones((1, 1))
        # 所有模型用同一相关矩阵、同一动量篮子，差别来自各资产的方差预测。
        for name, variance in forecast_arrays.items():
            sigma = np.sqrt(variance[i, active])
            if not np.isfinite(sigma).all():
                continue
            weighted = basket[active] * sigma
            annual_vol = np.sqrt(max(weighted @ corr @ weighted, EPS) * PER_YEAR)
            targets[name][i] = basket * min(1.0, VOL_TARGET / annual_vol)
    return targets


def trade_at_open(open_values, cash, target, tradable, fee_rate):
    opening_nav = open_values.sum() + cash
    desired = opening_nav * target
    sells = np.where(tradable, np.maximum(open_values - desired, 0), 0)
    open_values = open_values - sells
    cash += sells.sum() * (1 - fee_rate)
    planned = np.where(tradable, np.maximum(desired - open_values, 0), 0)
    if planned.sum() > 0:
        planned *= min(1, cash / (planned.sum() * (1 + fee_rate)))
    cash -= planned.sum() * (1 + fee_rate)
    traded = float(sells.sum() + planned.sum())
    return open_values + planned, max(0.0, cash), traded / opening_nav, traded * fee_rate


def simulate(market, targets, strategy, start, end, cost_bps, order_days=None):
    dates = market.close.index
    open_array = market.opening.to_numpy()
    close_array = market.close.to_numpy()
    previous_marks = market.close.ffill().shift(1).to_numpy()
    if order_days is None:
        week = dates.to_period("W-SUN")
        order_days = np.r_[False, (week[1:] != week[:-1]) & (strategy != "hold_300")]
    notionals = np.zeros(len(CODES))
    cash = 1.0
    nav = 1.0
    desired = np.zeros(len(CODES))
    pending = np.zeros(len(CODES), dtype=bool)
    signal_time = pd.NaT
    rows = []
    for i in range(start, end):
        live = np.isfinite(open_array[i])
        overnight_live = live & np.isfinite(previous_marks[i])
        notionals[overnight_live] *= open_array[i, overnight_live] / previous_marks[i, overnight_live]
        opening_nav = cash + notionals.sum()
        scheduled = i == start or order_days[i]
        delayed = not scheduled and (pending & live).any()
        if scheduled:
            desired = targets[i - 1]
            signal_time = dates[i - 1]
        turnover, fee = 0.0, 0.0
        if scheduled or delayed:
            pending = ~live & (np.abs(notionals / opening_nav - desired) > EPS)
            notionals, cash, turnover, fee = trade_at_open(notionals, cash, desired, live, cost_bps / 10000)
        notionals[live] *= close_array[i, live] / open_array[i, live]
        close_nav = cash + notionals.sum()
        weights = notionals / close_nav
        rows.append((
            dates[i], signal_time if scheduled or delayed else pd.NaT,
            scheduled, delayed, opening_nav, close_nav, close_nav / nav - 1, turnover, fee, cash / close_nav,
            *weights, *desired,
        ))
        nav = close_nav
    return pd.DataFrame(
        rows, columns=[
            "date", "signal_date", "scheduled", "resumed_order", "open_nav_before_trade", "nav",
            "net_return", "turnover", "cost_nav", "cash_weight", *(f"w_{code}" for code in CODES),
            *(f"target_{code}" for code in CODES),
        ],
    ).set_index("date")


def drawdown(returns):
    nav = (1 + returns).cumprod()
    return nav / nav.cummax().clip(lower=1) - 1


def metrics(frame):
    daily = frame["net_return"]
    n = len(daily)
    terminal = float((1 + daily).prod())
    std = daily.std(ddof=1)
    under = drawdown(daily).lt(-EPS)
    dd = float(drawdown(daily).min())
    cagr = terminal ** (PER_YEAR / n) - 1
    negative = np.minimum(daily.to_numpy(), 0)
    downside = np.sqrt(np.mean(negative ** 2))
    return {
        "first_date": frame.index[0].date().isoformat(), "last_date": frame.index[-1].date().isoformat(),
        "trading_days": n, "total_return": terminal - 1, "cagr": cagr, "annual_vol": std * np.sqrt(PER_YEAR),
        "max_drawdown": dd, "sharpe": daily.mean() / std * np.sqrt(PER_YEAR) if std > 0 else np.nan,
        "sortino": daily.mean() / downside * np.sqrt(PER_YEAR) if downside > 0 else np.nan,
        "calmar": cagr / -dd if dd < 0 else np.nan,
        "drawdown_days": int(under.groupby((~under).cumsum()).sum().max()),
        "annual_turnover_two_way": frame["turnover"].sum() * PER_YEAR / n,
        "fee_over_period_start_nav": frame["cost_nav"].sum() / (frame["nav"].iloc[0] / (1 + daily.iloc[0])),
        "mean_gross_weight": 1 - frame["cash_weight"].mean(),
        "mean_equity_weight": frame[[f"w_{code}" for code in CODES[:3]]].sum(axis=1).mean(),
    }


def evaluate_market(market, risk, basis):
    dates = market.close.index
    validation_start = dates.searchsorted(pd.Timestamp(VALIDATION_YEAR, 1, 1))
    test_start = dates.searchsorted(pd.Timestamp(TEST_YEAR, 1, 1))
    targets = build_targets(market, risk)
    base_paths = {}
    records = []
    for bps in COSTS:
        for strategy in STRATEGIES:
            for period, start, end in (
                ("validation", validation_start, test_start), ("test", test_start, len(dates)),
            ):
                path = simulate(market, targets[strategy], strategy, start, end, bps)
                path["drawdown"] = drawdown(path["net_return"])
                if bps == BASE_COST:
                    base_paths[(period, strategy)] = path
                tag = {"price_basis": basis, "strategy": strategy, "one_way_bps": bps}
                records.append({**tag, "period": period, **metrics(path)})
                for year, section in path.groupby(path.index.year):
                    records.append({**tag, "period": str(year), **metrics(section)})
    return pd.DataFrame(records), base_paths


def matched_forecasts(risk, period):
    dates = risk.realized.index
    start_year = VALIDATION_YEAR if period == "validation" else TEST_YEAR
    start = dates.searchsorted(pd.Timestamp(start_year, 1, 1))
    end = dates.searchsorted(pd.Timestamp(TEST_YEAR, 1, 1)) if period == "validation" else len(dates)
    actual = risk.realized.iloc[start:end - HORIZON]
    forecast = {key: values.loc[actual.index] for key, values in risk.forecasts.items()}
    mask = actual.notna()
    for values in forecast.values():
        mask &= values.notna()
    return actual.where(mask), {key: values.where(mask) for key, values in forecast.items()}


def summarize_risk(risk, basis):
    rows = []
    for period in ("validation", "test"):
        actual, forecast = matched_forecasts(risk, period)
        for key, values in forecast.items():
            rows.append({
                "price_basis": basis, "period": period, "model": key, "observations": int(actual.count().sum()),
                "qlike": qlike(actual, values).stack().mean(),
                "mae_variance": (actual - values).abs().stack().mean(),
                "predicted_over_realized": values.stack().sum() / actual.stack().sum(),
            })
    return pd.DataFrame(rows)


def event_risk(risk):
    actual, forecast = matched_forecasts(risk, "test")
    counts = risk.shock_counts.loc[actual.index]
    rows = []
    for caption, mask in (
        ("无冲击", counts.eq(0)), ("单次冲击", counts.eq(1)), ("重复冲击", counts.ge(2)),
    ):
        key = mask & actual.notna()
        values = actual.where(key)
        rows.append({
            "event": caption, "observations": int(values.count().sum()),
            **{name: qlike(values, forecast[name].where(key)).stack().mean() for name in ("ewma", "fixed", "chrono")},
            "short_gate": risk.short_gate.loc[actual.index].where(key).stack().mean(),
            "long_gate": risk.long_gate.loc[actual.index].where(key).stack().mean(),
        })
    return pd.DataFrame(rows)


def bootstrap_checks(paths, risk, best_baseline):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    chrono = paths[("test", "chrono")]["net_return"].to_numpy()
    references = dict.fromkeys(("mom_ewma", best_baseline))
    actual, predicted = matched_forecasts(risk, "test")
    paired_losses = (qlike(actual, predicted["chrono"]) - qlike(actual, predicted["ewma"])).mean(axis=1).dropna().to_numpy()
    tests = []

    def block_positions(length, block):
        starts = rng.integers(length, size=(BOOTSTRAP_REPS, (length + block - 1) // block))
        return ((starts[:, :, None] + np.arange(block)) % length).reshape(BOOTSTRAP_REPS, -1)[:, :length]

    def sharpe(values):
        return np.sqrt(PER_YEAR) * values.mean(axis=-1) / values.std(axis=-1, ddof=1)

    def record(kind, left, right, block, point, samples):
        low, high = np.quantile(samples, (0.025, 0.975))
        tests.append({"measure": kind, "model": left, "reference": right, "block": block,
                      "difference": point, "lower": low, "upper": high})

    for block in (10, 20, 40):
        positions = block_positions(len(chrono), block)
        for reference in references:
            bench = paths[("test", reference)]["net_return"].to_numpy()
            record("sharpe", "chrono", reference, block, sharpe(chrono) - sharpe(bench),
                   sharpe(chrono[positions]) - sharpe(bench[positions]))
        loss_positions = block_positions(len(paired_losses), block)
        record("qlike", "chrono", "ewma", block, paired_losses.mean(), paired_losses[loss_positions].mean(axis=1))
    return pd.DataFrame(tests)


def check_engine(paths=None):
    """用可手算的情景核对隔夜、成交时点、停牌、手续费和每日净值。"""
    dates = pd.to_datetime(["2021-02-03", "2021-02-04", "2021-02-05", "2021-02-08", "2021-02-09", "2021-02-10"])
    opening = pd.DataFrame(100.0, index=dates, columns=CODES)
    close = opening.copy()
    primary = CODES[0]
    opening.loc[dates[3:], primary] = [110.0, 121.0, 135.0]
    close.loc[dates[3:], primary] = [120.0, 132.0, 140.0]
    volume = opening.copy()
    targets = np.zeros(opening.shape)
    targets[1, 0] = 1
    sample = Market(opening, close, volume)
    ordinary = simulate(sample, targets, "weekly", 2, 6, 0)
    np.testing.assert_allclose(ordinary["nav"], [1, 1.10, 1.10, 1.10])
    assert ordinary.loc[dates[3], "signal_date"] == dates[2]

    hold = simulate(sample, np.tile([1, 0, 0, 0, 0], (len(dates), 1)), "hold_300", 2, 6, BASE_COST)
    np.testing.assert_allclose(hold["nav"].iloc[-1], 1.4 / (1 + BASE_COST / 10000))
    np.testing.assert_allclose(hold["cost_nav"].iloc[0], 1 - 1 / (1 + BASE_COST / 10000))

    opening.loc[dates[3], primary] = close.loc[dates[3], primary] = np.nan
    delayed = simulate(sample, targets, "weekly", 2, 6, 0)
    np.testing.assert_allclose(delayed["nav"], [1, 1, 1.21, 1.21])
    assert delayed.loc[dates[4], "resumed_order"] and delayed.loc[dates[4], "signal_date"] == dates[2]

    for path in (paths or {}).values():
        recorded = path["signal_date"].notna()
        assert path.loc[recorded, "signal_date"].lt(path.index[recorded]).all()
        weights = path[[f"w_{code}" for code in CODES] + ["cash_weight"]]
        assert weights.ge(-EPS).all().all()
        np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=EPS)
        np.testing.assert_allclose((1 + path["net_return"]).cumprod(), path["nav"], rtol=0, atol=EPS)
        np.testing.assert_allclose(path["cost_nav"], path["turnover"] * path["open_nav_before_trade"] * BASE_COST / 10000)


def save_evidence(folder, metrics_table, paths, risk):
    metrics_table.to_csv(folder / "portfolio_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    daily = pd.concat(paths, names=["period", "strategy"])
    daily.to_csv(folder / "daily_backtest.csv", encoding="utf-8-sig", float_format="%.12g")

    dates = risk.realized.index
    actual = risk.realized.copy()
    boundary = dates.searchsorted(pd.Timestamp(TEST_YEAR, 1, 1))
    actual.iloc[boundary - HORIZON:boundary] = np.nan
    fields = {"realized_var": actual, **risk.forecasts,
              "shock_count_5d": risk.shock_counts, "short_gate": risk.short_gate, "long_gate": risk.long_gate}
    long = pd.DataFrame({
        key: table.loc[dates.year >= VALIDATION_YEAR].to_numpy().ravel() for key, table in fields.items()
    }, index=pd.MultiIndex.from_product((dates[dates.year >= VALIDATION_YEAR], CODES), names=["date", "symbol"]))
    long.to_csv(folder / "risk_forecasts.csv", encoding="utf-8-sig", float_format="%.12g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-price-audit", action="store_true", help="离线运行主回测，跳过收益口径敏感性实验")
    parser.add_argument("--tmave", action="store_true", help="运行 T-MAVE 风控改进与历史回放")
    parser.add_argument("--strategy", action="store_true", help="运行成熟策略上的 T-MAVE 三时钟实验")
    parser.add_argument("--sleeves", action="store_true", help="运行因子层与 T-MAVE 分组风控")
    parser.add_argument("--alpha", action="store_true", help="运行 T-MAVE 股票与黄金相对收益实验")
    parser.add_argument("--horizon", action="store_true", help="运行 T-MAVE 自适应协方差记忆实验")
    parser.add_argument("--event-risk", action="store_true", help="运行 T-MAVE 重复冲击和事件风控实验")
    parser.add_argument("--vol-target", action="store_true", help="运行 T-MAVE 组合波动目标实验")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results", help="运行目录的上一级路径")
    args = parser.parse_args()
    now = datetime.now(TZ_SHANGHAI)
    folder = args.output_root / now.strftime("%Y%m%d-%H%M%S")
    if args.vol_target:
        from volatility_policy import run_volatility_policy

        return run_volatility_policy(folder, now)
    if args.event_risk:
        from event_policy import run_event_policy

        return run_event_policy(folder, now)
    if args.horizon:
        from horizon import run_horizon

        return run_horizon(folder, now)
    if args.alpha:
        from allocation_alpha import run_alpha

        return run_alpha(folder, now)
    if args.sleeves:
        from sleeves import run_sleeves

        return run_sleeves(folder, now)
    if args.strategy:
        from strategy import run_strategy

        return run_strategy(folder, now)
    if args.tmave:
        from tmave import run_study

        return run_study(folder, now)
    check_engine()
    market, reference, hashes = load_market()
    risk = fit_risk(market.close, market.volume)
    main_metrics, paths = evaluate_market(market, risk, "hfq")
    check_engine(paths)
    validation = main_metrics.query("period == 'validation' and one_way_bps == @BASE_COST")
    traditional = validation.loc[validation["strategy"].isin(TRADITIONAL)]
    best_baseline = traditional.loc[traditional["sharpe"].idxmax(), "strategy"]
    risk_metrics = summarize_risk(risk, "hfq")
    audit = None
    if not args.skip_price_audit:
        alternate, differences, actions = audit_prices(market, reference)
        audit_risk = fit_risk(alternate.close, alternate.volume, fixed_lambda=risk.ewma_lambda)
        audit_metrics, _ = evaluate_market(alternate, audit_risk, "action_audit")
        main_metrics = pd.concat((main_metrics, audit_metrics), ignore_index=True)
        risk_metrics = pd.concat((risk_metrics, summarize_risk(audit_risk, "action_audit")), ignore_index=True)
        audit = (differences, actions)
    events = event_risk(risk)
    intervals = bootstrap_checks(paths, risk, best_baseline)
    folder.mkdir(parents=True, exist_ok=False)
    save_evidence(folder, main_metrics, paths, risk)
    from report import write_report

    write_report(
        folder, now, market, hashes, main_metrics, paths, risk, risk_metrics, events,
        intervals, best_baseline, audit,
    )
    selected = main_metrics.query("price_basis == 'hfq' and period == 'test' and one_way_bps == @BASE_COST")
    print(selected[["strategy", "total_return", "cagr", "max_drawdown", "sharpe"]].to_string(index=False))
    print(f"报告与收益图：{folder}")


if __name__ == "__main__":
    main()
