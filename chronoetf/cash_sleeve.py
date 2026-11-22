"""银华日利场内现金账本，使用不复权行情和公告分红。"""

from datetime import date
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import CODES, EPS
from prepare_daily import ENDPOINT, load_history, write_csv


ROOT = Path(__file__).resolve().parent
SYMBOL = "511880"
CAPITAL = 5_000_000
LOT = 100
MIN_COMMISSION = 5 / CAPITAL
ANNOUNCEMENTS = (
    ("2021-12-28", "2021-12-30", "2021-12-31", "2022-01-06", 2.142,
     "https://static.cninfo.com.cn/finalpage/2021-12-28/1212036246.PDF"),
    ("2022-12-27", "2022-12-29", "2022-12-30", "2023-01-05", 1.717,
     "https://www.howbuy.com/fund/info/dtl/5118803094266.htm"),
    ("2023-12-26", "2023-12-28", "2023-12-29", "2024-01-04", 1.783,
     "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2023-12-26/511880_20231226_116U.pdf"),
    ("2024-12-26", "2024-12-30", "2024-12-31", "2025-01-06", 1.5521,
     "https://static.cninfo.com.cn/finalpage/2024-12-26/1222146837.PDF"),
    ("2025-12-26", "2025-12-30", "2025-12-31", "2026-01-07", 1.1553,
     "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2025-12-26/511880_20251226_IIQW.pdf"),
)
TRADING_SOURCE = "https://edu.sse.com.cn/etfgame/gametopic/c/4940402.shtml"
LOT_SOURCE = "https://www.sse.com.cn/assortment/fund/etf/question/"


def load_cash(dates):
    start, stop = date(2020, 12, 31), dates[-1].date()
    folder = ROOT / "cash_data"
    folder.mkdir(exist_ok=True)
    file = folder / f"{SYMBOL}_{start:%y%m%d}-{stop:%y%m%d}_raw.csv"
    if not file.exists():
        bars = load_history("sh" + SYMBOL, start, stop, adjustment="")
        if not bars:
            raise ValueError("511880 行情为空")
        write_csv(file, ("date", "open", "high", "low", "close", "volume_lots"),
                  [(day, *bar) for day, bar in sorted(bars.items())])
    raw = pd.read_csv(file, parse_dates=["date"]).set_index("date")
    expected = dates[dates >= pd.Timestamp(start)]
    assert raw.index.equals(expected), "511880 和原行情的交易日不一致"
    assert np.isfinite(raw).all().all() and (raw > 0).all().all()
    assert (raw.low <= raw[["open", "close"]].min(axis=1)).all()
    assert (raw.high >= raw[["open", "close"]].max(axis=1)).all()
    events = []
    for announced, record, ex, pay, amount, url in ANNOUNCEMENTS:
        if not raw.index[0] <= pd.Timestamp(record) <= raw.index[-1]:
            continue
        assert all(pd.Timestamp(x) in raw.index for x in (record, ex, pay))
        assert announced <= record < ex < pay
        reset = raw.loc[ex, "open"] - raw.loc[record, "close"]
        assert abs(reset + amount) < .5, (ex, reset, amount)
        events.append({"announced": announced, "record": record, "ex": ex, "paid": pay, "per_share": amount,
                       "raw_open_gap": float(reset), "source": url})
    manifest = {"symbol": SYMBOL, "source": ENDPOINT, "basis": "day / raw", "filename": file.name,
                "sha256": sha256(file.read_bytes()).hexdigest(), "rows": len(raw), "first": str(expected[0].date()),
                "last": str(expected[-1].date()), "trading_source": TRADING_SOURCE, "lot_source": LOT_SOURCE,
                "announcements": events}
    return raw.reindex(dates), manifest


def simulate_cash(market, targets, interval, risky_bps, quotes, events, cash_bps=2):
    """只交易既有五个风险目标，把余款买入整手货币 ETF。"""
    dates = market.close.index
    opening, closing = market.opening.to_numpy(), market.close.to_numpy()
    previous = market.close.ffill().shift().to_numpy()
    money_open = quotes.open.to_numpy() / CAPITAL
    money_close = quotes.close.to_numpy() / CAPITAL
    assert np.isfinite(money_open[interval[0]:interval[1]]).all()
    week = dates.to_period("W-SUN")
    scheduled = np.r_[False, week[1:] != week[:-1]]
    risky_rate, money_rate = risky_bps / 10_000, cash_bps / 10_000
    records = {dates.get_loc(x["record"]): (dates.get_loc(x["ex"]), dates.get_loc(x["paid"]), x["per_share"] / CAPITAL)
               for x in events}
    cash, nav, shares = 1., 1., 0
    last_money, receivable = 0., 0.
    pending = np.zeros(len(CODES), dtype=bool)
    notionals = np.zeros(len(CODES))
    desired = np.zeros(len(CODES))
    entitlements, unpaid, rows = {}, {}, []
    signal = pd.NaT

    for i in range(*interval):
        paid = unpaid.pop(i, 0.)
        cash += paid
        receivable -= paid
        payout, accrual = entitlements.pop(i, (None, 0.))
        if accrual:
            unpaid[payout] = unpaid.get(payout, 0.) + accrual
            receivable += accrual
        live = np.isfinite(opening[i])
        live_overnight = live & np.isfinite(previous[i])
        stock_overnight = notionals[live_overnight].copy()
        notionals[live_overnight] *= opening[i, live_overnight] / previous[i, live_overnight]
        risky_pnl = (notionals[live_overnight] - stock_overnight).sum()
        money_pnl = shares * (money_open[i] - last_money) + accrual
        opening_nav = cash + notionals.sum() + shares * money_open[i] + receivable
        order = i == interval[0] or scheduled[i]
        delayed = not order and (pending & live).any()
        if order:
            desired = targets[i - 1]
            signal = dates[i - 1]
        shares_traded = 0
        trade, fees_risky, fees_cash = 0., 0., 0.
        if order or delayed:
            desired_value = desired * opening_nav
            pending = ~live & (np.abs(notionals / opening_nav - desired) > EPS)
            sells = np.where(live, np.maximum(notionals - desired_value, 0), 0)
            notionals -= sells
            cash += sells.sum() * (1 - risky_rate)
            buys = np.where(live, np.maximum(desired_value - notionals, 0), 0)
            needed = buys.sum() * (1 + risky_rate) - cash
            if needed > 1e-12 and shares:
                gross_needed = max(needed / (1 - money_rate), needed + MIN_COMMISSION)
                shares_traded = -min(shares, LOT * int(np.ceil(gross_needed / (LOT * money_open[i]))))
                sale = -shares_traded * money_open[i]
                fees_cash = max(sale * money_rate, MIN_COMMISSION)
                cash += sale - fees_cash
                shares += shares_traded
            bought = buys.sum()
            if bought:
                buys *= min(1., cash / (bought * (1 + risky_rate)))
                cash -= buys.sum() * (1 + risky_rate)
                notionals += buys
            if shares_traded == 0 and cash > MIN_COMMISSION:
                available = min(cash / (1 + money_rate), cash - MIN_COMMISSION)
                shares_traded = LOT * int(available / (LOT * money_open[i]))
                if shares_traded:
                    purchase = shares_traded * money_open[i]
                    fees_cash = max(purchase * money_rate, MIN_COMMISSION)
                    cash -= purchase + fees_cash
                    shares += shares_traded
            trade = sells.sum() + buys.sum()
            fees_risky = trade * risky_rate
        intraday_open = notionals[live].copy()
        notionals[live] *= closing[i, live] / opening[i, live]
        risky_pnl += (notionals[live] - intraday_open).sum()
        money_pnl += shares * (money_close[i] - money_open[i])
        if i in records:
            ex, pay, amount = records[i]
            entitlements[ex] = (pay, shares * amount)
        last_money = money_close[i]
        money_value = shares * last_money
        next_nav = cash + notionals.sum() + money_value + receivable
        fees = fees_risky + fees_cash
        assert cash >= -1e-12 and receivable >= -1e-12 and shares >= 0
        assert abs(next_nav - nav - risky_pnl - money_pnl + fees) < 1e-10
        rows.append((dates[i], signal, order, delayed, opening_nav, next_nav, next_nav / nav - 1,
                     (trade + abs(shares_traded) * money_open[i]) / opening_nav, fees, cash / next_nav,
                     *(notionals / next_nav), *desired, shares, shares_traded, money_value / next_nav,
                     receivable, accrual, paid, money_pnl, fees_cash, fees_risky))
        nav = next_nav
    columns = ["date", "signal_date", "scheduled", "resumed_order", "open_nav_before_trade", "nav", "net_return",
               "turnover", "cost_nav", "cash_weight", *(f"w_{s}" for s in CODES), *(f"target_{s}" for s in CODES),
               "cash_etf_shares", "cash_etf_trade_shares", "cash_etf_weight", "dividend_receivable", "dividend_accrual",
               "dividend_paid", "cash_etf_pnl", "cash_etf_cost", "risk_cost"]
    return pd.DataFrame(rows, columns=columns).set_index("date")
