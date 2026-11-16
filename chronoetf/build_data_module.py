"""生成 ChronoETF 的五份后复权日线 CSV。"""

import argparse
import json
from datetime import date
from pathlib import Path

import akshare as ak

from prepare_daily import PRICE_COLUMNS, load_history, write_csv


PROJECT = Path(__file__).resolve().parent
REFERENCE = PROJECT / "market_reference.json"
BAR_FIELDS = (*PRICE_COLUMNS, "volume_lots")
ETF_FIELDS = ("date", *BAR_FIELDS, "status")
MARKET_PREFIX = {"SSE": "sh", "SZSE": "sz"}


def get_calendar(start, end):
    calendar = ak.tool_trade_date_hist_sina()["trade_date"].tolist()
    if calendar != sorted(set(calendar)) or not calendar[0] <= start <= end <= calendar[-1]:
        raise ValueError("新浪交易日历范围不足或交易日不唯一")
    return [day for day in calendar if start <= day <= end]


def clean_etf(symbol, prices, sessions, listing, suspensions):
    if prices.keys() - set(sessions):
        raise ValueError(f"{symbol} 有落在非交易日的行情")
    rows = []
    for day in sessions:
        bar = prices.get(day)
        if day < listing:
            status = "pre_listing"
        elif day in suspensions:
            status = "suspended"
        else:
            status = "traded"
        if status == "traded" and (bar is None or bar[4] <= 0):
            raise ValueError(f"{symbol} {day} 缺少有效行情，需核实停牌或源数据")
        if status != "traded" and bar is not None:
            raise ValueError(f"{symbol} {day} 在停牌或上市前出现行情")
        values = bar if bar is not None else ("",) * len(BAR_FIELDS)
        rows.append((day, *values, status))
    return rows


def build(output_dir):
    reference = json.loads(REFERENCE.read_bytes())
    start = date.fromisoformat(reference["coverage"]["start"])
    end = date.fromisoformat(reference["coverage"]["end"])
    sessions = get_calendar(start, end)
    securities = reference["securities"]
    exports = {}
    for security in securities:
        symbol = security["symbol"]
        ticker = f"{MARKET_PREFIX[security['exchange']]}{symbol}"
        suspensions = {
            date.fromisoformat(row["date"])
            for row in reference["suspensions"] if row["symbol"] == symbol
        }
        prices = load_history(ticker, start, end)
        exports[symbol] = clean_etf(
            symbol, prices, sessions, date.fromisoformat(security["listing_date"]), suspensions
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{sessions[0]:%y%m%d}-{sessions[-1]:%y%m%d}"
    for symbol, rows in exports.items():
        name = f"{symbol}_{suffix}.csv"
        write_csv(output_dir / name, ETF_FIELDS, rows)
        print(f"{name}：{len(rows)} 行")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "data")
    build(parser.parse_args().output_dir)
