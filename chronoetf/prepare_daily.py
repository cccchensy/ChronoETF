"""获取并校验腾讯财经 ETF 后复权日线。"""

import csv
import json
from datetime import date, timedelta
from decimal import Decimal
from time import sleep
from urllib.parse import urlencode
from urllib.request import urlopen


ENDPOINT = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fqkline/get"
PRICE_COLUMNS = ("open", "high", "low", "close")


def download(ticker, start, stop, adjustment="hfq"):
    # 不复权价仅供敏感性实验在内存中使用。
    endpoint = ENDPOINT if adjustment else ENDPOINT.replace("https://", "http://", 1)
    query = urlencode({"param": f"{ticker},day,{start},{stop},640,{adjustment}"}, safe=",")
    for attempt in range(2):
        try:
            with urlopen(f"{endpoint}?{query}", timeout=20) as response:
                payload = json.load(response)
            break
        except OSError as exc:
            if attempt:
                raise OSError(f"{ticker} {start} 至 {stop} 下载失败：{exc}") from exc
            sleep(1)
    series = (payload.get("data") or {}).get(ticker) or {}
    key = f"{adjustment}day"
    lines = series.get(key)
    if payload.get("code") != 0 or not isinstance(lines, list) or len(lines) >= 640:
        raise ValueError(f"{ticker} {start} 至 {stop} 的 {key} 行情响应无效")
    return lines


def clean_bars(lines, start, stop, ticker):
    bars = {}
    for trade_date, opening, closing, high, low, volume, *_ in lines:
        day = date.fromisoformat(trade_date)
        # 接口可能包含末日，分段按左闭右开合并。
        if day == stop:
            continue
        bar = tuple(map(Decimal, (opening, high, low, closing, volume)))
        if not all(value.is_finite() for value in bar):
            raise ValueError(f"{ticker} {day} 存在空值或非有限数值")
        op, hi, lo, cl, vol = bar
        if not (start <= day < stop and 0 < lo <= min(op, cl) <= max(op, cl) <= hi and vol >= 0):
            raise ValueError(f"{ticker} {day} 日期、OHLC 或成交量异常")
        if day in bars and bars[day] != bar:
            raise ValueError(f"{ticker} {day} 存在内容冲突的重复行情")
        bars[day] = bar
    return bars


def load_history(ticker, start, end, adjustment="hfq"):
    bars = {}
    terminal = end + timedelta(days=1)
    while start < terminal:
        # 两年内的交易日数低于接口单次 640 条的上限。
        stop = min(date(start.year + 2, 1, 1), terminal)
        bars.update(clean_bars(download(ticker, start, stop, adjustment), start, stop, ticker))
        start = stop
    return bars


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)
