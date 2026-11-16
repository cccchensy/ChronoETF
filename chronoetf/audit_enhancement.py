"""独立复算收益增强账户、现金计息、历史一致性和未来扰动检查。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import CODES, Market, load_market, simulate
from enhancement import build_experiment, group_inputs
from volatility_policy import holding_schedule, risk_inputs
from tmave import period_ranges
from strategy import allocations, base_portfolio


def audit(folder):
    market, _, hashes = load_market()
    daily = pd.read_csv(folder / "daily_backtest.csv", parse_dates=["date"])
    metrics = pd.read_csv(folder / "portfolio_metrics.csv", dtype={"year": str})
    count = 0
    for (phase, key), path in daily.groupby(["phase", "strategy"], sort=False):
        nav = path.nav.to_numpy()
        ret = path.net_return.to_numpy()
        np.testing.assert_allclose(np.cumprod(1 + ret), nav, atol=1e-10)
        np.testing.assert_allclose(path[[f"w_{c}" for c in CODES]].sum(axis=1) + path.cash_weight, 1, atol=1e-10)
        np.testing.assert_allclose(path.cost_nav, path.turnover * path.open_nav_before_trade * .0005, atol=1e-10)
        np.testing.assert_allclose(path[["pnl_equity", "pnl_gold", "pnl_bond", "pnl_fee"]].sum(axis=1), np.diff(np.r_[1, nav]), atol=1e-10)
        for year, section in [("all", path), *[(str(y), p) for y, p in path.groupby(path.date.dt.year)]]:
            r = section.net_return.to_numpy()
            wealth = np.cumprod(1 + r)
            depth = 1 - wealth / np.maximum.accumulate(np.maximum(wealth, 1))
            reference = metrics.query("phase == @phase and strategy == @key and year == @year and one_way_bps == 5").iloc[0]
            values = [wealth[-1] ** (252 / len(r)) - 1, r.std(ddof=1) * np.sqrt(252), -depth.max(),
                      np.sort(depth)[-int(np.ceil(.05 * len(r))):].mean()]
            np.testing.assert_allclose(values, reference[["cagr", "annual_vol", "max_drawdown", "cdar95"]].to_numpy(dtype=float), atol=1e-10)
            count += 1
    frozen = pd.read_csv(Path(__file__).parent / "results/20261116-163605/portfolio_metrics.csv", dtype={"year": str})
    for _, row in metrics[metrics.strategy.isin(["full_75", "full_90", "budget_60"])].iterrows():
        old = frozen[(frozen.phase == row.phase) & (frozen.strategy == row.strategy) & (frozen.year == row.year) & (frozen.one_way_bps == row.one_way_bps)].iloc[0]
        cols = ["cagr", "annual_vol", "max_drawdown", "cdar95", "annual_turnover_two_way"]
        np.testing.assert_allclose(row[cols].to_numpy(float), old[cols].to_numpy(float), atol=1e-10)
    cash = pd.read_csv(folder / "cash_daily.csv", parse_dates=["date"])
    for (phase, key, rate), path in cash.groupby(["phase", "strategy", "cash_assumed_rate"]):
        if rate == 0:
            original = daily[(daily.phase == phase) & (daily.strategy == key)]
            np.testing.assert_allclose(path.nav, original.nav, atol=1e-10)
        else:
            previous_cash = (path.cash_weight * path.nav).shift().fillna(0)
            interest = previous_cash * np.expm1(np.log1p(rate) * path.date.diff().dt.days.fillna(0) / 365)
            np.testing.assert_allclose(interest, path.pnl_cash, atol=1e-10)
            np.testing.assert_allclose(path[["pnl_equity", "pnl_gold", "pnl_bond", "pnl_fee", "pnl_cash"]].sum(axis=1), np.diff(np.r_[1, path.nav]), atol=1e-10)
    zeros = np.zeros(market.close.shape)
    start, end = 1500, 1550
    pure_cash = simulate(market, zeros, "cash_test", start, end, 25, cash_annual_rate=.02)
    exact = 1.02 ** ((pure_cash.index - pure_cash.index[0]).days / 365)
    np.testing.assert_allclose(pure_cash.nav, exact, atol=1e-12)
    boundary = market.close.index.searchsorted("2024-01-01")
    original = build_experiment(market)
    altered = Market(market.opening.copy(), market.close.copy(), market.volume.copy())
    rng = np.random.default_rng(20261116)
    shocks = np.exp(rng.normal(0, .005, size=altered.close.iloc[boundary:].shape).cumsum(axis=0))
    altered.opening.iloc[boundary:] *= shocks
    altered.close.iloc[boundary:] *= shocks
    altered.volume.iloc[boundary:] *= 1.3
    changed = build_experiment(altered)
    for key in original[0]:
        np.testing.assert_allclose(original[0][key][:boundary], changed[0][key][:boundary], atol=1e-11, rtol=0)
    pd.testing.assert_frame_equal(original[-1], changed[-1])
    old, new = original[-2], changed[-2]
    before = old.signal_row < boundary
    pd.testing.assert_frame_equal(old[before], new[before])
    # 独立检查一个开盘到开盘股票标签，所有五个未来收益都在信号后成交起算。
    periods, _ = period_ranges(market.close.index, "2026-11-16")
    schedule = holding_schedule(market.close.index, periods)
    ids = schedule.signal_row.to_numpy()
    book = base_portfolio(market, allocations(market.close, ids)["budget_60"])[ids]
    raw = risk_inputs(market, schedule, book)
    grouped, _ = group_inputs(market, schedule, book, raw, 0)
    i = schedule.index[schedule.phase.eq("replay")][10]
    t = ids[i]
    w = book[i, :3] / book[i, :3].sum()
    prices = market.opening.iloc[t + 1:t + 7, :3].to_numpy()
    expected = np.mean((np.log(prices[1:] / prices[:-1]) @ w) ** 2)
    np.testing.assert_allclose(grouped.realized.iloc[i], expected, atol=1e-15)
    decisions = pd.read_csv(folder / "decisions.csv")
    constrained = decisions[decisions.strategy.str.contains("split")]
    assert constrained[["risk_ratio60", "risk_ratio20"]].max().max() <= 1 + 1e-6
    train = pd.read_csv(folder / "train_folds.csv", parse_dates=["first_signal_date", "last_label_end_open"])
    assert train.last_label_end_open.lt(train.first_signal_date).all()
    result = {"status": "passed", "daily_rows": len(daily), "metric_groups_independently_recomputed": count,
              "frozen_baselines": 3, "cash_rows": len(cash), "causal_target_strategies": len(original[0]),
              "future_perturbation_start": "2024-01-01", "group_open_label": "passed", "cash_calendar_compounding": "passed",
              "train_boundary_rows": len(train), "input_hashes": hashes}
    (folder / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)
