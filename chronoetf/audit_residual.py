"""独立核对 EWMA 残差记忆实验的标签、收缩、账户和因果边界。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import CODES, Market, load_market
from residual_policy import prepare, forecast, books, GAMMAS


def audit(folder):
    market, _, hashes = load_market()
    weekly = pd.read_csv(folder / "weekly_forecasts.csv", parse_dates=["signal_date", "execution_date"])
    states = pd.read_csv(folder / "tmave_error_states.csv", parse_dates=["date"])
    daily = pd.read_csv(folder / "daily_backtest.csv", parse_dates=["date"])
    metrics = pd.read_csv(folder / "portfolio_metrics.csv", dtype={"year": str})
    scores = pd.read_csv(folder / "forecast_scores.csv")
    losses = pd.read_csv(folder / "forecast_losses.csv")
    folds = pd.read_csv(folder / "train_folds.csv")
    shrink = pd.read_csv(folder / "shrinkage.csv")
    riskcal = pd.read_csv(folder / "risk_calibration.csv")
    assert riskcal.gap.dropna().abs().max() < 1e-6
    assert (folds.last_train_label < folds.first_signal_row).all()
    for item in folds.itertuples():
        section = shrink[(shrink.year == item.year) & (shrink.model == item.model)]
        assert (section.last_validation_label < item.first_signal_row).all()
        assert section.selected.sum() == section.paired_selected.sum() == 1
    # Rebuild past group returns independently from asset returns and yesterday's volatility weights.
    close = market.close
    logret = np.log(close).diff()
    inv = 1 / logret.iloc[:, :3].rolling(60, min_periods=55).std().shift()
    weights = inv.div(inv.sum(axis=1), axis=0)
    equity = np.log1p((weights * close.iloc[:, :3].pct_change(fill_method=None)).sum(axis=1, skipna=False))
    for group, returns in (("equity", equity), ("gold", logret.iloc[:, 3]), ("bond", logret.iloc[:, 4])):
        squared = returns.to_numpy() ** 2
        ewma = np.full(len(squared), np.nan)
        value, observations = np.nan, 0
        for i, point in enumerate(squared):
            if np.isfinite(point):
                value = point if not np.isfinite(value) else .94 * value + .06 * point
                observations += 1
            if observations >= 20:
                ewma[i] = value
        actual = pd.Series(squared).rolling(5, min_periods=5).mean().to_numpy()
        reference = np.log(np.maximum(actual, 1e-10) / np.maximum(np.r_[np.full(5, np.nan), ewma[:-5]], 1e-10))
        np.testing.assert_allclose(states.loc[states.group.eq(group), "mature_error"], reference, atol=1e-10, equal_nan=True)
    for (phase, model), section in losses.groupby(["phase", "model"]):
        ratio = section.realized.to_numpy() / section.forecast.to_numpy()
        expected = ratio - np.log(ratio) - 1
        np.testing.assert_allclose(section.qlike, expected, atol=1e-10)
        point = scores[(scores.phase == phase) & (scores.model == model) & scores.subset.eq("all")].iloc[0]
        np.testing.assert_allclose(point.qlike, expected.mean(), atol=1e-10)
    recomputed = 0
    for (phase, strategy), path in daily.groupby(["phase", "strategy"], sort=False):
        np.testing.assert_allclose(np.cumprod(1 + path.net_return), path.nav, atol=1e-10)
        np.testing.assert_allclose(path[[f"w_{c}" for c in CODES]].sum(axis=1) + path.cash_weight, 1, atol=1e-10)
        np.testing.assert_allclose(path.cost_nav, path.turnover * path.open_nav_before_trade * .0005, atol=1e-10)
        np.testing.assert_allclose(path[["pnl_equity", "pnl_gold", "pnl_bond", "pnl_fee"]].sum(axis=1), np.diff(np.r_[1, path.nav]), atol=1e-10)
        for year, part in [("all", path), *[(str(y), p) for y, p in path.groupby(path.date.dt.year)]]:
            r = part.net_return.to_numpy()
            wealth = np.cumprod(1 + r)
            depth = 1 - wealth / np.maximum.accumulate(np.maximum(wealth, 1))
            values = [wealth[-1] ** (252 / len(r)) - 1, r.std(ddof=1) * np.sqrt(252), -depth.max(), np.sort(depth)[-int(np.ceil(.05 * len(r))):].mean()]
            row = metrics[(metrics.phase == phase) & (metrics.strategy == strategy) & (metrics.year == year) & (metrics.one_way_bps == 5)].iloc[0]
            np.testing.assert_allclose(values, row[["cagr", "annual_vol", "max_drawdown", "cdar95"]].to_numpy(float), atol=1e-10)
            recomputed += 1
    frozen = pd.read_csv(Path(__file__).parent / "results/20261116-163605/portfolio_metrics.csv", dtype={"year": str})
    for level in (75, 90):
        for row in metrics[metrics.strategy.eq(f"common_ewma_{level}")].itertuples():
            old = frozen[(frozen.phase == row.phase) & (frozen.year == row.year) & (frozen.one_way_bps == row.one_way_bps) & frozen.strategy.eq(f"ewma_{level}")].iloc[0]
            np.testing.assert_allclose([row.cagr, row.annual_vol, row.max_drawdown], old[["cagr", "annual_vol", "max_drawdown"]].to_numpy(float), atol=1e-10)
    # Independently reconstruct paired one-SE choice using prequential predictions, not fitted training errors.
    checked_shrinkage = 0
    original = prepare(market)
    schedule, risk, _, _, base = original[:5]
    checked_labels = 0
    for item in schedule.itertuples():
        t = item.signal_row
        prices = market.opening.iloc[t + 1:t + 7, :4].to_numpy()
        if len(prices) != 6:
            continue
        weight = base[t, :4] / base[t, :4].sum()
        use = weight > 0
        moves = np.log(prices[1:, use] / prices[:-1, use]) @ weight[use]
        np.testing.assert_allclose(risk.realized[item.Index], np.mean(moves ** 2), atol=1e-14)
        checked_labels += 1
    predictions, original_folds, _, original_shrink = forecast(original[0], original[1], original[2])
    np.testing.assert_allclose(weekly[predictions.add_prefix("forecast_").columns], predictions,
                               rtol=1e-9, atol=1e-12, equal_nan=True)
    original_targets, original_cal = books(market, original[0], original[4], predictions, original[5])
    for item in original_folds[original_folds.year >= 2021].itertuples():
        schedule, risk = original[:2]
        mask = ((schedule.execution_date.dt.year >= max(2018, item.year - 3)) & (schedule.label_row < item.first_signal_row)
                & risk.realized.notna() & predictions.calibrated.notna() & predictions[f"{item.model}_raw"].notna())
        cal, raw, actual = predictions.calibrated[mask].to_numpy(), predictions[f"{item.model}_raw"][mask].to_numpy(), risk.realized[mask].to_numpy()
        rr = actual[:, None] / (cal[:, None] * (raw / cal)[:, None] ** GAMMAS)
        loss = rr - np.log(rr) - 1
        means = loss.mean(axis=0)
        best = means.argmin()
        differences = loss - loss[:, best, None]
        n = len(differences) // 4
        block_means = differences[:n * 4].reshape(n, 4, 5).mean(axis=1)
        se = block_means.std(axis=0, ddof=1) / np.sqrt(n)
        expected = GAMMAS[np.flatnonzero(means - means[best] <= se + 1e-14)[0]]
        assert item.paired_gamma == expected
        checked_shrinkage += 1
    boundary = market.close.index.searchsorted("2024-01-01")
    altered = Market(market.opening.copy(), market.close.copy(), market.volume.copy())
    rng = np.random.default_rng(20261116)
    shock = np.exp(rng.normal(0, .004, size=altered.close.iloc[boundary:].shape).cumsum(axis=0))
    altered.close.iloc[boundary:] *= shock
    altered.opening.iloc[boundary:] *= shock
    altered.volume.iloc[boundary:] *= 1.4
    changed = prepare(altered)
    new_predictions, new_folds, _, new_shrink = forecast(changed[0], changed[1], changed[2])
    new_targets, new_cal = books(altered, changed[0], changed[4], new_predictions, changed[5])
    before = original[0].signal_row < boundary
    pd.testing.assert_frame_equal(predictions[before], new_predictions[before])
    pd.testing.assert_frame_equal(original_folds[original_folds.year <= 2024], new_folds[new_folds.year <= 2024])
    pd.testing.assert_frame_equal(original_shrink[original_shrink.year <= 2024], new_shrink[new_shrink.year <= 2024])
    pd.testing.assert_frame_equal(original_cal, new_cal)
    for key in original_targets:
        np.testing.assert_allclose(original_targets[key][:boundary], new_targets[key][:boundary], atol=1e-11, rtol=0)
    result = {"status": "passed", "daily_rows": len(daily), "metric_groups": recomputed,
              "forecast_loss_rows": len(losses), "mature_error_rows": len(states), "train_boundaries": len(folds),
              "paired_shrink_choices": checked_shrinkage, "five_open_interval_labels": checked_labels,
              "causal_target_strategies": len(original_targets),
              "development_vol_match_max_abs_gap": float(riskcal.gap.dropna().abs().max()),
              "baseline_ewma_matches_frozen": True, "future_perturbation_start": "2024-01-01", "input_hashes": hashes}
    (folder / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)
