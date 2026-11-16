"""Frozen candidate: cross-asset risk alerts and simple-memory controls.

Run from repository root: python -B -X utf8 chronoetf/transfer_research.py
Only compact evidence is intended for publication; full accounts remain local.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from backtest import CODES, Market, load_market
from prepare_daily import load_history, write_csv, clean_bars, ENDPOINT
from residual_policy import prepare, forecast, books, prediction_scores
from sleeves import evaluate_sleeves
from strategy import circular_blocks, bootstrap_difference, performance_stats
from volatility_policy import tail_stats
from horizon import holm_adjust

ROOT = Path(__file__).resolve().parent
PANELS = {"style": ("510050", "510180", "510880"), "sector": ("159928", "159929", "159930")}
READOUTS = ("joint_static", "joint_fixed", "joint_trade", "price", "joint_full", "lag", "rolling", "exponential")
MAIN = "joint_full_paired"
CONTROLS = ("calibrated", "lag_paired", "rolling_paired", "exponential_paired")
METHODS = ("ewma", "calibrated", *(f"{m}_paired" for m in READOUTS))
ECONOMIC = (*CONTROLS, MAIN)


def unchanged_adjustment_history(code):
    """Handle Tencent's day key only when all three requested adjustment modes agree."""
    bars, notes = {}, []
    start, terminal = date(2015, 1, 5), date(2026, 10, 31)
    while start < terminal:
        stop = min(date(start.year + 2, 1, 1), terminal)
        responses = []
        for adjustment in ("hfq", "qfq", ""):
            query = urlencode({"param": f"sz{code},day,{start},{stop},640,{adjustment}"}, safe=",")
            with urlopen(f"{ENDPOINT}?{query}", timeout=20) as response:
                payload = json.load(response)
            series = payload.get("data", {}).get("sz" + code, {})
            assert payload.get("code") == 0 and "day" in series and len(series["day"]) < 640
            responses.append(clean_bars(series["day"], start, stop, code))
        assert responses[0] == responses[1] == responses[2], f"Unresolved adjustment difference: {code}"
        bars.update(responses[0])
        notes.append({"start": str(start), "stop": str(stop), "returned_key": "day", "hfq_qfq_raw_equal": True, "rows": len(responses[0])})
        start = stop
    return bars, notes


def transfer_data():
    original, _, hashes = load_market()
    dates = original.close.index
    folder = ROOT / "transfer_data"
    folder.mkdir(exist_ok=True)

    def fetch(code):
        path = folder / f"{code}_150105-261030.csv"
        if not path.exists():
            if code in ("159929", "159930"):
                prices, notes = unchanged_adjustment_history(code)
                (folder / f"{code}_adjustment.json").write_text(json.dumps(notes, indent=2), encoding="utf8")
            else:
                prices = load_history(("sh" if code.startswith("5") else "sz") + code, date(2015, 1, 5), date(2026, 10, 30))
            rows = [(d.date(), *(prices[d.date()] if d.date() in prices else ("",) * 5),
                     "traded" if d.date() in prices else "missing") for d in dates]
            write_csv(path, ("date", "open", "high", "low", "close", "volume_lots", "status"), rows)
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        if code in ("159929", "159930"):
            notes = json.loads((folder / f"{code}_adjustment.json").read_text(encoding="utf8"))
            assert all(item["hfq_qfq_raw_equal"] for item in notes)
        assert frame.index.is_unique and frame.index.equals(dates), code
        valid = frame[["open", "high", "low", "close", "volume_lots"]].notna().all(axis=1)
        f = frame[valid]
        assert (f.low > 0).all() and (f.volume_lots > 0).all(), code
        assert (f.low <= f[["open", "close"]].min(axis=1)).all(), code
        assert (f.high >= f[["open", "close"]].max(axis=1)).all(), code
        formation = valid & (dates < "2018-01-01")
        summary = {"symbol": code, "rows": len(frame), "valid_rows": int(valid.sum()),
                   "missing_rows": int((~valid).sum()), "formation_rows": int(formation.sum()),
                   "coverage": valid.mean(), "max_abs_log_return": np.log(frame.close).diff().abs().max(),
                   "first": str(frame.index[valid][0].date()), "last": str(frame.index[valid][-1].date()),
                   "eligible": bool(valid.mean() >= .99 and formation.sum() >= 500),
                   "adjustment_response": "day; hfq/qfq/raw requests agree" if code in ("159929", "159930") else "hfqday",
                   "sha256": sha256(path.read_bytes()).hexdigest()}
        print(f"Input {code}: {summary['valid_rows']}/{len(frame)} dates", flush=True)
        return code, frame, summary

    with ThreadPoolExecutor(max_workers=2) as pool:
        fetched = list(pool.map(fetch, [code for panel in PANELS.values() for code in panel]))
    quality = pd.DataFrame([x[2] for x in fetched])
    frames = {code: frame for code, frame, _ in fetched}
    markets = {"original": original}
    mapping = {"original": dict(zip(CODES, CODES))}
    for name, stocks in PANELS.items():
        if not quality.set_index("symbol").loc[list(stocks), "eligible"].all():
            continue
        fields = []
        for attr, field in (("opening", "open"), ("close", "close"), ("volume", "volume_lots")):
            table = getattr(original, attr).copy()
            for slot, code in zip(CODES[:3], stocks):
                table[slot] = frames[code][field]
            fields.append(table)
        markets[name] = Market(*fields)
        mapping[name] = dict(zip(CODES, (*stocks, *CODES[3:])))
    return markets, quality, mapping, hashes


def simple_features(prepared):
    schedule, _, features, states, base = prepared[:5]
    ids = schedule.signal_row.to_numpy()
    weights = np.column_stack((base[ids, :3].sum(axis=1), base[ids, 3]))
    weights /= weights.sum(axis=1, keepdims=True)
    memories = {key: [] for key in ("lag", "rolling", "exponential")}
    for field in ("candidate", "price_pressure_candidate"):
        values = pd.DataFrame(states[field].fillna(0).to_numpy().reshape(-1, 3))
        variants = {
            "lag": [values.shift(n).fillna(0) for n in (1, 5)],
            "rolling": [values.rolling(n, min_periods=1).mean() for n in (5, 30)],
            "exponential": [values.ewm(halflife=n, adjust=False).mean() for n in (5, 30)],
        }
        for name, collection in variants.items():
            memories[name].extend([(matrix.to_numpy()[ids, :2] * weights).sum(axis=1) for matrix in collection])
    for name, series in memories.items():
        features[name] = np.column_stack((features["joint_static"], *series))
        assert features[name].shape == features["joint_full"].shape
    return features


def alerts(schedule, risk, predictions):
    valid = schedule.label_in_phase & np.isfinite(risk.realized)
    development = valid & schedule.phase.eq("development")
    truth_threshold = risk.realized[development].quantile(.90)
    rows, thresholds = [], []
    for model in METHODS:
        for lane in ("same_risk", "same_dev_budget"):
            cutoff = truth_threshold if lane == "same_risk" else predictions.loc[development, model].quantile(.90)
            thresholds.append({"model": model, "lane": lane, "truth_threshold": truth_threshold,
                               "forecast_threshold": cutoff, "fit_last_label_row": int(schedule.loc[development, "label_row"].max())})
            for i in schedule.index[valid & schedule.phase.isin(["development", "replay"])]:
                rows.append({"phase": schedule.phase[i], "signal_date": schedule.signal_date[i],
                             "year": schedule.execution_date[i].year, "model": model, "lane": lane,
                             "truth": bool(risk.realized[i] > truth_threshold),
                             "alert": bool(predictions.loc[i, model] > cutoff),
                             "severity": max(risk.realized[i] / truth_threshold - 1, 0)})
    detail = pd.DataFrame(rows)
    scores = []
    for tag, section in detail.groupby(["phase", "model", "lane"], sort=False):
        for year, part in [("all", section), *[(str(y), g) for y, g in section.groupby("year")]]:
            truth, alarm = part.truth.to_numpy(), part.alert.to_numpy()
            tp, fp, fn, tn = ((truth & alarm).sum(), (~truth & alarm).sum(), (truth & ~alarm).sum(), (~truth & ~alarm).sum())
            scores.append(dict(zip(("phase", "model", "lane"), tag)) | {
                "year": year, "weeks": len(part), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                "alert_rate": alarm.mean(), "missed_severity": (part.severity.to_numpy() * ~alarm).mean()})
    return detail, pd.DataFrame(scores), pd.DataFrame(thresholds)


def prediction_tests(losses):
    rows, rng = [], np.random.default_rng(42)
    panel = losses.query("phase == 'replay' and basket != 'original'")
    # Average losses within the same date first; resample weeks, never individual basket rows.
    common = panel.groupby(["signal_date", "model"]).agg(qlike=("qlike", "mean"), baskets=("basket", "nunique"))
    wide = common.qlike.unstack()
    assert common.baskets.eq(panel.basket.nunique()).all()
    scopes = [("new_pooled", wide)]
    scopes += [(name, part.pivot(index="signal_date", columns="model", values="qlike")) for name, part in losses.query("phase == 'replay'").groupby("basket")]
    for scope, values in scopes:
        for block in (2, 4, 8):
            ix = circular_blocks(rng, len(values), block)
            for benchmark in CONTROLS:
                delta = values[benchmark].to_numpy() - values[MAIN].to_numpy()
                rows.append({"scope": scope, "benchmark": benchmark, "block_weeks": block, "weeks": len(delta),
                             **bootstrap_difference(delta.mean(), delta[ix].mean(axis=1), True)})
    result = pd.DataFrame(rows)
    for _, part in result.groupby(["scope", "block_weeks"]):
        result.loc[part.index, "p_holm"] = holm_adjust(part.p_raw.to_numpy())
    return result


def alert_tests(detail):
    rng, rows = np.random.default_rng(42), []
    for scope in ["new_pooled", *detail.basket.unique()]:
        section = detail.query("phase == 'replay' and lane == 'same_dev_budget'")
        section = section[section.basket.ne("original") if scope == "new_pooled" else section.basket.eq(scope)]
        truth = section.drop_duplicates(["signal_date", "basket"]).pivot(index="signal_date", columns="basket", values="truth").to_numpy(bool)
        alarms = {name: section[section.model.eq(name)].pivot(index="signal_date", columns="basket", values="alert").to_numpy(bool) for name in (MAIN, "calibrated")}

        def rates(a, y):
            axes = (-2, -1)
            return {"recall": (a & y).sum(axis=axes) / np.maximum(y.sum(axis=axes), 1),
                    "false_positive_rate": (a & ~y).sum(axis=axes) / np.maximum((~y).sum(axis=axes), 1),
                    "alert_rate": a.mean(axis=axes)}
        aa, bb = rates(alarms[MAIN], truth), rates(alarms["calibrated"], truth)
        for block in (2, 4, 8):
            ix = circular_blocks(rng, len(truth), block)
            a, b = rates(alarms[MAIN][ix], truth[ix]), rates(alarms["calibrated"][ix], truth[ix])
            for metric in aa:
                rows.append({"scope": scope, "metric": metric, "block_weeks": block,
                             **bootstrap_difference(aa[metric] - bb[metric], a[metric] - b[metric], True)})
    result = pd.DataFrame(rows)
    for _, part in result.groupby(["scope", "block_weeks"]):
        result.loc[part.index, "p_holm"] = holm_adjust(part.p_raw.to_numpy())
    return result


def economic_tests(all_paths):
    rng, rows = np.random.default_rng(42), []
    def stats(r):
        nav = np.cumprod(1 + r, axis=-1)
        depth = 1 - nav / np.maximum.accumulate(np.maximum(nav, 1), axis=-1)
        return {"cagr": np.expm1(252 * np.log1p(r).mean(axis=-1)), "annual_vol": r.std(axis=-1, ddof=1) * np.sqrt(252),
                "max_drawdown": -depth.max(axis=-1), "cdar95": np.sort(depth, axis=-1)[..., -int(np.ceil(.05 * r.shape[-1])):].mean(axis=-1)}
    for basket, paths in all_paths.items():
        for level in (75, 90):
            a = paths["replay", f"matched_{MAIN}_{level}"].net_return.to_numpy()
            b = paths["replay", f"matched_calibrated_{level}"].net_return.to_numpy()
            aa, bb = stats(a), stats(b)
            for block in (10, 20, 40):
                ix = circular_blocks(rng, len(a), block)
                boot_a, boot_b = stats(a[ix]), stats(b[ix])
                for metric in aa:
                    rows.append({"basket": basket, "level": level, "metric": metric, "block_days": block,
                                 **bootstrap_difference(aa[metric] - bb[metric], boot_a[metric] - boot_b[metric], True)})
    result = pd.DataFrame(rows)
    for _, part in result.groupby("block_days"):
        result.loc[part.index, "p_holm"] = holm_adjust(part.p_raw.to_numpy())
    return result


def gates(scores, tests, alert_scores, metrics):
    output = []
    for basket in PANELS:
        s = scores.query("basket == @basket and phase == 'replay' and subset == 'all'").set_index("model")
        a = alert_scores.query("basket == @basket and phase == 'replay' and year == 'all' and lane == 'same_dev_budget'").set_index("model")
        output.append({"basket": basket, "gate": "forecast_better_than_calibrated", "passed": s.loc[MAIN, "qlike"] < s.loc["calibrated", "qlike"]})
        for control in CONTROLS[1:]:
            output.append({"basket": basket, "gate": f"not_worse_than_{control}", "passed": s.loc[MAIN, "qlike"] <= s.loc[control, "qlike"]})
        output.append({"basket": basket, "gate": "alert_recall", "passed": a.loc[MAIN, "recall"] >= a.loc["calibrated", "recall"]})
        for metric in ("false_positive_rate", "alert_rate"):
            output.append({"basket": basket, "gate": metric, "passed": a.loc[MAIN, metric] <= a.loc["calibrated", metric]})
        for level in (75, 90):
            for fee in (5, 10):
                m = metrics.query("basket == @basket and phase == 'replay' and year == 'all' and one_way_bps == @fee").set_index("strategy")
                full, base = m.loc[f"matched_{MAIN}_{level}"], m.loc[f"matched_calibrated_{level}"]
                passed = (full.cagr >= base.cagr + .003 and full.annual_vol <= base.annual_vol
                          and full.max_drawdown >= base.max_drawdown and full.cdar95 <= base.cdar95)
                output.append({"basket": basket, "gate": f"economic_{level}_{fee}bp", "passed": passed})
    point = tests.query("scope == 'new_pooled' and block_weeks == 4 and benchmark == 'calibrated'").iloc[0]
    output.append({"basket": "new_pooled", "gate": "main_forecast_significant", "passed": point.difference > 0 and point.p_holm < .05})
    return pd.DataFrame(output)


def run(folder):
    markets, quality, mapping, original_hashes = transfer_data()
    folder.mkdir(parents=True, exist_ok=False)
    quality.to_csv(folder / "data_quality.csv", index=False)
    source_notes = {code: json.loads((ROOT / "transfer_data" / f"{code}_adjustment.json").read_text(encoding="utf8"))
                    for code in ("159929", "159930")}
    (folder / "adjustment_evidence.json").write_text(json.dumps(source_notes, indent=2), encoding="utf8")
    if set(PANELS) - set(markets):
        raise ValueError("Fixed asset basket fails data coverage; see data_quality.csv. No replacement selected.")
    collected, all_paths, contexts = {}, {}, {}
    def keep(name, frame, basket):
        collected.setdefault(name, []).append(frame.assign(basket=basket))
    for basket, market in markets.items():
        print(f"Fitting frozen rules: {basket}", flush=True)
        context = prepare(market)
        schedule, risk, _, states, base, periods, _ = context
        features = simple_features(context)
        predictions, folds, coefficients, shrinkage = forecast(schedule, risk, features, READOUTS)
        scores, losses = prediction_scores(schedule, risk, predictions.loc[:, list(METHODS)])
        detail, alarm_scores, thresholds = alerts(schedule, risk, predictions)
        target, calibration = books(market, schedule, base, predictions, periods, METHODS, ECONOMIC)
        metrics, paths = evaluate_sleeves(market, target, {}, periods)
        metrics = metrics.merge(tail_stats(paths), how="left", on=["phase", "strategy", "year", "one_way_bps"], validate="one_to_one")
        for name, frame in (("forecast_scores", scores), ("forecast_losses", losses), ("train_folds", folds),
                            ("coefficients", coefficients), ("shrinkage", shrinkage), ("alert_detail", detail),
                            ("alert_scores", alarm_scores), ("alert_thresholds", thresholds), ("portfolio_metrics", metrics),
                            ("risk_calibration", calibration)):
            keep(name, frame, basket)
        evidence = pd.concat([schedule, risk, predictions.add_prefix("forecast_")], axis=1)
        evidence.loc[~schedule.label_in_phase, "realized"] = np.nan
        keep("weekly_forecasts", evidence, basket)
        # Keep full accounts local only, with actual asset identifiers replacing internal positional slots.
        account = pd.concat(paths, names=["phase", "strategy"]).reset_index()
        account.rename(columns={f"{prefix}_{slot}": f"{prefix}_{actual}" for prefix in ("w", "target") for slot, actual in mapping[basket].items()}, inplace=True)
        account.to_csv(folder / f"local_accounts_{basket}.csv.gz", index=False, compression="gzip")
        all_paths[basket], contexts[basket] = paths, (context, predictions, target, thresholds)
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
    tables["forecast_tests"] = prediction_tests(tables["forecast_losses"])
    tables["alert_tests"] = alert_tests(tables["alert_detail"])
    tables["economic_tests"] = economic_tests(all_paths)
    tables["acceptance_gates"] = gates(tables["forecast_scores"], tables["forecast_tests"], tables["alert_scores"], tables["portfolio_metrics"])
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    manifest = {"status": "candidate_not_accepted", "protocol": "TRANSFER_RESEARCH_PROTOCOL.md",
                "protocol_sha256": sha256((ROOT.parent / "TRANSFER_RESEARCH_PROTOCOL.md").read_text(encoding="utf8").encode()).hexdigest(),
                "input_hashes_original": original_hashes, "input_hashes_new": dict(zip(quality.symbol, quality.sha256)),
                "asset_slot_mapping": mapping, "primary_unit": "same-week average across two correlated new baskets",
                "new_prospective_sessions": 0, "readouts": READOUTS}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(tables["forecast_scores"].query("phase == 'replay' and subset == 'all'")[["basket", "model", "qlike"]].to_string(index=False), flush=True)
    print(f"Results: {folder}", flush=True)
    return markets, contexts, tables, all_paths


def audit(folder):
    """Check only new paths and the evidence used in this candidate decision."""
    if not all((folder / f"local_accounts_{name}.csv.gz").exists() for name in ("original", *PANELS)):
        raise ValueError("Compact published evidence omits large accounts. Run transfer_research.py first, then audit its new result directory.")
    markets, quality, _, hashes = transfer_data()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
    assert hashes == manifest["input_hashes_original"]
    assert dict(zip(quality.symbol, quality.sha256)) == manifest["input_hashes_new"]
    weekly = pd.read_csv(folder / "weekly_forecasts.csv", parse_dates=["signal_date", "execution_date"])
    metrics = pd.read_csv(folder / "portfolio_metrics.csv", dtype={"year": str})
    folds = pd.read_csv(folder / "train_folds.csv")
    shrink = pd.read_csv(folder / "shrinkage.csv")
    thresholds = pd.read_csv(folder / "alert_thresholds.csv")
    calibration = pd.read_csv(folder / "risk_calibration.csv")
    assert calibration.gap.dropna().abs().max() < 1e-6
    assert (folds.last_train_label < folds.first_signal_row).all()
    for row in folds.itertuples():
        h = shrink[(shrink.basket == row.basket) & (shrink.model == row.model) & (shrink.year == row.year)]
        assert (h.last_validation_label < row.first_signal_row).all() and h.paired_selected.sum() == 1
    checked_labels, checked_accounts, checked_metrics, checked_memories = 0, 0, 0, 0
    sector_context = None
    for basket, market in markets.items():
        part = weekly[weekly.basket.eq(basket)].reset_index(drop=True)
        context = prepare(market)
        features = simple_features(context)
        schedule, risk, _, states, base = context[:5]
        if basket == "sector":
            sector_context = context
        for row in schedule.itertuples():
            t = row.signal_row
            prices = market.opening.iloc[t + 1:t + 7, :4].to_numpy()
            if len(prices) != 6:
                continue
            w = base[t, :4] / base[t, :4].sum()
            use = w > 0
            label = np.mean((np.log(prices[1:, use] / prices[:-1, use]) @ w[use]) ** 2)
            np.testing.assert_allclose(risk.realized[row.Index], label, atol=1e-14, equal_nan=True)
            checked_labels += 1
        # Independent lag, arithmetic mean and EW recursion at three signal positions per basket.
        for position in (100, 300, len(schedule) - 2):
            t = int(schedule.signal_row[position])
            w = np.array([base[t, :3].sum(), base[t, 3]])
            w /= w.sum()
            expected = {name: [] for name in ("lag", "rolling", "exponential")}
            for field in ("candidate", "price_pressure_candidate"):
                values = states[field].fillna(0).to_numpy().reshape(-1, 3)[:t + 1, :2]
                expected["lag"].extend(values[-1 - n] @ w for n in (1, 5))
                expected["rolling"].extend(values[-n:].mean(axis=0) @ w for n in (5, 30))
                for n in (5, 30):
                    alpha = 1 - 2 ** (-1 / n)
                    average = values[0].copy()
                    for value in values[1:]:
                        average = (1 - alpha) * average + alpha * value
                    expected["exponential"].append(average @ w)
            for name in expected:
                np.testing.assert_allclose(features[name][position, -4:], expected[name], atol=1e-12)
                checked_memories += 1
        for row in thresholds[thresholds.basket.eq(basket)].itertuples():
            dev = part.phase.eq("development") & part.realized.notna()
            expected_truth = part.loc[dev, "realized"].quantile(.90)
            expected_cut = expected_truth if row.lane == "same_risk" else part.loc[dev, "forecast_" + row.model].quantile(.90)
            np.testing.assert_allclose([row.truth_threshold, row.forecast_threshold], [expected_truth, expected_cut], atol=1e-12)
            assert row.fit_last_label_row < part.loc[part.phase.eq("replay"), "signal_row"].iloc[0]
        accounts = pd.read_csv(folder / f"local_accounts_{basket}.csv.gz", parse_dates=["date"])
        for (phase, strategy), path in accounts.groupby(["phase", "strategy"]):
            np.testing.assert_allclose(np.cumprod(1 + path.net_return), path.nav, atol=1e-10)
            wcols = [c for c in path if c.startswith("w_")]
            np.testing.assert_allclose(path[wcols].sum(axis=1) + path.cash_weight, 1, atol=1e-10)
            np.testing.assert_allclose(path.cost_nav, path.turnover * path.open_nav_before_trade * .0005, atol=1e-10)
            np.testing.assert_allclose(path[["pnl_equity", "pnl_gold", "pnl_bond", "pnl_fee"]].sum(axis=1), np.diff(np.r_[1, path.nav]), atol=1e-10)
            net = path.net_return.to_numpy()
            nav = np.cumprod(1 + net)
            depth = 1 - nav / np.maximum.accumulate(np.maximum(nav, 1))
            point = metrics.query("basket == @basket and phase == @phase and strategy == @strategy and year == 'all' and one_way_bps == 5").iloc[0]
            expected = [nav[-1] ** (252 / len(net)) - 1, net.std(ddof=1) * np.sqrt(252), -depth.max(), np.sort(depth)[-int(np.ceil(.05 * len(depth))):].mean()]
            np.testing.assert_allclose(point[["cagr", "annual_vol", "max_drawdown", "cdar95"]].to_numpy(float), expected, atol=1e-10)
            checked_metrics += 1
        checked_accounts += len(accounts)
    # Existing full-model columns must remain identical on the old basket after the optional API extension.
    old = pd.read_csv(ROOT / "results/20261116-190622/weekly_forecasts.csv")
    current = weekly[weekly.basket.eq("original")].reset_index(drop=True)
    for name in ("ewma", "calibrated", MAIN, "joint_fixed_paired", "joint_trade_paired", "price_paired"):
        np.testing.assert_allclose(current["forecast_" + name], old["forecast_" + name], atol=1e-12, equal_nan=True)
    # New basket future perturbation checks every new/simple readout without re-running old audits.
    market = markets["sector"]
    before = market.close.index < "2024-01-01"
    changed = Market(market.opening.copy(), market.close.copy(), market.volume.copy())
    rng = np.random.default_rng(20261116)
    shock = np.exp(rng.normal(0, .004, size=changed.close.loc[~before].shape).cumsum(axis=0))
    changed.opening.loc[~before] *= shock
    changed.close.loc[~before] *= shock
    changed.volume.loc[~before] *= 1.4
    original_pred, original_fold, _, original_shrink = forecast(sector_context[0], sector_context[1], sector_context[2], READOUTS)
    new_context = prepare(changed)
    new_pred, new_fold, _, new_shrink = forecast(new_context[0], new_context[1], simple_features(new_context), READOUTS)
    early = sector_context[0].signal_date.lt("2024-01-01")
    pd.testing.assert_frame_equal(original_pred[early], new_pred[early])
    pd.testing.assert_frame_equal(original_fold[original_fold.year <= 2024], new_fold[new_fold.year <= 2024])
    pd.testing.assert_frame_equal(original_shrink[original_shrink.year <= 2024], new_shrink[new_shrink.year <= 2024])
    np.testing.assert_allclose(sector_context[4][before], new_context[4][before], atol=1e-12)
    saved = weekly[weekly.basket.eq("sector")].reset_index(drop=True)
    np.testing.assert_allclose(saved[original_pred.add_prefix("forecast_").columns], original_pred, atol=1e-12, equal_nan=True)
    # Reconstruct alert confusion counts from archived predictions, independently of alert generation.
    alarm_scores = pd.read_csv(folder / "alert_scores.csv", dtype={"year": str})
    checked_alerts = 0
    for row in thresholds.itertuples():
        for phase in ("development", "replay"):
            part = weekly[(weekly.basket == row.basket) & (weekly.phase == phase) & weekly.realized.notna()]
            y, a = part.realized > row.truth_threshold, part["forecast_" + row.model] > row.forecast_threshold
            counts = [(y & a).sum(), (~y & a).sum(), (y & ~a).sum(), (~y & ~a).sum()]
            got = alarm_scores.query("basket == @row.basket and model == @row.model and lane == @row.lane and phase == @phase and year == 'all'").iloc[0]
            np.testing.assert_array_equal(got[["tp", "fp", "fn", "tn"]].to_numpy(int), counts)
            checked_alerts += 1
    result = {"status": "passed", "five_open_interval_labels": checked_labels, "daily_account_rows": checked_accounts,
              "full_period_account_metrics": checked_metrics, "manual_memory_checks": checked_memories,
              "alert_confusion_tables": checked_alerts, "training_boundaries": len(folds),
              "future_perturbation_basket": "sector", "future_perturbation_start": "2024-01-01",
              "original_forecast_columns_unchanged": True, "development_vol_max_gap": float(calibration.gap.abs().max())}
    (folder / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


def summarize(folder):
    """Compact candidate-only publication; never updates the two accepted documents."""
    from report import fonts
    import matplotlib.pyplot as plt
    fonts()
    scores = pd.read_csv(folder / "forecast_scores.csv")
    metrics = pd.read_csv(folder / "portfolio_metrics.csv", dtype={"year": str})
    alert = pd.read_csv(folder / "alert_scores.csv", dtype={"year": str})
    tests = pd.read_csv(folder / "forecast_tests.csv")
    gates_frame = pd.read_csv(folder / "acceptance_gates.csv")
    q = scores.query("phase == 'replay' and subset == 'all'").pivot(index="basket", columns="model", values="qlike")
    models = ["calibrated", "lag_paired", "rolling_paired", "exponential_paired", "price_paired", "joint_trade_paired", MAIN]
    labels = ["校准 EWMA", "固定滞后", "滚动平均", "指数平均", "单价格记忆", "仅交易日", "完整 T-MAVE"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)
    for ax, basket, title in zip(axes, PANELS, ["宽基 / 红利组", "消费 / 医药 / 能源组"]):
        values = q.loc[basket, models].to_numpy()
        ax.barh(labels, values, color=["#889799"] * 6 + ["#147d72"])
        ax.invert_yaxis()
        for i, value in enumerate(values):
            ax.text(value + .006, i, f"{value:.4f}", va="center")
        ax.set(xlim=(0, max(values) * 1.18), title=title, xlabel="QLIKE（越低越好）")
    fig.suptitle("固定规则跨资产验证 · 2024–2026 历史迁移，每组 145 周")
    fig.tight_layout()
    fig.savefig(folder / "transfer_forecasts.png", dpi=150)
    plt.close(fig)
    selected = metrics.query("phase == 'replay' and year == 'all' and one_way_bps == 5 and strategy in ['matched_calibrated_75','matched_joint_full_paired_75','matched_calibrated_90','matched_joint_full_paired_90']")
    qtable = "| 预测模型 | 原篮子诊断 | 宽基/红利迁移 | 行业迁移 |\n|---|---:|---:|---:|\n"
    for model, name in zip(models, labels):
        qtable += f"| {name} | {q.loc['original',model]:.6f} | {q.loc['style',model]:.6f} | {q.loc['sector',model]:.6f} |\n"
    atable = "| 新篮子 | 模型 | 高风险事件 | 命中 | 误报 | 漏报 | 提示频率 |\n|---|---|---:|---:|---:|---:|---:|\n"
    for row in alert.query("basket != 'original' and phase == 'replay' and year == 'all' and lane == 'same_dev_budget' and model in ['calibrated','joint_full_paired']").itertuples():
        atable += f"| {'宽基/红利' if row.basket=='style' else '行业'} | {'校准 EWMA' if row.model=='calibrated' else '完整 T-MAVE'} | {row.tp+row.fn} | {row.tp} | {row.fp} | {row.fn} | {row.alert_rate:.2%} |\n"
    etable = "| 篮子 | 风险匹配策略 | 净年化 | 波动 | 最大回撤 |\n|---|---|---:|---:|---:|\n"
    for row in selected[selected.basket.ne("original")].itertuples():
        label = ('校准 EWMA' if 'calibrated' in row.strategy else '完整 T-MAVE') + ' · ' + row.strategy.rsplit('_', 1)[-1]
        etable += f"| {'宽基/红利' if row.basket=='style' else '行业'} | {label} | {row.cagr:.4%} | {row.annual_vol:.4%} | {row.max_drawdown:.4%} |\n"
    p = tests.query("scope == 'new_pooled' and block_weeks == 4 and benchmark == 'calibrated'").iloc[0]
    gains = {basket: 1 - q.loc[basket, MAIN] / q.loc[basket, "calibrated"] for basket in PANELS}
    body = f"""# 候选研究结果：风险预警与跨资产验证

**状态：待用户决定，未写入 HR 文档或技术手册，未替换正式基线。** 本轮是预先固定规则的跨资产历史迁移，不是实盘或未来独立验证。

**建议：保留研究证据，暂不将完整候选升级为正式模型。** 行业组出现预测改善，但宽基/红利组没有复制该优势；主预警没有普遍减少漏报，低风险收益提高门槛也未通过。

完整 T-MAVE 相对校准 EWMA：宽基/红利组 QLIKE 改善 {gains['style']:.2%}，行业组改善 {gains['sector']:.2%}。两组平均损失差为 {p.difference:.6f}，4 周同步区块 95% 区间 [{p.lower:.6f}, {p.upper:.6f}]，Holm p={p.p_holm:.4f}。两组共享市场日期、黄金和国债，仅有 145 个同期周，不能把 290 个篮子周当作独立样本。

## 1. 风险预测与简单记忆

{qtable}
![固定规则跨资产比较](transfer_forecasts.png)

原篮子结果只是旧数据上的新增简单对照诊断。新篮子中的单价格与仅交易日和完整模型十分接近；完整机制各部分的独立必要性仍需证据。三种简单记忆都使用相同 7 个静态输入及 4 个额外记忆特征，相同训练和收缩。没有根据测试结果继续调整窗口。

行业组相对校准 EWMA 的 QLIKE 在 2024、2025、2026 三年都更低；宽基/红利组在 2024 年更低、2025 年更高、2026 年相同。历史规则为宽基/红利组在 2026 年选择 gamma=0，退回校准 EWMA；行业组在三个评价年度均选 gamma=0.5。收缩能撤销不受历史证据支持的修正，但不能保证下一年一定优于基线。[年度评分](forecast_scores.csv) · [训练与收缩](train_folds.csv)

## 2. 风险预警实际用途

高风险真值为未来五日实现方差超过该篮子开发期的 90% 分位。主预警把各模型开发期提示频率校准到约 10%；未来提示频率允许漂移。下表显示本轮回放中已经明显偏离开发期预算，不能称为未来同等告警数量。

{atable}
完整模型减少预测低估，并不自动意味着能减少预警遗漏。逐年与同风险门槛路线见 alert_scores.csv；配对区间见 alert_tests.csv。这是监测原型的离线评价，不是已有业务采用或价值证明。

## 3. 投资使用的收益代价

下表为仅开发期匹配实际波动后，回放单边 5 bp 的结果。10/25 bp 和全部简单对照在 portfolio_metrics.csv。保持国债目标、零现金收益、无杠杆，未来实际风险不保证相同。

{etable}
验收表共 {len(gates_frame)} 项条件，未通过 {int((~gates_frame.passed).sum())} 项；逐项见 [接受门槛](acceptance_gates.csv)。预测、预警与投资替换的门槛分别判断，不能挑某个改善数字作为整体通过。两篮子、两档在 5/10 bp 下的八项经济联合门槛全部未通过；不能把这些条件计数当作模型质量得分。

## 4. 数据、复现与接受边界

新增六只股票 ETF 的日期、质量和指纹见 data_quality.csv；原黄金和国债继续使用冻结数据。159929/159930 的腾讯后复权请求返回 day 字段，逐段与前复权及不复权请求核对完全一致；供应商口径单列，未独立审计全部公司行动。159930 有四个形成期缺失日期，保留空值，不冒充已确认停牌。新增资产并非按收益排名选择，但属于有目的选择的存续 ETF，不能推广到全部市场或忽略存续选择。

运行 `python -B -X utf8 chronoetf/transfer_research.py` 生成新目录并自动执行本轮核验；`--audit <新生成目录>` 可单独核验；`--summarize <目录>` 生成本候选说明及主图。源码复用原预测、控仓和账户逻辑。原始行情及大型逐日账户留在本地，发布共同周度预测、训练/收缩、关键指标与区间。公开精简目录没有完整账户，因此重新审计须先运行入口生成新目录，并核对 manifest 的输入指纹；不能直接对刚下载的精简目录运行完整账户审计。

本轮完成四项工作：明确风险用途、固定规则迁移、补充简单记忆对照、分开验证预警与经济效果。跨篮子稳定增量、完整机制必要性和低风险收益提高仍按门槛判断；真正前瞻样本尚为 0。用户接受后才更新两个正式 MD。
"""
    (folder / "CANDIDATE_RESULT.md").write_text(body, encoding="utf8")
    print(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.audit:
        audit(args.audit)
    elif args.summarize:
        summarize(args.summarize)
    elif args.download_only:
        print(transfer_data()[1].to_string(index=False))
    else:
        folder = ROOT / "results" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
        run(folder)
        audit(folder)
        summarize(folder)
