"""五日方差预测、事件门控及按年向前训练。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


HORIZON = 5
VALIDATION_YEAR = 2021
TEST_YEAR = 2024
EWMA_GRID = (0.90, 0.94, 0.97)
RIDGE = 0.02
VAR_FLOOR = 1e-10
EVENT_QUANTILE = 0.90
SHORT_HALF_LIFE = 5
LONG_HALF_LIFE = 30

RISK_NAMES = {
    "rolling": "60 日方差",
    "ewma": "EWMA",
    "single": "单记忆",
    "fixed": "双记忆固定写入",
    "chrono": "ChronoETF",
    "calendar": "ChronoETF 自然日",
}


@dataclass
class RiskStudy:
    log_returns: pd.DataFrame
    realized: pd.DataFrame
    forecasts: dict
    shock_counts: pd.DataFrame
    short_gate: pd.DataFrame
    long_gate: pd.DataFrame
    ewma_lambda: float
    ewma_validation: dict
    folds: pd.DataFrame
    dynamic_states: np.ndarray

    def losses(self):
        return {key: qlike(self.realized, frame) for key, frame in self.forecasts.items()}


def qlike(actual, forecast):
    """用同量纲的日均方差计算 QLIKE。"""
    ratio = np.maximum(actual, VAR_FLOOR) / np.maximum(forecast, VAR_FLOOR)
    return ratio - np.log(ratio) - 1


def memory_states(candidate, short_gate, long_gate, elapsed):
    states = np.zeros((*candidate.shape, 2))
    previous = np.zeros((candidate.shape[1], 2))
    half_lives = np.array([SHORT_HALF_LIFE, LONG_HALF_LIFE])
    decay = np.exp2(-elapsed[:, None] / half_lives[None, :])
    for i, observation in enumerate(candidate):
        prior = previous * decay[i]
        gates = np.column_stack((short_gate[i], long_gate[i]))
        previous = prior + gates * (observation[:, None] - prior)
        states[i] = previous
    return states


def fit_readout(features, anchor, realized, train, prediction_range):
    """拟合带岭惩罚的对数方差读出，所有模型使用同一目标函数。"""
    valid = np.isfinite(features).all(axis=2) & np.isfinite(anchor)
    eligible = train[:, None] & valid & np.isfinite(realized)
    x = features[eligible]
    relative_y = np.maximum(realized[eligible], VAR_FLOOR) / anchor[eligible]

    def objective(beta):
        linear = np.einsum("ij,j->i", x, beta)
        ratio = relative_y * np.exp(-linear)
        loss = np.mean(ratio + linear - 1 - np.log(relative_y)) + RIDGE / 2 * (beta @ beta)
        grad = np.einsum("ij,i->j", x, 1 - ratio) / len(x) + RIDGE * beta
        return loss, grad

    fit = minimize(
        objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
        bounds=[(-1.5, 1.5)] * x.shape[1], options={"ftol": 1e-11},
    )
    if not fit.success:
        raise RuntimeError(f"方差读出优化未收敛：{fit.message}")
    x_predict = features[prediction_range]
    forecast = anchor[prediction_range] * np.exp(np.einsum("ijk,k->ij", x_predict, fit.x))
    return np.maximum(forecast, VAR_FLOOR), len(x)


def readout_inputs(short_variance, anchor, surprise, relative_volume):
    days, assets = anchor.shape
    indicators = np.broadcast_to(np.eye(assets), (days, assets, assets))
    data = np.stack((np.log(short_variance / anchor), surprise, relative_volume), axis=2)
    return np.concatenate((np.clip(data, -3, 3), indicators), axis=2)


def fit_risk(close, volume, fixed_lambda=None):
    dates = close.index
    positions = np.arange(len(dates))
    log_returns = np.log(close).diff()
    squared = log_returns.pow(2)
    realized = squared.rolling(HORIZON).mean().shift(-HORIZON)
    short_variance = squared.rolling(20, min_periods=18).mean().clip(lower=VAR_FLOOR)
    anchor = squared.rolling(60, min_periods=55).mean().clip(lower=VAR_FLOOR)
    relative_volume = np.log(volume / volume.rolling(20, min_periods=18).median().shift(1))
    surprise = np.log(squared / anchor + 0.05).clip(-3, 3)
    z_score = log_returns.abs() / np.sqrt(anchor.shift(1))

    ewma_candidates = {
        decay: squared.ewm(alpha=1 - decay, adjust=False, min_periods=20, ignore_na=True).mean()
        for decay in EWMA_GRID
    }
    validation = (dates.year >= VALIDATION_YEAR) & (dates.year < TEST_YEAR)
    test_start = dates.searchsorted(pd.Timestamp(TEST_YEAR, 1, 1))
    valid_labels = validation & (positions + HORIZON < test_start)
    ewma_validation = {
        decay: float(qlike(realized.loc[valid_labels], forecast.loc[valid_labels]).stack().mean())
        for decay, forecast in ewma_candidates.items()
    }
    ewma_lambda = min(ewma_validation, key=ewma_validation.get) if fixed_lambda is None else fixed_lambda
    forecasts = {"rolling": anchor, "ewma": ewma_candidates[ewma_lambda]}
    for name in ("single", "fixed", "chrono", "calendar"):
        forecasts[name] = pd.DataFrame(np.nan, index=dates, columns=close.columns)

    common = readout_inputs(short_variance, anchor, surprise, relative_volume)
    anchor_array = anchor.to_numpy()
    observed = np.isfinite(z_score.to_numpy()) & np.isfinite(surprise.to_numpy())
    candidate = np.nan_to_num(surprise.to_numpy(), nan=0.0)
    natural_days = np.diff(dates.to_numpy(dtype="datetime64[D]")).astype(int)
    elapsed_calendar = np.r_[1.0, natural_days * (5 / 7)]
    counts = np.zeros(close.shape, dtype=int)
    short_export = np.full(close.shape, np.nan)
    long_export = np.full(close.shape, np.nan)
    dynamic_export = np.full((*close.shape, 2), np.nan)
    folds = []

    for year in range(VALIDATION_YEAR, dates[-1].year + 1):
        start = dates.searchsorted(pd.Timestamp(year, 1, 1))
        end = dates.searchsorted(pd.Timestamp(year + 1, 1, 1))
        train = positions[:end] + HORIZON < start
        threshold = float(np.nanquantile(z_score.iloc[:end].to_numpy()[train], EVENT_QUANTILE))
        hits = z_score.iloc[:end].ge(threshold)
        recent_hits = hits.rolling(HORIZON, min_periods=1).sum().to_numpy(dtype=int)
        shock = hits.to_numpy()
        high_volume = relative_volume.iloc[:end].to_numpy() > 0
        confirmed = shock & high_volume
        persistent = shock & (recent_hits >= 2)
        short = (0.12 + 0.28 * shock + 0.10 * confirmed) * observed[:end]
        long = (0.035 + 0.06 * confirmed + 0.10 * persistent) * observed[:end]
        gate_train = train[:, None] & observed[:end]
        average_short = float(short[gate_train].mean())
        average_long = float(long[gate_train].mean())
        fixed = memory_states(
            candidate[:end], average_short * observed[:end], average_long * observed[:end], np.ones(end),
        )
        dynamic = memory_states(candidate[:end], short, long, np.ones(end))
        calendar = memory_states(candidate[:end], short, long, elapsed_calendar[:end])
        # 首个验证日从上一个交易日的收盘信号建立头寸。
        prediction_range = slice(start - int(year == VALIDATION_YEAR), end)
        for name, states in (
            ("single", fixed[:, :, :1]), ("fixed", fixed), ("chrono", dynamic), ("calendar", calendar),
        ):
            features = np.concatenate((common[:end], states), axis=2)
            prediction, count = fit_readout(
                features, anchor_array[:end], realized.to_numpy()[:end], train, prediction_range,
            )
            forecasts[name].iloc[prediction_range] = prediction
        counts[prediction_range] = recent_hits[prediction_range]
        short_export[prediction_range] = np.where(observed[:end], short, np.nan)[prediction_range]
        long_export[prediction_range] = np.where(observed[:end], long, np.nan)[prediction_range]
        dynamic_export[prediction_range] = dynamic[prediction_range]
        folds.append({
            "year": year, "train_signal_max": dates[start - HORIZON - 1].date().isoformat(),
            "train_label_max": dates[start - 1].date().isoformat(), "oos_first": dates[start].date().isoformat(),
            "oos_last": dates[end - 1].date().isoformat(), "training_asset_days": count,
            "shock_threshold": threshold, "fixed_short_gate": average_short, "fixed_long_gate": average_long,
        })

    return RiskStudy(
        log_returns, realized, forecasts,
        pd.DataFrame(counts, index=dates, columns=close.columns),
        pd.DataFrame(short_export, index=dates, columns=close.columns),
        pd.DataFrame(long_export, index=dates, columns=close.columns),
        ewma_lambda, ewma_validation, pd.DataFrame(folds), dynamic_export,
    )
