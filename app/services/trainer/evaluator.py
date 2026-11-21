"""Единый evaluator для Attention-GRU и ARIMAX-LSTM.

Принимает model_dir + test_df + n_test (опционально). Не знает про БД,
metadata.json, папки версий. Используется:
  - weekly.py: challenger vs champion на одном rolling-окне (n_test=720)
  - viz_eval.py: визуализация (n_test=720)
  - audit-скрипты (n_test=None — все точки)

Честное выравнивание: metrics только на последних n_test точках test_df,
минус MAX_H (где target_h известен) → n_test - MAX_H (702 при n_test=720).
Для GRU и ARIMAX — одинаковый n=702 (без train-контекста).

GRU: target = make_diff_targets(close) (НЕ из CSV), point = anchor + diff.
ARIMAX: target = close(t+h) (цена), point = arima_pred + corr.

ARIMAX-часть 1:1 с app/services/arimax_predictor.py:
  - окно lookback + max_h, y_win = close_win[1:]
  - state.append после каждого t
  - residual_h(t') доступен в residual_history в момент t' + h - 1
  - corr из LSTM_h по последним lookback residuals
  - arima_only = fc[h-1] без corr
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from app.services.trainer.base import (
    HORIZONS, LOOKBACK, MAX_H, create_mimo_sequences, make_diff_targets,
)

log = logging.getLogger(__name__)
ModelName = Literal["attention_gru", "arimax_lstm"]


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def calc_metrics(y_true, y_pred, anchor) -> dict[str, float]:
    """MAE, RMSE, R², MAPE, IA, Skill, MDA±, DA, std_ratio."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    anchor = np.asarray(anchor, dtype=np.float64).ravel()

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(anchor)
    if not mask.any():
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "mape": float("nan"), "ia": float("nan"), "skill": float("nan"),
                "mda_plus": float("nan"), "mda_minus": float("nan"),
                "da": float("nan"), "std_ratio": float("nan"), "n": 0}
    y_true, y_pred, anchor = y_true[mask], y_pred[mask], anchor[mask]

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)
    rmse_naive = float(np.sqrt(np.mean((y_true - anchor) ** 2)))
    skill = float(1 - rmse / rmse_naive) if rmse_naive > 0 else float("nan")
    y_m = float(np.mean(y_true))
    num = float(np.sum((y_true - y_pred) ** 2))
    den = float(np.sum((np.abs(y_true - y_m) + np.abs(y_pred - y_m)) ** 2))
    ia = float(1 - num / den) if den > 0 else float("nan")
    a_dir = np.sign(y_true - anchor)
    p_dir = np.sign(y_pred - anchor)
    m_up, m_dn, m_nz = a_dir > 0, a_dir < 0, a_dir != 0
    mda_p = float(np.mean(p_dir[m_up] == a_dir[m_up]) * 100) if m_up.any() else float("nan")
    mda_m = float(np.mean(p_dir[m_dn] == a_dir[m_dn]) * 100) if m_dn.any() else float("nan")
    da = float(np.mean(p_dir[m_nz] == a_dir[m_nz]) * 100) if m_nz.any() else float("nan")
    std_p = float(np.std(y_pred))
    std_a = float(np.std(y_true))
    return {
        "mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "ia": ia,
        "skill": skill, "mda_plus": mda_p, "mda_minus": mda_m, "da": da,
        "std_ratio": std_p / std_a if std_a > 0 else float("nan"),
        "n": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Attention-GRU
# ---------------------------------------------------------------------------

def _eval_attention_gru(
    model_dir: Path, test_df: pd.DataFrame,
    feature_cols: list[str], n_test: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    from keras.models import load_model
    import app.services.attention_layer  # noqa: F401

    cfg = json.loads((model_dir / "model_config.json").read_text())
    lookback = int(cfg.get("lookback", LOOKBACK))
    horizons = list(cfg.get("horizons", HORIZONS))

    df = test_df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    X = df[feature_cols].to_numpy(np.float32)
    close = df["close"].to_numpy(np.float64)
    dates = df["date"].to_numpy()

    t5 = make_diff_targets(close, MAX_H)[:, [h - 1 for h in horizons]]

    scaler_X = joblib.load(model_dir / "scaler_X.pkl")
    scaler_y = joblib.load(model_dir / "scaler_y.pkl")
    X_s = scaler_X.transform(X).astype(np.float32)
    t_s = scaler_y.transform(t5).astype(np.float32)
    X_seq, y_seq = create_mimo_sequences(X_s, t_s, lookback)

    # anchor_idx: если n_test задан — последние n_test точек, минус MAX_H.
    # Иначе — все (lookback-1 .. len - MAX_H).
    if n_test is not None:
        anchor_idx = np.arange(len(df) - n_test, len(df) - MAX_H)
    else:
        anchor_idx = np.arange(lookback - 1, len(df) - MAX_H)
    anchor = close[anchor_idx]
    anchor_dates = dates[anchor_idx]

    # X_seq[i] соответствует точке i + lookback - 1 в df.
    seq_start = anchor_idx[0] - (lookback - 1)
    n_keep = len(anchor_idx)
    if seq_start < 0:
        raise ValueError(f"n_test={n_test} слишком велик: seq_start={seq_start} < 0")
    X_seq = X_seq[seq_start:seq_start + n_keep]
    y_seq = y_seq[seq_start:seq_start + n_keep]

    model = load_model(model_dir / "attention_gru.keras", compile=False)
    pred_s = model.predict(X_seq, verbose=0, batch_size=256)
    pred_price = anchor[:, None] + scaler_y.inverse_transform(pred_s)
    y_price = anchor[:, None] + scaler_y.inverse_transform(y_seq)

    metrics: dict[str, dict] = {}
    rows: list[dict] = []
    for pos, h in enumerate(horizons):
        m = calc_metrics(y_price[:, pos], pred_price[:, pos], anchor)
        metrics[f"h{h}"] = m
        for i in range(len(anchor)):
            actual = float(y_price[i, pos]) if np.isfinite(y_price[i, pos]) else float("nan")
            rows.append({
                "horizon": h,
                "date": pd.Timestamp(anchor_dates[i]).isoformat(),
                "anchor": float(anchor[i]),
                "actual": actual,
                "pred": float(pred_price[i, pos]),
            })
    return metrics, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ARIMAX-LSTM
# ---------------------------------------------------------------------------

def _eval_arimax_lstm(
    model_dir: Path, test_df: pd.DataFrame,
    feature_cols: list[str], arima_only: bool = False,
    n_test: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    from keras.models import load_model
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults

    cfg = json.loads((model_dir / "config.json").read_text())
    lookback = int(cfg.get("lookback_lstm", LOOKBACK))
    max_h = int(cfg.get("max_h", MAX_H))
    horizons = list(cfg.get("horizons", HORIZONS))
    exog_cols = list(cfg.get("exog_cols", feature_cols))

    df = test_df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    X = df[exog_cols].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    dates = df["date"].to_numpy()

    arima_res = SARIMAXResults.load(str(model_dir / "arima_result.pkl"))

    # LSTM + scaler
    lstm_models: dict[int, Any] = {}
    lstm_scalers: dict[int, Any] = {}
    for h in horizons:
        lstm_models[h] = load_model(
            str(model_dir / f"models_per_h/lstm_h{h}.keras"), compile=False,
        )
        lstm_scalers[h] = joblib.load(model_dir / f"models_per_h/scaler_h{h}.pkl")

    # residual_history стартует с val residuals
    val_res = pd.read_csv(model_dir / "val_residuals_by_horizon.csv")
    residual_history: dict[int, list[float]] = {}
    for h in horizons:
        col = f"residual_h{h}"
        vals = (val_res[col].dropna().to_numpy(np.float64)
                if col in val_res.columns else np.array([]))
        residual_history[h] = list(vals[-lookback:]) if len(vals) else []

    win = lookback + max_h
    if len(df) < win:
        raise ValueError(f"ARIMAX: нужно ≥{win} строк, получено {len(df)}")

    # n_iter: если n_test задан — n_test (только test, без train),
    # иначе — len(df) - win (все точки после прогрева).
    if n_test is not None:
        n_iter = n_test
        # anchor_idx_all: последние n_test точек, минус MAX_H (где target_h известен)
        anchor_idx_all = np.arange(len(df) - n_test, len(df) - MAX_H)
        n_iter = len(anchor_idx_all)   # n_test - MAX_H = 702
        # прогрев state на точках до anchor_idx_all[0]
        state = arima_res
        for i in range(anchor_idx_all[0]):
            state = state.append(
                endog=np.array([close[i]]),
                exog=X[i:i + 1],
                refit=False,
            )
    else:
        n_iter = len(df) - win
        anchor_idx_all = np.arange(win - max_h, win - max_h + n_iter)
        state = arima_res
        for i in range(win - max_h):
            state = state.append(
                endog=np.array([close[i]]),
                exog=X[i:i + 1],
                refit=False,
            )

    anchors = close[anchor_idx_all]
    anchor_dates = dates[anchor_idx_all]
    preds_hybrid = np.full((n_iter, len(horizons)), np.nan, dtype=np.float64)
    preds_arima = np.full_like(preds_hybrid, np.nan)
    residual_buffer: dict[int, list[float]] = {h: [] for h in horizons}

    for t in range(n_iter):
        i_now = anchor_idx_all[t]
        x_now = X[i_now:i_now + 1]

        # residual_h(t') доступен при t = t' + h - 1
        for h in horizons:
            idx_avail = t - h + 1
            if 0 <= idx_avail < len(residual_buffer[h]):
                residual_history[h].append(residual_buffer[h][idx_avail])

        x_future = np.tile(x_now, (max_h, 1))
        try:
            fc = np.asarray(state.forecast(steps=max_h, exog=x_future)).reshape(-1)
        except Exception as e:
            log.warning("forecast t=%s: %s", t, e)
            break

        for pos, h in enumerate(horizons):
            arima_pred = float(fc[h - 1])
            if arima_only:
                preds_hybrid[t, pos] = arima_pred
            else:
                hist = residual_history[h]
                if len(hist) >= lookback:
                    seq = np.asarray(hist[-lookback:], dtype=np.float64).reshape(-1, 1)
                    seq_s = lstm_scalers[h].transform(seq).reshape(1, lookback, 1)
                    corr_s = lstm_models[h].predict(seq_s, verbose=0)
                    corr = float(lstm_scalers[h].inverse_transform(corr_s)[0, 0])
                else:
                    corr = 0.0
                preds_hybrid[t, pos] = arima_pred + corr
            preds_arima[t, pos] = arima_pred

        # residual_h(t) для будущих шагов
        for pos, h in enumerate(horizons):
            target_idx = i_now + h
            if target_idx < len(close):
                residual_buffer[h].append(
                    float(close[target_idx] - fc[h - 1])
                )

        # state.append факт y_t = close(t+1)
        if i_now + 1 < len(close):
            state = state.append(
                endog=np.array([close[i_now + 1]]),
                exog=x_now,
                refit=False,
            )

    # y_true = close(anchor_idx_all + h) — NaN, если за пределами df
    y_true = np.full((n_iter, len(horizons)), np.nan, dtype=np.float64)
    for pos, h in enumerate(horizons):
        for t in range(n_iter):
            j = anchor_idx_all[t] + h
            if j < len(close):
                y_true[t, pos] = close[j]

    preds = preds_hybrid if not arima_only else preds_arima

    metrics: dict[str, dict] = {}
    rows: list[dict] = []
    for pos, h in enumerate(horizons):
        m = calc_metrics(y_true[:, pos], preds[:, pos], anchors)
        metrics[f"h{h}"] = m
        for i in range(n_iter):
            rows.append({
                "horizon": h,
                "date": pd.Timestamp(anchor_dates[i]).isoformat(),
                "anchor": float(anchors[i]),
                "actual": float(y_true[i, pos]) if np.isfinite(y_true[i, pos]) else float("nan"),
                "pred": float(preds_hybrid[i, pos]) if np.isfinite(preds_hybrid[i, pos]) else float("nan"),
                "arima_only": float(preds_arima[i, pos]) if np.isfinite(preds_arima[i, pos]) else float("nan"),
            })
    return metrics, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def evaluate(
    model_name: ModelName, model_dir: str | Path,
    test_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_test: int | None = None,
) -> dict[str, Any]:
    """Оценивает модель на test_df.

    Args:
        model_name: "attention_gru" | "arimax_lstm".
        model_dir: папка с артефактами (deploy/, eval/, archive/vX/).
        test_df: DataFrame (date, close, feature_cols).
                 Должен содержать контекст (lookback + max_h) до начала test.
        feature_cols: если None — из model_config.json/config.json.
        n_test: если задан — metrics только на последних n_test точках test_df,
                минус MAX_H (где target_h известен) → n_test - MAX_H.
                Если None — все точки (lookback-1 .. len - MAX_H).
                Для выравнивания GRU/ARIMAX: n_test=720 → n=702 для обеих.

    Returns:
        {model, test_range, n, metrics, metrics_arima_only, predictions}
    """
    model_dir = Path(model_dir)
    if model_name == "attention_gru":
        metrics, preds = _eval_attention_gru(
            model_dir, test_df, feature_cols or [], n_test=n_test,
        )
        return {
            "model": model_name,
            "test_range": (
                [preds["date"].iloc[0], preds["date"].iloc[-1]]
                if len(preds) else None
            ),
            "n": int(metrics.get("h1", {}).get("n", 0)),
            "metrics": metrics,
            "metrics_arima_only": None,
            "predictions": preds,
        }
    elif model_name == "arimax_lstm":
        mh, ph = _eval_arimax_lstm(
            model_dir, test_df, feature_cols or [], arima_only=False, n_test=n_test,
        )
        ma, _ = _eval_arimax_lstm(
            model_dir, test_df, feature_cols or [], arima_only=True, n_test=n_test,
        )
        return {
            "model": model_name,
            "test_range": (
                [ph["date"].iloc[0], ph["date"].iloc[-1]]
                if len(ph) else None
            ),
            "n": int(mh.get("h1", {}).get("n", 0)),
            "metrics": mh,
            "metrics_arima_only": ma,
            "predictions": ph,
        }
    raise ValueError(f"unknown model: {model_name}")