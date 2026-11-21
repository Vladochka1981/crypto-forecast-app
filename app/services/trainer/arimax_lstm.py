"""ARIMAX-LSTM: обучение SARIMAX(4,1,5) + per-horizon LSTM на residuals.

Перенос логики из train_arimax_from_csv.py (Colab-скрипт) в пакет trainer.
Убрано: CLI, --backup, rolling на test, metrics на test (это evaluator.py).
Добавлено: докатка ARIMA state до конца val (чтобы arima_result.pkl был
в состоянии «конец val», а не «конец walk-forward»), winsor_params в сохранении.

Схема:
  1. ARIMA(4,1,5) fit на train_df (target = close(t+1)).
  2. Walk-forward на val_df → residuals_h (для LSTM).
  3. LSTM_h на residuals_h (85/15 внутри val).
  4. Докатка state на последних MAX_H точках val (те, для которых
     не было target_h в walk-forward).
  5. Сохранение: arima_result.pkl (докатанный), lstm_h*.keras,
     scaler_h*.pkl, config.json (с exog_cols), val_residuals_by_horizon.csv,
     winsor_params.json (для Web).

Используется weekly.py:
    train_arimax_lstm(train_df, val_df, out_dir, feature_cols,
                      winsor_params=None)
    → возвращает dict с путями и метаданными для registry.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.services.trainer.base import HORIZONS, LOOKBACK, MAX_H

log = logging.getLogger(__name__)

# Гиперпараметры (соответствуют Colab-скрипту)
N_LSTM_UNITS = 64
DROPOUT = 0.4
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
EPOCHS = 150
PATIENCE_EARLY = 10
PATIENCE_REDUCE = 5
ARIMA_ORDER = (4, 1, 5)
SEASONAL_ORDER = (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# LSTM на residuals_h
# ---------------------------------------------------------------------------

def _create_sequences(data: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Режет одномерный ряд на последовательности (X, y) длиной lookback."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def _train_lstm_for_horizon(
    residuals_h: list[float], h: int
) -> tuple[Any, Any, float]:
    """Обучает LSTM_h на residuals_h. Возвращает (model, scaler, time_sec).

    Если residuals мало (< LOOKBACK + 100) — возвращает (None, None, 0.0),
    и в предикторе LSTM-коррекция не применяется (corr=0).
    """
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam

    arr = np.asarray(residuals_h, dtype=np.float64)
    if len(arr) < LOOKBACK + 100:
        log.warning("h=%s: мало residuals (%s), пропуск LSTM", h, len(arr))
        return None, None, 0.0

    scaler = StandardScaler()
    res_scaled = scaler.fit_transform(arr.reshape(-1, 1)).flatten()
    X_seq, y_seq = _create_sequences(res_scaled, LOOKBACK)
    split = int(len(X_seq) * 0.85)
    X_tr, y_tr = X_seq[:split], y_seq[:split]
    X_va, y_va = X_seq[split:], y_seq[split:]
    if len(X_tr) < 50 or len(X_va) < 10:
        log.warning("h=%s: мало последовательностей (train=%s, val=%s)",
                    h, len(X_tr), len(X_va))
        return None, None, 0.0

    X_tr = X_tr.reshape(-1, LOOKBACK, 1)
    X_va = X_va.reshape(-1, LOOKBACK, 1)

    model = Sequential([
        LSTM(N_LSTM_UNITS, input_shape=(LOOKBACK, 1), return_sequences=False),
        Dropout(DROPOUT),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse", metrics=["mae"],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_EARLY,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=PATIENCE_REDUCE, min_lr=1e-7, verbose=0),
    ]
    t0 = time.time()
    model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=0,
    )
    tt = time.time() - t0
    log.info("h=%s: LSTM train=%s, val=%s, time=%.1fs",
             h, len(X_tr), len(X_va), tt)
    return model, scaler, tt


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def train_arimax_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: str | Path,
    feature_cols: list[str],
    winsor_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Обучает ARIMAX-LSTM на train_df + val_df, сохраняет в out_dir.

    Args:
        train_df: train, содержит date, close, feature_cols.
                  target_h = close(t+h) строятся внутри (shift(-h)).
        val_df: val, содержит date, close, feature_cols.
                На нём walk-forward для residuals + LSTM.
        out_dir: папка для артефактов (arima_result.pkl, models_per_h/,
                 config.json, val_residuals_by_horizon.csv, timing.json,
                 winsor_params.json).
        feature_cols: 51 признак (тот же порядок, что у GRU).
        winsor_params: параметры winsorization (сохраняются в winsor_params.json
                       для Web-инференса).

    Returns:
        dict с путями и метаданными:
          {
            "out_dir": str,
            "arima_order": [4,1,5],
            "n_arima": int, "n_lstm": int,
            "exog_cols": list[str],
            "lstm_trained": {h: bool},
            "timing": {"arima_fit_sec", "lstm_train_sec", "total_sec"},
            "trained_at": iso,
          }
    """
    out_dir = Path(out_dir)
    models_dir = out_dir / "models_per_h"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # === 0. Подготовка: target_h = close(t+h) ===
    def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["y"] = df["close"].shift(-1)
        for h in HORIZONS:
            df[f"target_{h}"] = df["close"].shift(-h)
        return df

    train_full = _add_targets(train_df).dropna(
        subset=["y"] + [f"target_{h}" for h in HORIZONS]
    ).reset_index(drop=True)
    val_full = _add_targets(val_df).dropna(
        subset=["y"] + [f"target_{h}" for h in HORIZONS]
    ).reset_index(drop=True)

    if len(train_full) < 1000:
        raise ValueError(f"ARIMAX: мало train после dropna: {len(train_full)}")
    if len(val_full) < MAX_H + 100:
        raise ValueError(f"ARIMAX: мало val после dropna: {len(val_full)}")

    log.info(
        "ARIMAX train: %s (%s..%s), val: %s (%s..%s)",
        len(train_full), train_full["date"].min(), train_full["date"].max(),
        len(val_full), val_full["date"].min(), val_full["date"].max(),
    )

    # === 1. ARIMA(4,1,5) fit на train ===
    X_train = train_full[feature_cols].to_numpy(np.float64)
    y_train = train_full["y"].to_numpy(np.float64)
    log.info("ARIMA fit: X=%s, y=%s...", X_train.shape, y_train.shape)
    t0 = time.time()
    arima_model = SARIMAX(
        y_train, order=ARIMA_ORDER, seasonal_order=SEASONAL_ORDER,
        exog=X_train, enforce_stationarity=False, enforce_invertibility=False,
    )
    arima_res = arima_model.fit(disp=False)
    arima_fit_sec = time.time() - t0
    log.info("ARIMA fit done: AIC=%.2f, time=%.1fs",
             arima_res.aic, arima_fit_sec)

    # === 2. Walk-forward на val → residuals_h ===
    X_val = val_full[feature_cols].to_numpy(np.float64)
    y_val = val_full["y"].to_numpy(np.float64)
    targets_val = {h: val_full[f"target_{h}"].to_numpy(np.float64) for h in HORIZONS}
    n_val_iter = len(val_full) - MAX_H

    residuals_val_by_h: dict[int, list[float]] = {h: [] for h in HORIZONS}
    state = arima_res
    log.info("Walk-forward на val (%s итераций)...", n_val_iter)
    t0 = time.time()
    for t in range(n_val_iter):
        x_t = X_val[t:t + 1]
        y_t = y_val[t]
        x_future = np.tile(x_t, (MAX_H, 1))
        try:
            fc = np.asarray(state.forecast(steps=MAX_H, exog=x_future)).reshape(-1)
        except Exception as e:
            log.warning("forecast t=%s: %s", t, e)
            break
        for h in HORIZONS:
            idx = t + h - 1
            if idx < n_val_iter:
                residuals_val_by_h[h].append(
                    float(targets_val[h][t] - fc[h - 1])
                )
        state = state.append(endog=[y_t], exog=x_t, refit=False)
        if (t + 1) % 200 == 0:
            log.info("  val wf %s/%s (%.0fs)", t + 1, n_val_iter, time.time() - t0)
    log.info(
        "Walk-forward done: %.1fs, residuals: %s",
        time.time() - t0,
        {h: len(residuals_val_by_h[h]) for h in HORIZONS},
    )

    # === 3. LSTM_h на residuals ===
    log.info("Обучение LSTM по горизонтам...")
    lstm_models: dict[int, Any] = {}
    lstm_scalers: dict[int, Any] = {}
    lstm_times: dict[int, float] = {}
    for h in HORIZONS:
        m, s, tt = _train_lstm_for_horizon(residuals_val_by_h[h], h)
        lstm_models[h], lstm_scalers[h], lstm_times[h] = m, s, tt

    # === 4. ДОКАТКА state до конца val (последние MAX_H точек) ===
    # Walk-forward выше дошёл до t = n_val_iter - 1. Осталось докатить
    # точки [n_val_iter, len(val_full) - 1] (их MAX_H штук), чтобы
    # arima_result.pkl был в состоянии «конец val», а не «конец walk-forward».
    n_append = len(val_full) - n_val_iter   # = MAX_H
    log.info("Докатка ARIMA state до конца val: %s точек", n_append)
    for t in range(n_val_iter, len(val_full)):
        if t + 1 < len(val_full):
            y_t = float(val_full["close"].iloc[t + 1])
            state = state.append(endog=[y_t], exog=X_val[t:t + 1], refit=False)
        # последняя точка val: close(t+1) неизвестен → state остаётся
        # на close(t-1+1)=close(t). arimax_predictor докатит её при инференсе.

    # === 5. Сохранение ===
    state.save(str(out_dir / "arima_result.pkl"))
    log.info("saved: arima_result.pkl (state до конца val)")

    for h in HORIZONS:
        if lstm_models[h] is not None:
            lstm_models[h].save(str(models_dir / f"lstm_h{h}.keras"))
            joblib.dump(lstm_scalers[h], models_dir / f"scaler_h{h}.pkl")
    log.info("saved: models_per_h/lstm_h*.keras + scaler_h*.pkl")

    # config.json (с exog_cols — чтобы ARIMAXPredictor не зависел от GRU-конфига)
    cfg = {
        "model": "ARIMA(4,1,5) + per-horizon LSTM",
        "arima_order": list(ARIMA_ORDER),
        "seasonal_order": list(SEASONAL_ORDER),
        "horizons": HORIZONS,
        "max_h": MAX_H,
        "lookback_lstm": LOOKBACK,
        "lstm_units": N_LSTM_UNITS,
        "lstm_dropout": DROPOUT,
        "exog_assumption": "persistence (X(t) frozen for all future steps)",
        "anchor": "close(t)",
        "target": "close(t+h) (price, NOT diff)",
        "n_features": len(feature_cols),
        "exog_cols": list(feature_cols),
        "n_arima": int(len(train_full)),
        "n_lstm": int(len(val_full)),
        "arima_train_from": str(train_full["date"].min()),
        "arima_train_to": str(train_full["date"].max()),
        "lstm_train_from": str(val_full["date"].min()),
        "lstm_train_to": str(val_full["date"].max()),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "state_end": "end_of_val (докатан)",
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    log.info("saved: config.json (с exog_cols)")

    # val_residuals_by_horizon.csv (для conformal в ARIMAXPredictor)
    max_len = max(len(r) for r in residuals_val_by_h.values())
    val_res = pd.DataFrame({"date": val_full["date"].iloc[:max_len]})
    for h in HORIZONS:
        r = residuals_val_by_h[h]
        val_res[f"residual_h{h}"] = (
            pd.Series(r).reindex(range(len(val_res))).values
        )
    val_res.to_csv(out_dir / "val_residuals_by_horizon.csv", index=False)
    log.info("saved: val_residuals_by_horizon.csv")

    # timing.json
    total_sec = time.time() - t_start
    timing = {
        "arima_fit_sec": float(arima_fit_sec),
        "lstm_train_sec": {str(h): float(lstm_times[h]) for h in HORIZONS},
        "total_sec": float(total_sec),
        "n_arima": int(len(train_full)),
        "n_lstm": int(len(val_full)),
        "n_features": len(feature_cols),
    }
    with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    log.info("saved: timing.json (total=%.1fs)", total_sec)

    # winsor_params.json (для Web)
    if winsor_params is not None:
        with open(out_dir / "winsor_params.json", "w", encoding="utf-8") as f:
            json.dump(winsor_params, f, indent=2, ensure_ascii=False)
        log.info("saved: winsor_params.json")

    return {
        "out_dir": str(out_dir),
        "arima_order": list(ARIMA_ORDER),
        "n_arima": int(len(train_full)),
        "n_lstm": int(len(val_full)),
        "exog_cols": list(feature_cols),
        "lstm_trained": {h: lstm_models[h] is not None for h in HORIZONS},
        "timing": timing,
        "trained_at": cfg["trained_at"],
    }