"""ARIMAX-LSTM предиктор: ARIMA(4,1,5) + per-horizon LSTM на residuals.

Логика инференса:
  1. Загружаем сохранённые артефакты (arima_result.pkl, lstm_h*.keras, scaler_h*.pkl).
  2. Walk-forward по последним (lookback + max_h) точкам: обновляем состояние
     SARIMAX фактами (state.append), попутно собираем residuals для LSTM.
     Это НЕ дообучение — параметры ARIMA не меняются, обновляется только
     фильтр Калмана, чтобы прогноз шёл от текущей даты, а не от конца train.
  3. Финальный прогноз — от обновлённого state.
  4. LSTM-коррекция по последним lookback residuals.
  5. Интервалы — эмпирические квантили |residual_h| из val_residuals_by_horizon.csv.

Важно: target_1 = close(t+1), поэтому ARIMA прогнозирует ЦЕНУ, а не diff.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from app.services.aci import AdaptiveConformal
from app.core.config import settings


logger = logging.getLogger(__name__)


class ARIMAXPredictor:
    """Инференс ARIMA(4,1,5) + per-horizon LSTM."""

    def __init__(self, model_dir: Path | None = None) -> None:
        base = Path(model_dir or settings.arimax_model_dir)
        cfg_path = base / settings.arimax_config_file
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        winsor_path = Path(settings.arimax_model_dir) / "winsor_params.json"
        self.winsor_params = (
            json.loads(winsor_path.read_text(encoding="utf-8"))
            if winsor_path.exists() else None
        )
        self.horizons: list[int] = list(cfg["horizons"])
        self.max_h: int = max(self.horizons)
        self.lookback: int = int(cfg.get("lookback_lstm", 30))

        # ARIMA: загружаем сохранённые параметры (fit не нужен)
        arima_path = base / settings.arimax_state_file
        logger.info("ARIMAX: loading ARIMA from %s", arima_path)
        self.arima_res = SARIMAXResults.load(str(arima_path))

        # LSTM + scaler по горизонтам
        models_dir = base / "models_per_h"
        self.lstm_models: dict[int, object] = {}
        self.scalers: dict[int, object] = {}
        for h in self.horizons:
            lstm_path = models_dir / f"lstm_h{h}.keras"
            scaler_path = models_dir / f"scaler_h{h}.pkl"
            logger.info("ARIMAX: loading LSTM h=%s from %s", h, lstm_path)
            self.lstm_models[h] = load_model(str(lstm_path), compile=False)
            self.scalers[h] = joblib.load(scaler_path)

        # feature_cols — те же 51, что у Attention-GRU
        gru_cfg_path = Path(settings.gru_model_dir) / settings.gru_config_file
        with open(gru_cfg_path, encoding="utf-8") as f:
            gru_cfg = json.load(f)
        self.feature_cols: list[str] = list(gru_cfg["feature_cols"])

        # ── ACI ──
        self.aci_path = base / "aci_state.json"
        if not self.aci_path.exists():
            raise RuntimeError(
                f"ARIMAX: {self.aci_path} не найден. "
                f"Модель не откалибрована — прогони weekly."
            )
        self.aci = AdaptiveConformal.load(self.aci_path)
        logger.info("ARIMAX: ACI loaded from %s", self.aci_path)

    

    # ------------------------------------------------------------------
    def _walk_forward_state(
        self, features_df: pd.DataFrame
    ) -> tuple[object, dict[int, list[float]], float, np.ndarray]:
        """Докатывает state ARIMA по последним (lookback + max_h) точкам.

        Возвращает:
          state        — обновлённый SARIMAXResults
          residuals_h  — residuals walk-forward по каждому h
          anchor       — close на текущей (последней) точке
          X_now        — (51,) признаки на текущей точке, для финального forecast
        """
        win = self.lookback + self.max_h
        if len(features_df) < win:
            raise ValueError(
                f"ARIMAX: нужно ≥{win} строк features_df, получено {len(features_df)}"
            )

        tail = features_df.iloc[-win:].reset_index(drop=True)
        X_win = tail[self.feature_cols].to_numpy(dtype=np.float64)        # (win, 51)
        close_win = tail["close"].to_numpy(dtype=np.float64)              # (win,)

        # y[t] = close(t+1), длина win-1
        y_win = close_win[1:]                                             # (win-1,)
        n_y = len(y_win)

        anchor = float(close_win[-1])
        X_now = X_win[-1]

        state = self.arima_res
        residuals_h: dict[int, list[float]] = {h: [] for h in self.horizons}

        for t in range(n_y):
            x_t = X_win[t : t + 1]
            y_t = y_win[t]
            X_future = np.tile(x_t, (self.max_h, 1))
            fc = np.asarray(
                state.forecast(steps=self.max_h, exog=X_future)
            ).reshape(-1)
            for h in self.horizons:
                idx = t + h - 1
                if idx < n_y:                     # ← было: idx < win (ошибка)
                    residuals_h[h].append(float(y_win[idx] - fc[h - 1]))
            state = state.append(endog=[y_t], exog=x_t, refit=False)

        return state, residuals_h, anchor, X_now

    # ------------------------------------------------------------------
    def predict(self, features_df: pd.DataFrame, horizon: int) -> dict:
        """Прогноз цены BTC на горизонт h (h ∈ {1,3,6,12,18})."""
        if horizon not in self.horizons:
            raise ValueError(
                f"ARIMAX: horizon={horizon} не поддерживается. Доступны: {self.horizons}"
            )

        state, residuals_h, anchor, X_now = self._walk_forward_state(features_df)

        # ARIMA-прогноз на horizon от обновлённого state
        X_future = np.tile(X_now.reshape(1, -1), (self.max_h, 1))
        fc_now = np.asarray(
            state.forecast(steps=self.max_h, exog=X_future)
        ).reshape(-1)
        arima_pred = float(fc_now[horizon - 1])

        # LSTM-коррекция
        hist = residuals_h[horizon]
        if len(hist) >= self.lookback:
            seq = np.asarray(hist[-self.lookback:], dtype=np.float64).reshape(-1, 1)
            seq_scaled = self.scalers[horizon].transform(seq).reshape(
                1, self.lookback, 1
            )
            corr_scaled = self.lstm_models[horizon].predict(seq_scaled, verbose=0)
            corr = float(
                self.scalers[horizon].inverse_transform(corr_scaled)[0, 0]
            )
        else:
            logger.warning(
                "ARIMAX h=%s: residuals=%s < lookback=%s, corr=0",
                horizon, len(hist), self.lookback,
            )
            corr = 0.0

        point = arima_pred + corr
        q = self.aci.get_quantile(horizon)

        return {
            "model": "arimax_lstm",
            "horizon": horizon,
            "hours": horizon * 4,
            "point": float(point),
            "lower": float(point - q),
            "upper": float(point + q),
            "anchor": float(anchor),
            "q_used": float(q),                              # ← NEW
            "alpha_t": float(self.aci.alpha_t[horizon]),
            "is_stale": False,
        }
        
    def update_aci(
        self, horizon: int, y_true: float, y_pred: float, q_used: float
    ) -> float:
        """Обновить ACI по наступившему факту (вызывается scheduler'ом).

        Args:
            horizon: 1, 3, 6, 12, 18.
            y_true: фактический close(target_time).
            y_pred: point из сохранённого прогноза.
            q_used: q, использованный при построении интервала.

        Returns:
            miss (0.0 — попадание, 1.0 — промах).
        """
        miss = self.aci.update(horizon, y_true, y_pred, q_used)
        try:
            self.aci.save(self.aci_path)
        except Exception:
            logger.exception("ARIMAX: не удалось сохранить ACI")
        logger.info(
            "ARIMAX: ACI update h=%s miss=%s alpha_t=%.4f (n=%s)",
            horizon, miss, self.aci.alpha_t[horizon], self.aci.n_updates[horizon],
        )
        return miss


# Глобальный экземпляр — загрузка артефактов при импорте
arimax_predictor = ARIMAXPredictor()
