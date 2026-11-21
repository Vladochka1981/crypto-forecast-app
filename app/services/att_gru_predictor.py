"""
Предиктор на основе Attention-GRU.

Загружает:
- attention_gru.keras — модель (MIMO, 5 выходов).
- scaler_X.pkl — StandardScaler для признаков.
- scaler_y.pkl — StandardScaler для целевой переменной (diff).
- conformal_quantiles.json — квантили для 95% интервалов.
- model_config.json — список feature_cols, lookback, horizons.

Метод predict(features_df, horizon, model) возвращает:
{
    "model": "attention_gru",
    "horizon": int,
    "hours": int,
    "point": float,        # точечный прогноз цены
    "lower": float,        # нижняя граница 95% CI
    "upper": float,        # верхняя граница 95% CI
    "anchor": float,       # close(t) — текущая цена
    "is_stale": bool,      # данные из кэша?
}
"""
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from app.services.attention_layer import ArticleStyleTemporalAttention # noqa: F401
from app.services.aci import AdaptiveConformal
from keras.models import load_model 

from app.core.config import settings

logger = logging.getLogger(__name__)


class AttentionGRUPredictor:
    """Инференс Attention-GRU с conformal-интервалами."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir or settings.gru_model_dir)

        # --- конфиг модели ---
        config_path = self.model_dir / settings.gru_config_file
        with config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.feature_cols: list[str] = self.config["feature_cols"]
        self.lookback: int = self.config.get("lookback", 30)
        self.horizons: list[int] = self.config.get("horizons", [1, 3, 6, 12, 18])

        # --- scaler'ы ---
        self.scaler_x = joblib.load(self.model_dir / settings.gru_scaler_x_file)
        self.scaler_y = joblib.load(self.model_dir / settings.gru_scaler_y_file)

        # ── ACI ──
        self.aci_path = self.model_dir / "aci_state.json"
        if not self.aci_path.exists():
            raise RuntimeError(
                f"Attention-GRU: {self.aci_path} не найден. "
                f"Модель не откалибрована — прогони weekly."
            )
        self.aci = AdaptiveConformal.load(self.aci_path)
        logger.info("Attention-GRU: ACI loaded from %s", self.aci_path)

        # --- winsor_params (для feature_engineer.build_features) ---
        winsor_path = self.model_dir / "winsor_params.json"
        if winsor_path.exists():
            with winsor_path.open("r", encoding="utf-8") as f:
                self.winsor_params = json.load(f)
        else:
            self.winsor_params = None
            logger.warning("winsor_params.json не найден в %s", self.model_dir)

        # --- модель ---
        model_path = self.model_dir / settings.gru_model_file
        self.model = load_model(model_path, compile=False)
        logger.info(f"Attention-GRU загружена: {model_path}")

    # ---------- инференс ----------

    def _prepare_input(self, features_df: pd.DataFrame) -> tuple[np.ndarray, float]:
        """
        Взять последние lookback строк, стандартизировать, вернуть
        (X shape (1, lookback, n_features), anchor_close).
        """
        if len(features_df) < self.lookback:
            raise RuntimeError(
                f"Нужно минимум {self.lookback} строк, а есть {len(features_df)}."
            )

        # строго по feature_cols
        missing = [c for c in self.feature_cols if c not in features_df.columns]
        if missing:
            raise ValueError(f"В features_df нет колонок: {missing}")

        tail = features_df[self.feature_cols].tail(self.lookback).copy()
        if tail.isna().any().any():
            nan_cols = tail.columns[tail.isna().any()].tolist()
            raise ValueError(f"NaN в последних {self.lookback} строках: {nan_cols}")

        X_raw = tail.values.astype(np.float32)
        X_scaled = self.scaler_x.transform(X_raw).astype(np.float32)
        X = X_scaled.reshape(1, self.lookback, len(self.feature_cols))

        anchor_close = float(features_df["close"].iloc[-1])
        return X, anchor_close

    def predict(
        self,
        features_df: pd.DataFrame,
        horizon: int,
        model_name: str = "attention_gru",
            ) -> dict:
        """
        Прогноз для заданного горизонта.

        Параметры:
            features_df: DataFrame с 51 признаком (из FeatureEngineer).
            horizon: 1, 3, 6, 12 или 18.
            model_name: "attention_gru" (для совместимости с Orchestrator).

        Возвращает:
            dict с полями: model, horizon, hours, point, lower, upper, anchor.
        """
        if horizon not in self.horizons:
            raise ValueError(
                f"Горизонт {horizon} не поддерживается. "
                f"Допустимые: {self.horizons}"
            )

        X, anchor_close = self._prepare_input(features_df)

        # MIMO: один forward pass → (1, 5) в стандартизированном пространстве
        pred_scaled = self.model.predict(X, verbose=0)[0]   # shape (5,)

        # обратно в доллары (diff = close(t+h) - close(t))
        pred_diff = self.scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]

        # индекс горизонта в выходе модели
        pos = self.horizons.index(horizon)
        point_diff = float(pred_diff[pos])
        point_price = anchor_close + point_diff

        # ACI: q — адаптивный квантиль (буфер ошибок гибрида на val + онлайн)
        q = self.aci.get_quantile(horizon)
        lower = anchor_close + (point_diff - q)
        upper = anchor_close + (point_diff + q)
        alpha_t = float(self.aci.alpha_t[horizon])

        return {
            "model": "attention_gru",
            "horizon": int(horizon),
            "hours": int(horizon) * 4,
            "point": float(point_price),
            "lower": float(lower),
            "upper": float(upper),
            "anchor": float(anchor_close),
            "q_used": float(q),              # ← ACI
            "alpha_t": alpha_t,  
            "is_stale": False,   # Orchestrator может перезаписать
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
            logger.exception("Attention-GRU: не удалось сохранить ACI")
        logger.info(
            "Attention-GRU: ACI update h=%s miss=%s alpha_t=%.4f (n=%s)",
            horizon, miss, self.aci.alpha_t[horizon], self.aci.n_updates[horizon],
        )
        return miss

# Глобальный экземпляр
att_gru_predictor = AttentionGRUPredictor()