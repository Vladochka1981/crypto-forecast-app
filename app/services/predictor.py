"""ModelPredictor — фасад над двумя предикторами (GRU + ARIMAX).

Снаружи — один класс ModelPredictor с методом predict(horizon, model, version).
Внутри — два независимых предиктора (AttentionGRUPredictor, ARIMAXPredictor),
которые выбираются по имени модели. При передаче version создаётся
временный предиктор для нужной версии из archive/.
"""
import logging
from pathlib import Path

from app.services.att_gru_predictor import (
    AttentionGRUPredictor, att_gru_predictor,
)
from app.services.arimax_predictor import (
    ARIMAXPredictor, arimax_predictor,
)
from app.services.feature_engineer import feature_engineer

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Единый фасад для инференса Attention-GRU и ARIMAX-LSTM."""

    def __init__(self) -> None:
        self._predictors = {
            "attention_gru": att_gru_predictor,
            "arimax_lstm": arimax_predictor,
        }
        self._version_cache = {}

    def get_predictor(self, model: str, version: str | None = None):
        """Возвращает предиктор для конкретной версии (или дефолтный)."""
        if not version:
            return self._predictors[model]

        key = f"{model}::{version}"
        if key not in self._version_cache:
            from app.core.config import settings
            base = Path(settings.models_dir) / model
            model_dir = base / "archive" / f"{model}_{version}"
            if not model_dir.exists():
                raise ValueError(f"Версия не найдена: {model_dir}")
            
            if model == "attention_gru":
                self._version_cache[key] = AttentionGRUPredictor(model_dir)
            else:
                self._version_cache[key] = ARIMAXPredictor(model_dir)
            logger.info("Загружен предиктор %s", key)
        return self._version_cache[key]

    def predict(self, horizon: int, model: str = "attention_gru", version: str | None = None) -> dict:
        """Инференс выбранной модели на заданный горизонт."""
        if model not in self._predictors:
            raise ValueError(f"Неизвестная модель: {model}")

        predictor = self.get_predictor(model, version)

        if model == "arimax_lstm":
            n_keep = predictor.lookback + predictor.max_h
            features = feature_engineer.build_features(
                limit=300, winsor_params=predictor.winsor_params,
                n_keep=n_keep,
            )
        else:
            features = feature_engineer.build_features(
                limit=300, winsor_params=predictor.winsor_params,
            )

        if model == "arimax_lstm":
            return predictor.predict(features, horizon)
        return predictor.predict(features, horizon, model)

    def predict_all_horizons(self, model: str = "attention_gru", version: str | None = None) -> list[dict]:
        """Инференс на все 5 горизонтов (1, 3, 6, 12, 18) для одной модели."""
        if model not in self._predictors:
            raise ValueError(f"Неизвестная модель: {model}")

        predictor = self.get_predictor(model, version)

        if model == "arimax_lstm":
            n_keep = predictor.lookback + predictor.max_h
            features = feature_engineer.build_features(
                limit=300, winsor_params=predictor.winsor_params,
                n_keep=n_keep,
            )
        else:
            features = feature_engineer.build_features(
                limit=300, winsor_params=predictor.winsor_params,
            )

        out = []
        for h in [1, 3, 6, 12, 18]:
            if model == "arimax_lstm":
                out.append(predictor.predict(features, h))
            else:
                out.append(predictor.predict(features, h, model))
        return out


# Глобальный фасад
model_predictor = ModelPredictor()