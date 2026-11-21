"""Trainer package: обучение, оценка, регистрация и переобучение моделей.

Основные точки входа:
  - weekly:     python -m app.services.trainer.weekly --model both
                (еженедельное переобучение deploy + eval, promote/keep, ACI, backtest)

Библиотечные модули (используются внутри weekly и в Web):
  - attention_gru:     train_attention_gru (GRU, MIMO, diff-target)
  - arimax_lstm:       train_arimax_lstm  (SARIMAX + LSTM, price-target)
  - evaluator:         evaluate (метрики + predictions DataFrame)
  - splits:            split_by_bars (нарезка на train/val/test без утечек)
  - registry:          save_version, promote, save_to_db, get_champion, get_history
  - aci_calibration:   calibrate_aci (Adaptive Conformal Inference)
  - backtest:          backtest_from_predictions, backtest_for_model
  - plots:             make_all_plots (графики для отчёта, CLI: python -m app.services.trainer.plots --model ...)
  - base:              HORIZONS, LOOKBACK, MAX_H, dataset_hash
"""

from app.services.trainer.base import HORIZONS, LOOKBACK, MAX_H, dataset_hash

__all__ = [
    "HORIZONS",
    "LOOKBACK",
    "MAX_H",
    "dataset_hash",
]