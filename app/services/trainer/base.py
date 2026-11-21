"""Общие утилиты для Trainer: таргеты, последовательности, метаданные.

Ключевые принципы:
  - make_diff_targets: target = close(t+h) - close(t) (как в Colab).
  - dataset_hash: sha256 от train-DataFrame (для воспроизводимости).
  - save_metadata: версия, дата, hash, метрики, гиперпараметры.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HORIZONS = [1, 3, 6, 12, 18]
MAX_H = max(HORIZONS)
LOOKBACK = 30


def make_diff_targets(close: np.ndarray, max_h: int = MAX_H) -> np.ndarray:
    """Строит MIMO-таргеты: r[t, k] = close[t+k+1] - close[t].

    Args:
        close: массив цен закрытия (n,).
        max_h: максимальный горизонт (18).

    Returns:
        np.ndarray формы (n, max_h); NaN там, где target недоступен.
    """
    n = len(close)
    r = np.full((n, max_h), np.nan, dtype=np.float64)
    for k in range(max_h):
        h = k + 1
        if h < n:
            r[:n - h, k] = close[h:] - close[:n - h]
    return r


def create_mimo_sequences(
    feats: np.ndarray, tgts: np.ndarray, lookback: int = LOOKBACK
) -> tuple[np.ndarray, np.ndarray]:
    """Создаёт последовательности (lookback) для MIMO-обучения.

    Args:
        feats: (n, n_features) — признаки.
        tgts: (n, n_targets) — таргеты.
        lookback: длина окна (30).

    Returns:
        (X, y): X — (N, lookback, n_features), y — (N, n_targets).
    """
    X, y = [], []
    for i in range(lookback - 1, len(feats)):
        X.append(feats[i - lookback + 1:i + 1])
        y.append(tgts[i])
    return np.asarray(X), np.asarray(y)


def dataset_hash(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Вычисляет sha256-отпечаток датасета (для воспроизводимости).

    Хэш строится по каноническому представлению: сортировка по date,
    только feature_cols, hash_pandas_object. Это позволяет проверить,
    что модель обучена на том же датасете.

    Args:
        df: DataFrame с колонкой date и признаками.
        feature_cols: список признаков (51).

    Returns:
        Строка вида "sha256:abc123...".
    """
    d = df.sort_values("date").reset_index(drop=True)
    payload = pd.util.hash_pandas_object(d[feature_cols], index=True).values
    h = hashlib.sha256(payload.tobytes()).hexdigest()
    return f"sha256:{h}"


def save_metadata(
    out_dir: Path,
    model_name: str,
    *,
    dataset_info: dict[str, Any],
    metrics: dict[str, Any],
    hyperparams: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Сохраняет metadata.json с версией, датой, hash, метриками.

    Args:
        out_dir: директория модели (data/models/<model>).
        model_name: "attention_gru" или "arimax_lstm".
        dataset_info: {"hash", "rows", "range", "cutoff_val", "cutoff_test"}.
        metrics: метрики обучения/валидации.
        hyperparams: гиперпараметры модели.
        extra: дополнительные поля (git_commit, notes).

    Returns:
        Путь к сохранённому metadata.json.
    """
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y.%m.%d-%H%M%S")
    meta = {
        "model": model_name,
        "version": version,
        "trained_at": now.isoformat(),
        "dataset": dataset_info,
        "hyperparams": hyperparams,
        "metrics": metrics,
        "extra": extra or {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    log.info("metadata → %s (version=%s, hash=%s)",
             path, version, dataset_info.get("hash"))
    return path


def save_json(obj: dict[str, Any], path: Path) -> None:
    """Сохраняет dict в JSON (создаёт родительские директории)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)