"""Калибровка ACI: ошибки гибрида на val → aci_state.json."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.services.aci import AdaptiveConformal, HORIZONS

log = logging.getLogger(__name__)


def calibrate_aci(
    out_dir: str | Path,
    predictions_df: pd.DataFrame,
    *,
    alpha_target: float = 0.05,
    gamma: float = 0.01,
    lookback: int = 100,
    alpha_min: float = 0.01,
    alpha_max: float = 0.5,
) -> Path:
    """Создаёт aci_state.json из DataFrame с колонками horizon/actual/pred.

    predictions_df — результат evaluate(...)["predictions"] на val.
    Ошибки = |actual - pred| по каждому горизонту.
    """
    aci = AdaptiveConformal(
        alpha_target=alpha_target, gamma=gamma, lookback=lookback,
        alpha_min=alpha_min, alpha_max=alpha_max,
    )
    for h in HORIZONS:
        sub = predictions_df[predictions_df["horizon"] == h]
        errs = (sub["actual"] - sub["pred"]).abs().dropna().tolist()
        if len(errs) < lookback:
            raise ValueError(
                f"ACI калибровка h={h}: ошибок {len(errs)} < lookback {lookback}"
            )
        aci.errors[h] = [float(e) for e in errs[-lookback:]]

    path = Path(out_dir) / "aci_state.json"
    aci.save(path)
    log.info(
        "ACI calibration saved: %s (n per h: %s)",
        path, {h: len(aci.errors[h]) for h in HORIZONS},
    )
    return path