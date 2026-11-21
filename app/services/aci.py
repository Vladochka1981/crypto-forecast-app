"""Adaptive Conformal Inference (ACI) для онлайн-калибровки интервалов.

Одна ACI на пару (модель, горизонт). Состояние персистится в JSON.
Ссылка: Gibbs & Candès, 2021.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HORIZONS: tuple[int, ...] = (1, 3, 6, 12, 18)


class AdaptiveConformal:
    """ACI per horizon: alpha_{t+1} = alpha_t + gamma * (alpha - 1{miss})."""

    def __init__(
        self,
        alpha_target: float = 0.05,
        gamma: float = 0.01,
        lookback: int = 100,
        alpha_min: float = 0.01,
        alpha_max: float = 0.5,
    ) -> None:
        self.alpha_target = float(alpha_target)
        self.gamma = float(gamma)
        self.lookback = int(lookback)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_t: dict[int, float] = {h: self.alpha_target for h in HORIZONS}
        self.errors: dict[int, list[float]] = {h: [] for h in HORIZONS}
        self.n_updates: dict[int, int] = {h: 0 for h in HORIZONS}
        self.last_update: dict[int, str | None] = {h: None for h in HORIZONS}

    def get_quantile(self, horizon: int) -> float:
        if horizon not in self.alpha_t:
            raise ValueError(f"ACI: horizon={horizon} не поддерживается")
        errs = self.errors[horizon]
        if len(errs) < self.lookback:
            raise RuntimeError(
                f"ACI h={horizon}: буфер {len(errs)} < lookback {self.lookback}. "
                f"aci_state.json не откалиброван."
            )
        return float(np.quantile(errs[-self.lookback:], 1.0 - self.alpha_t[horizon]))

    def update(
        self, horizon: int, y_true: float, y_pred: float, q_used: float
    ) -> float:
        if horizon not in self.alpha_t:
            raise ValueError(f"ACI: horizon={horizon} не поддерживается")
        if not (np.isfinite(y_true) and np.isfinite(y_pred) and np.isfinite(q_used)):
            logger.warning("ACI h=%s: нефинитные значения, update пропущен", horizon)
            return 0.0
        err = abs(float(y_true) - float(y_pred))
        self.errors[horizon].append(err)
        cap = self.lookback * 5
        if len(self.errors[horizon]) > cap:
            self.errors[horizon] = self.errors[horizon][-self.lookback:]
        miss = 1.0 if err > float(q_used) else 0.0
        a = self.alpha_t[horizon] + self.gamma * (self.alpha_target - miss)
        self.alpha_t[horizon] = float(np.clip(a, self.alpha_min, self.alpha_max))
        self.n_updates[horizon] += 1
        self.last_update[horizon] = datetime.utcnow().isoformat()
        return miss

    def to_dict(self) -> dict:
        return {
            "method": "ACI",
            "alpha_target": self.alpha_target,
            "gamma": self.gamma,
            "lookback": self.lookback,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "horizons": {
                str(h): {
                    "alpha_t": round(self.alpha_t[h], 6),
                    "errors_tail": [round(x, 2) for x in self.errors[h][-self.lookback:]],
                    "n_updates": self.n_updates[h],
                    "last_update": self.last_update[h],
                }
                for h in HORIZONS
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveConformal":
        obj = cls(
            alpha_target=d.get("alpha_target", 0.05),
            gamma=d.get("gamma", 0.01),
            lookback=d.get("lookback", 100),
            alpha_min=d.get("alpha_min", 0.01),
            alpha_max=d.get("alpha_max", 0.5),
        )
        for h_str, v in d.get("horizons", {}).items():
            h = int(h_str)
            if h not in obj.alpha_t:
                continue
            obj.alpha_t[h] = float(np.clip(
                v.get("alpha_t", obj.alpha_target), obj.alpha_min, obj.alpha_max
            ))
            obj.errors[h] = [float(x) for x in v.get("errors_tail", [])][-obj.lookback:]
            obj.n_updates[h] = int(v.get("n_updates", 0))
            obj.last_update[h] = v.get("last_update")
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveConformal":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))