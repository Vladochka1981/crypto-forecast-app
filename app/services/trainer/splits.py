"""Сплиты для Trainer: train / val / test по барам."""
from __future__ import annotations
import logging
import pandas as pd

log = logging.getLogger(__name__)


def split_by_bars(
    df: pd.DataFrame,
    test_bars: int,
    val_bars: int,
    train_end_bars: int | None = None,
) -> dict[str, pd.DataFrame]:
    n = len(df)
    if train_end_bars is not None:
        train_end = train_end_bars - val_bars
        if train_end <= 0:
            raise ValueError(f"train_end_bars={train_end_bars} <= val_bars={val_bars}")
        train = df.iloc[:train_end].reset_index(drop=True)
        val = df.iloc[train_end:train_end_bars].reset_index(drop=True)
        test = df.iloc[0:0].reset_index(drop=True)
    else:
        if test_bars + val_bars >= n:
            raise ValueError(f"test_bars+val_bars={test_bars+val_bars} >= n={n}")
        train_end = n - test_bars - val_bars
        train = df.iloc[:train_end].reset_index(drop=True)
        val = df.iloc[train_end:train_end + val_bars].reset_index(drop=True)
        test = df.iloc[train_end + val_bars:].reset_index(drop=True)
    log.info("split: train=%s, val=%s, test=%s", len(train), len(val), len(test))
    return {"train": train, "val": val, "test": test}