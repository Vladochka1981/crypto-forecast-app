"""Сборка датасета для переобучения моделей (Attention-GRU и ARIMAX-LSTM).

Единый источник данных — БД (klines + onchain + macro + fear_greed).
Предобработка:
  - FeatureEngineer.build_features: ffill + индикаторы + лаги (единый эталон).
  - Winsorization: Z-score (open/high/low/onchain) + IQR (volume).
    ВАЖНО: close НЕ winsorized — target = close(t+h) - close(t) должен быть
    реальным, без искажений от прижатия выбросов.
  - Разделение: walk_forward (продакшен) или by_years (Colab).

Ключевые принципы:
  - Лаги НЕ пересчитываются здесь — берутся из FeatureEngineer.
  - Winsorization fit только на train, apply на val/test (без утечки).
  - validate_dataset проверяет качество, но не пересчитывает.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.database_repository import db_repository
from app.services.feature_engineer import feature_engineer

log = logging.getLogger(__name__)

# Winsorization: только open/high/low (close — целевая, не трогаем!)
_PRICE_COLS = ["open", "high", "low"]
_ONCHAIN_COLS = [
    "hash_rate",
    # "difficulty",  ← исключён (монотонный ряд)
    "n_tx",
    "trade_volume_btc",
    "miners_revenue_btc",
]
_VOLUME_COLS = ["volume"]

# Параметры winsorization (соответствуют Colab / тексту ВКР)
_ZSCORE_WINDOW = 30
_ZSCORE_THRESHOLD = 3.5
_IQR_K = 3.0

# Параметры сплита (легко менять)
TEST_WINDOW_DAYS = 30
VAL_FRAC = 0.15


# ---------------------------------------------------------------------------
# Загрузка сырых features из БД
# ---------------------------------------------------------------------------

def build_raw_features(limit: int | None = None) -> pd.DataFrame:
    """Строит сырые features (51 признак) из БД через FeatureEngineer.

    Перед build_features заполняет «дыры» в klines (4h) линейной
    интерполяцией, чтобы FeatureEngineer увидел регулярный ряд.

    Args:
        limit: сколько последних klines взять (None → все).

    Returns:
        DataFrame с колонками [date, <51 признак>], отсортированный по date.
    """
    # Заполнить «дыры» (идемпотентно)
    n_filled = fill_irregular_klines()
    total = limit or db_repository.count_klines()
    log.info("build_raw_features: klines в БД = %s (вставлено «дыр» = %s)",
             total, n_filled)

    df = feature_engineer.build_features(limit=total)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    log.info("build_raw_features: %s, %s..%s",
             df.shape, df["date"].iloc[0], df["date"].iloc[-1])
    return df

# ---------------------------------------------------------------------------
# Заполнение «дыр» в klines (4h)
# ---------------------------------------------------------------------------

def fill_irregular_klines() -> int:
    """Заполняет «дыры» в klines (интервал > 4h) линейной интерполяцией.

    Вставляет пропущенные 4h-точки в БД через UPSERT. Значения open/high/low/
    close/volume интерполируются между соседними свечами. После этого
    FeatureEngineer.build_features увидит регулярный ряд (шаг 4h).

    Идемпотентна: повторный вызов не вставляет дубликаты (UPSERT по date).

    Returns:
        Количество вставленных строк.
    """
    total = db_repository.count_klines()
    if total < 2:
        log.info("fill_irregular_klines: слишком мало строк (%s)", total)
        return 0

    df = db_repository.get_klines(limit=total)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    diffs = df["date"].diff()
    expected = pd.Timedelta(hours=4)
    gaps = df[diffs > expected]

    if gaps.empty:
        log.info("fill_irregular_klines: «дыр» нет")
        return 0

    new_rows: list[dict] = []
    for idx in gaps.index:
        prev_date = df["date"].iloc[idx - 1]
        curr_date = df["date"].iloc[idx]
        gap_hours = (curr_date - prev_date).total_seconds() / 3600
        n_missing = int(gap_hours / 4) - 1
        if n_missing <= 0:
            continue
        for k in range(1, n_missing + 1):
            insert_date = prev_date + pd.Timedelta(hours=4 * k)
            weight = k / (n_missing + 1)
            row = {"date": insert_date}
            for col in ["open", "high", "low", "close", "volume"]:
                prev_val = float(df[col].iloc[idx - 1])
                curr_val = float(df[col].iloc[idx])
                row[col] = prev_val + (curr_val - prev_val) * weight
            new_rows.append(row)

    if not new_rows:
        log.info("fill_irregular_klines: нечего вставлять")
        return 0

    new_df = pd.DataFrame(new_rows)
    n = db_repository.save_klines(new_df)
    log.info("fill_irregular_klines: вставлено %s строк", n)
    return n
# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------

def _rolling_zscore_winsorize(
    series: pd.Series, window: int = _ZSCORE_WINDOW, threshold: float = _ZSCORE_THRESHOLD
) -> pd.Series:
    """Скользящий Z-score (причинное окно) + winsorization.

    Для каждой точки t среднее и std считаются по [t-window, t-1] (shift(1)),
    чтобы не было утечки. Значения с |z| > threshold прижимаются к границе.

    Args:
        series: исходный ряд.
        window: ширина скользящего окна (в свечах).
        threshold: порог в стандартных отклонениях.

    Returns:
        Winsorized ряд (той же длины).
    """
    roll_mean = series.shift(1).rolling(window, min_periods=1).mean()
    roll_std = series.shift(1).rolling(window, min_periods=1).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    upper = roll_mean + threshold * roll_std
    lower = roll_mean - threshold * roll_std
    return series.where(z.abs() <= threshold, series.clip(lower, upper))


def _iqr_winsorize(
    series: pd.Series, k: float = _IQR_K, fit_series: pd.Series | None = None
) -> pd.Series:
    """IQR-винсоризация: квартили по fit_series (train), apply к series.

    Args:
        series: ряд, который обрезаем.
        k: коэффициент IQR (границы q1 - k*IQR, q3 + k*IQR).
        fit_series: ряд для расчёта квартилей (train). Если None — series.

    Returns:
        Winsorized ряд.
    """
    base = fit_series if fit_series is not None else series
    q1, q3 = base.quantile(0.25), base.quantile(0.75)
    iqr = q3 - q1
    return series.clip(q1 - k * iqr, q3 + k * iqr)


def apply_winsorization(
    df: pd.DataFrame, feature_cols: list[str], train_mask: pd.Series
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Применяет winsorization к open/high/low/onchain (Z-score) и volume (IQR).

    ВАЖНО: close НЕ winsorized — target (close(t+h) - close(t)) должен быть
    реальным.

    Args:
        df: DataFrame с признаками (и date).
        feature_cols: список признаков (для фильтрации).
        train_mask: булев Series, True для train-строк (для IQR).

    Returns:
        (winsorized_df, params) — params для сохранения в артефакты.
    """
    out = df.copy()
    params: dict[str, Any] = {}

    for c in _PRICE_COLS + _ONCHAIN_COLS:
        if c in out.columns and c in feature_cols:
            out[c] = _rolling_zscore_winsorize(out[c])
            params[c] = {
                "method": "zscore",
                "window": _ZSCORE_WINDOW,
                "threshold": _ZSCORE_THRESHOLD,
            }

    for c in _VOLUME_COLS:
        if c in out.columns and c in feature_cols:
            train_vals = out.loc[train_mask, c]
            q1, q3 = train_vals.quantile(0.25), train_vals.quantile(0.75)
            iqr = q3 - q1
            out[c] = _iqr_winsorize(out[c], k=_IQR_K, fit_series=train_vals)
            params[c] = {
                "method": "iqr",
                "k": _IQR_K,
                "q1": float(q1),
                "q3": float(q3),
            }

    log.info("apply_winsorization: обработано %s колонок (close НЕ winsorized)",
             len(params))
    return out, params


# ---------------------------------------------------------------------------
# Сплиты
# ---------------------------------------------------------------------------

def split_train_val_test(
    df: pd.DataFrame,
    cutoff_val: str = "2025-01-01",
    cutoff_test: str = "2026-01-01",
) -> dict[str, pd.DataFrame]:
    """Сплит по годам (как в Colab): train < cutoff_val, val < cutoff_test, test.

    Args:
        df: DataFrame с date.
        cutoff_val: начало val (конец train).
        cutoff_test: начало test (конец val).

    Returns:
        dict с ключами 'train', 'val', 'test'.
    """
    cv = pd.Timestamp(cutoff_val, tz="UTC")
    ct = pd.Timestamp(cutoff_test, tz="UTC")
    train = df[df["date"] < cv].reset_index(drop=True)
    val = df[(df["date"] >= cv) & (df["date"] < ct)].reset_index(drop=True)
    test = df[df["date"] >= ct].reset_index(drop=True)
    log.info("split by years: train=%s (%s..%s), val=%s, test=%s",
             len(train), train["date"].min(), train["date"].max(),
             len(val), len(test))
    return {"train": train, "val": val, "test": test}


def split_walk_forward(
    df: pd.DataFrame,
    test_window_days: int = 30,
    conformal_window: int = 48,   # ← 48 точек (8 дней) для conformal
) -> dict[str, pd.DataFrame]:
    """Walk-forward: train (2020..T-conformal), val (T-conformal..T), test (T..last).

    T = last_date - test_window_days.
    conformal_window — сколько последних точек trainval оставить для val
                       (только для conformal; в обучение не идёт).
    """
    last_date = df["date"].max()
    T = last_date - pd.Timedelta(days=test_window_days)

    trainval = df[df["date"] < T].reset_index(drop=True)
    test = df[df["date"] >= T].reset_index(drop=True)

    # val — только последние conformal_window точек (для conformal)
    n_val = min(conformal_window, len(trainval) - 1)
    val = trainval.iloc[-n_val:].reset_index(drop=True)
    train = trainval.iloc[:-n_val].reset_index(drop=True)

    log.info("walk-forward: T=%s, train=%s (%s..%s), val=%s (%s..%s), test=%s (%s..%s)",
             T,
             len(train), train["date"].min(), train["date"].max(),
             len(val), val["date"].min(), val["date"].max(),
             len(test), test["date"].min(), test["date"].max())
    return {"train": train, "val": val, "test": test}

# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------

def validate_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, pd.DataFrame] | None = None,
    winsor_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Проверяет качество датасета (структура, хронология, NaN, утечки).

    Raises:
        ValueError при критичных проблемах (NaN, дубликаты, утечка).

    Returns:
        Отчёт (dict) для сохранения в артефакты.
    """
    report: dict[str, Any] = {}

    # 1. Структура
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют признаки: {missing}")
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("Нет колонок date/close")
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Не numeric: {non_numeric}")
    report["n_rows"] = len(df)
    report["n_features"] = len(feature_cols)

    # 2. Хронология
    if not df["date"].is_monotonic_increasing:
        raise ValueError("date не монотонно возрастает")
    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].head().tolist()
        raise ValueError(f"Дубликаты date: {dups}")
    diffs = df["date"].diff().dropna()
    n_irregular = int((diffs != pd.Timedelta(hours=4)).sum())
    report["n_irregular_steps"] = n_irregular
    if n_irregular > 0:
        log.warning("Irregular date steps: %s (ожидалось 4h)", n_irregular)

    # 3. Пропуски
    feat = df[feature_cols]
    n_nan = int(feat.isna().sum().sum())
    n_inf = int(np.isinf(feat.to_numpy(dtype=np.float64)).sum())
    report["n_nan"] = n_nan
    report["n_inf"] = n_inf
    if n_nan or n_inf:
        bad = feat.columns[feat.isna().any() | np.isinf(feat).any()].tolist()
        raise ValueError(f"NaN/inf в признаках: {bad} (NaN={n_nan}, inf={n_inf})")

    # 4. Winsorization (доли обрезанных)
    if winsor_params:
        for col, p in winsor_params.items():
            if col not in df.columns:
                continue
            if p["method"] == "iqr":
                q1, q3, k = p["q1"], p["q3"], p["k"]
                iqr = q3 - q1
                n_clipped = int(((df[col] < q1 - k * iqr) | (df[col] > q3 + k * iqr)).sum())
                report[f"wins_{col}_clipped"] = n_clipped

    # 5. Утечки (train/val/test не пересекаются)
    if splits:
        for a, b in [("train", "val"), ("val", "test"), ("train", "test")]:
            if a in splits and b in splits and len(splits[a]) and len(splits[b]):
                if splits[a]["date"].max() >= splits[b]["date"].min():
                    raise ValueError(f"Утечка: max({a}) >= min({b})")
        report["splits"] = {
            k: {"n": len(v),
                "from": str(v["date"].min()) if len(v) else None,
                "to": str(v["date"].max()) if len(v) else None}
            for k, v in splits.items()
        }

    # 6. Диапазоны
    if "rsi_14" in df.columns:
        bad = int(((df["rsi_14"] < -1) | (df["rsi_14"] > 101)).sum())
        report["rsi_out_of_range"] = bad
        if bad > 0:
            log.warning("rsi_14 вне [0,100]: %s", bad)
    if "fear_greedy_value" in df.columns:
        bad = int(((df["fear_greedy_value"] < 0) | (df["fear_greedy_value"] > 100)).sum())
        report["fear_greed_out_of_range"] = bad
        if bad > 0:
            log.warning("fear_greed вне [0,100]: %s", bad)

    log.info("validate_dataset: %s", json.dumps(report, default=str, indent=2))
    return report


# ---------------------------------------------------------------------------
# Высокоуровневая сборка
# ---------------------------------------------------------------------------

def build_dataset_for_training(
    split: str = "walk_forward",
    cutoff_val: str = "2025-01-01",
    cutoff_test: str = "2026-01-01",
    test_window_days: int = TEST_WINDOW_DAYS,
    conformal_window: int = 48,
    limit: int | None = None,
) -> dict[str, Any]:
    """Собирает датасет: raw → winsorization → split → validate.

    Args:
        split: "walk_forward" (продакшен) или "by_years" (Colab).
        cutoff_val: начало val (для by_years).
        cutoff_test: начало test (для by_years).
        test_window_days: сколько дней для test (для walk_forward).
        val_frac: доля val от trainval (для walk_forward).
        limit: сколько последних klines (None → все).

    Returns:
        dict: splits, winsor_params, feature_cols, report.
    """
    feature_cols = feature_engineer.feature_cols
    df_raw = build_raw_features(limit=limit)

    # train_mask для IQR (зависит от сплита)
    if split == "walk_forward":
        T = df_raw["date"].max() - pd.Timedelta(days=test_window_days)
        train_mask = df_raw["date"] < T
    else:
        train_mask = df_raw["date"] < pd.Timestamp(cutoff_val, tz="UTC")

    df_wins, winsor_params = apply_winsorization(df_raw, feature_cols, train_mask)

    if split == "walk_forward":
        splits = split_walk_forward(df_wins, test_window_days, conformal_window)
    else:
        splits = split_train_val_test(df_wins, cutoff_val, cutoff_test)

    report = validate_dataset(df_wins, feature_cols, splits, winsor_params)
    return {
        "splits": splits,
        "winsor_params": winsor_params,
        "feature_cols": feature_cols,
        "report": report,
    }


def save_winsorization_params(params: dict[str, Any], path: Path) -> None:
    """Сохраняет параметры winsorization в JSON (для артефактов модели)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    log.info("winsorization params → %s", path)
    
    
def print_dataset_summary(ds: dict[str, Any]) -> None:
    """Печатает сводку по собранному датасету (для быстрой проверки).

    Args:
        ds: результат build_dataset_for_training.
    """
    splits = ds["splits"]
    report = ds["report"]
    winsor_params = ds["winsor_params"]
    feature_cols = ds["feature_cols"]

    print("\n" + "=" * 80)
    print("СВОДКА ПО ДАТАСЕТУ")
    print("=" * 80)

    # 1. Сплиты
    print(f"\n📊 Сплиты:")
    for name, df in splits.items():
        if len(df) == 0:
            print(f"  {name:>6}: пусто")
            continue
        print(f"  {name:>6}: {len(df):>6} строк  "
              f"({df['date'].min()} .. {df['date'].max()})")

    # 2. Признаки
    print(f"\n📊 Признаки:")
    print(f"  Всего: {len(feature_cols)}")
    print(f"  Список: {feature_cols[:5]}... {feature_cols[-3:]}")

    # 3. Winsorization
    print(f"\n📊 Winsorization (fit на train):")
    if not winsor_params:
        print("  (нет)")
    else:
        for col, p in winsor_params.items():
            if p["method"] == "zscore":
                print(f"  {col:>20}: zscore (window={p['window']}, "
                      f"threshold={p['threshold']})")
            elif p["method"] == "iqr":
                print(f"  {col:>20}: iqr (k={p['k']}, "
                      f"q1={p['q1']:.2f}, q3={p['q3']:.2f})")

    # 4. close winsorized?
    close_wins = "close" in winsor_params
    print(f"\n📊 close winsorized: {close_wins}  "
          f"({'⚠️ ИСКАЖАЕТ target' if close_wins else '✅ target честный'})")

    # 5. Валидация
    print(f"\n📊 Валидация (validate_dataset):")
    print(f"  n_rows:     {report.get('n_rows')}")
    print(f"  n_features: {report.get('n_features')}")
    print(f"  n_nan:      {report.get('n_nan')}  "
          f"{'❌' if report.get('n_nan') else '✅'}")
    print(f"  n_inf:      {report.get('n_inf')}  "
          f"{'❌' if report.get('n_inf') else '✅'}")
    print(f"  n_irregular_steps: {report.get('n_irregular_steps')}  "
          f"{'⚠️' if report.get('n_irregular_steps') else '✅'}")

    if "rsi_out_of_range" in report:
        v = report["rsi_out_of_range"]
        print(f"  rsi_14 out of range: {v}  {'⚠️' if v else '✅'}")
    if "fear_greed_out_of_range" in report:
        v = report["fear_greed_out_of_range"]
        print(f"  fear_greed out of range: {v}  {'⚠️' if v else '✅'}")

    # 6. Сплиты (из report)
    if "splits" in report:
        print(f"\n📊 Сплиты (из report):")
        for k, v in report["splits"].items():
            print(f"  {k:>6}: n={v['n']}, "
                  f"{v['from']} .. {v['to']}")

    print("\n" + "=" * 80)