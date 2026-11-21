"""
Инженер признаков: единый вход для Web и Trainer.

Собирает 51 признак из OHLCV, ончейн, макро и Fear & Greed.
Winsorization (как в dataset_builder):
  - Z-score (causal, window=30, threshold=3.5): open/high/low + onchain
  - IQR (k=3.0): volume (q1/q3 fit на fit_range, сохраняются)
  - close, difficulty — НЕ winsorized
  - Параметры window/threshold/k — КОНСТАНТЫ
  - q1/q3 — fit на train (fit_range), сохраняются в winsor_params.json

Режимы:
  Trainer: build_features(date_from, date_to, fit_range=(start, end),
                          return_winsor_params=True)
           → winsorization fit на fit_range, возвращает (df, winsor_params)
  Web:     build_features(limit=300, winsor_params=...)
           → winsorization с сохранёнными params, возвращает df
"""
import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import ta

from app.core.config import settings
from app.services.database_repository import db_repository

logger = logging.getLogger(__name__)

_PRICE_COLS = ["open", "high", "low"]
_ONCHAIN_COLS = [
    "hash_rate",
    "n_tx",
    "trade_volume_btc",
    "miners_revenue_btc",
]
_VOLUME_COLS = ["volume"]

_ZSCORE_WINDOW = 30
_ZSCORE_THRESHOLD = 3.5
_IQR_K = 3.0


def _run_db(coro):
    """Sync-обёртка для вызова async db_repository из sync FeatureEngineer.

    Почему так:
      - FeatureEngineer — sync (pandas, winsorize). Его вызывают:
          • predictor.predict (sync, из orchestrator через run_in_threadpool)
          • train_attention_gru / train_arimax_lstm / evaluator / weekly (sync)
        В этих контекстах running loop НЕТ → asyncio.run() безопасен.
      - Если вдруг кто-то вызовет из async-контекста напрямую (без
        run_in_threadpool) — asyncio.run() упадёт с RuntimeError. На этот
        случай выполняем корутину в отдельном потоке (там свой loop).

    Логика 1-в-1: просто дожидаемся результата корутины синхронно.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # нет running loop — обычный путь (predictor/train/weekly)
        return asyncio.run(coro)

    # есть running loop — уводим в отдельный поток
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class FeatureEngineer:
    """Сборка 51 признака + winsorization."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path or settings.gru_model_dir) / settings.gru_config_file
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.feature_cols: list[str] = self.config["feature_cols"]
        self.lookback: int = self.config.get("lookback", 30)

    # ---------- технические индикаторы (1:1 из исходника) ----------
    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["sma_7"] = ta.trend.sma_indicator(data["close"], window=7)
        data["sma_14"] = ta.trend.sma_indicator(data["close"], window=14)
        data["sma_30"] = ta.trend.sma_indicator(data["close"], window=30)
        data["sma_50"] = ta.trend.sma_indicator(data["close"], window=50)
        data["ema_12"] = ta.trend.ema_indicator(data["close"], window=12)
        data["ema_26"] = ta.trend.ema_indicator(data["close"], window=26)
        macd = ta.trend.MACD(data["close"])
        data["macd"] = macd.macd()
        data["macd_signal"] = macd.macd_signal()
        data["macd_diff"] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(data["close"], window=20, window_dev=2)
        data["bb_high"] = bb.bollinger_hband()
        data["bb_mid"] = bb.bollinger_mavg()
        data["bb_low"] = bb.bollinger_lband()
        data["bb_width"] = bb.bollinger_wband()
        data["bb_percent"] = bb.bollinger_pband()
        data["rsi_14"] = ta.momentum.rsi(data["close"], window=14)
        data["stoch_k"] = ta.momentum.stoch(
            data["high"], data["low"], data["close"], window=14, smooth_window=3)
        data["stoch_d"] = ta.momentum.stoch_signal(
            data["high"], data["low"], data["close"], window=14, smooth_window=3)
        data["cci_20"] = ta.trend.cci(
            data["high"], data["low"], data["close"], window=20)
        data["williams_r"] = ta.momentum.williams_r(
            data["high"], data["low"], data["close"], lbp=14)
        data["obv"] = ta.volume.on_balance_volume(data["close"], data["volume"])
        data["mfi_14"] = ta.volume.money_flow_index(
            data["high"], data["low"], data["close"], data["volume"], window=14)
        data["atr_14"] = ta.volatility.average_true_range(
            data["high"], data["low"], data["close"], window=14)
        return data

    @staticmethod
    def _add_returns_volatility(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["returns_1"] = data["close"].pct_change(periods=1)
        data["returns_5"] = data["close"].pct_change(periods=5)
        data["returns_10"] = data["close"].pct_change(periods=10)
        data["returns_30"] = data["close"].pct_change(periods=30)
        data["volatility_10"] = data["returns_1"].rolling(10).std()
        data["volatility_30"] = data["returns_1"].rolling(30).std()
        return data

    @staticmethod
    def _add_macro_lags(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        
        # ★ ФИКС: Приводим к нижнему регистру для поиска (Postgres возвращает sp500, а не SP500)
        cols_lower = {c.lower(): c for c in data.columns}
        
        for col in ["SP500", "VIXCLS", "DCOILWTICO", "NASDAQQGLDI"]:
            actual_col = cols_lower.get(col.lower())
            if actual_col:
                data[f"{col}_lag6"] = data[actual_col].shift(6)
                data[f"{col}_lag12"] = data[actual_col].shift(12)
        for col in ["DFF", "GEPUCURRENT"]:
            actual_col = cols_lower.get(col.lower())
            if actual_col:
                data[f"{col}_lag12"] = data[actual_col].shift(12)
                data[f"{col}_lag24"] = data[actual_col].shift(24)
        return data

    # ---------- winsorization (1:1 с dataset_builder) ----------
    @staticmethod
    def _rolling_zscore_winsorize(
        series: pd.Series, window: int = _ZSCORE_WINDOW,
        threshold: float = _ZSCORE_THRESHOLD,
    ) -> pd.Series:
        roll_mean = series.shift(1).rolling(window, min_periods=1).mean()
        roll_std = series.shift(1).rolling(window, min_periods=1).std()
        z = (series - roll_mean) / roll_std.replace(0, np.nan)
        upper = roll_mean + threshold * roll_std
        lower = roll_mean - threshold * roll_std
        return series.where(z.abs() <= threshold, series.clip(lower, upper))

    def _fit_and_apply_winsorization(
        self, df: pd.DataFrame, fit_range: tuple,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        out = df.copy()
        params: dict[str, Any] = {}
        start, end = fit_range

        def _to_utc(x):
            """Timestamp → tz-aware UTC. Строка → pd.Timestamp(tz='UTC')."""
            if x is None:
                return None
            ts = pd.Timestamp(x)
            return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

        mask = pd.Series(True, index=out.index)
        start_ts, end_ts = _to_utc(start), _to_utc(end)
        if start_ts is not None:
            mask &= out["date"] >= start_ts
        if end_ts is not None:
            mask &= out["date"] <= end_ts
        if not mask.any():
            raise ValueError(f"winsor fit_range пуст: {fit_range}")
        for c in _PRICE_COLS + _ONCHAIN_COLS:
            if c in out.columns and c in self.feature_cols:
                out[c] = self._rolling_zscore_winsorize(out[c])
                params[c] = {"method": "zscore", "window": _ZSCORE_WINDOW,
                             "threshold": _ZSCORE_THRESHOLD}
        for c in _VOLUME_COLS:
            if c in out.columns and c in self.feature_cols:
                train_vals = out.loc[mask, c]
                q1 = float(train_vals.quantile(0.25))
                q3 = float(train_vals.quantile(0.75))
                iqr = q3 - q1
                out[c] = out[c].clip(q1 - _IQR_K * iqr, q3 + _IQR_K * iqr)
                params[c] = {"method": "iqr", "k": _IQR_K, "q1": q1, "q3": q3}
        return out, params

    def _apply_saved_winsorization(
        self, df: pd.DataFrame, params: dict[str, Any],
    ) -> pd.DataFrame:
        out = df.copy()
        for c in _PRICE_COLS + _ONCHAIN_COLS:
            if c in out.columns and c in self.feature_cols and c in params:
                p = params[c]
                out[c] = self._rolling_zscore_winsorize(
                    out[c], window=p["window"], threshold=p["threshold"])
        for c in _VOLUME_COLS:
            if c in out.columns and c in self.feature_cols and c in params:
                p = params[c]
                q1, q3, k = p["q1"], p["q3"], p["k"]
                iqr = q3 - q1
                out[c] = out[c].clip(q1 - k * iqr, q3 + k * iqr)
        return out

    # ---------- сборка ----------
    def _build_raw(
        self,
        date_from: Optional[str | pd.Timestamp] = None,
        date_to: Optional[str | pd.Timestamp] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Собирает сырые 51 признак из БД (без winsorization)."""
        # 1. Заполнить дыры в klines (4h)
        self._fill_irregular_klines()

        # 2. Данные из БД
        ohlcv = _run_db(
            db_repository.get_klines(limit=limit, date_from=date_from, date_to=date_to)
        )
        if ohlcv.empty:
            raise RuntimeError("В БД нет OHLCV.")

        ext_limit = None
        ext_from = date_from
        if limit is not None and date_from is None:
            ext_limit = limit + 80
        elif date_from is not None:
            ext_from = pd.Timestamp(date_from) - pd.Timedelta(days=30)

        onchain = _run_db(
            db_repository.get_onchain(limit=ext_limit, date_from=ext_from, date_to=date_to)
        )
        macro = _run_db(
            db_repository.get_macro(limit=ext_limit, date_from=ext_from, date_to=date_to)
        )
        fng = _run_db(
            db_repository.get_fear_greed(limit=ext_limit, date_from=ext_from, date_to=date_to)
        )

        # 3. ★ ffill ТОЛЬКО GEPUCURRENT в macro (до merge).
        #    ★ ФИКС: Ищем колонку без учета регистра
        macro_cols_lower = {c.lower(): c for c in macro.columns}
        if "gepucurrent" in macro_cols_lower:
            actual_col = macro_cols_lower["gepucurrent"]
            macro[actual_col] = macro[actual_col].ffill()

        # 4. Merge по date
        df = ohlcv.merge(onchain, on="date", how="left")
        df = df.merge(macro, on="date", how="left")
        df = df.merge(fng, on="date", how="left")
        df = df.sort_values("date").reset_index(drop=True)

        # 5. ffill ВСЕХ ext_cols после merge (как dataset_builder)
        #    ★ ФИКС: Ищем колонки без учета регистра
        ext_cols = [
            "hash_rate", "difficulty", "n_tx", "miners_revenue_btc",
            "trade_volume_btc", "SP500", "VIXCLS", "DCOILWTICO",
            "NASDAQQGLDI", "DFF", "GEPUCURRENT", "fear_greedy_value",
        ]
        df_cols_lower = {c.lower(): c for c in df.columns}
        for col in ext_cols:
            actual_col = df_cols_lower.get(col.lower())
            if actual_col:
                df[actual_col] = df[actual_col].ffill()

        # 6. Индикаторы / returns / macro-lags
        df = self._add_indicators(df)
        df = self._add_returns_volatility(df)
        df = self._add_macro_lags(df)

        # 7. Drop исходных макро (лаги остаются)
        #    ★ ФИКС: Удаляем без учета регистра
        drop_cols = ["SP500", "VIXCLS", "DCOILWTICO", "NASDAQQGLDI", "DFF", "GEPUCURRENT"]
        drop_cols_lower = {c.lower() for c in drop_cols}
        df = df.drop(columns=[c for c in df.columns if c.lower() in drop_cols_lower])

        # 8. Обрезать первые 49 строк (прогрев индикаторов)
        df = df.iloc[49:].reset_index(drop=True)

        # 9. Остаточные NaN (от shift(12/24) на стыке) — ffill + dropna.
        df = df.ffill().dropna().reset_index(drop=True)

        # 10. Проверка + [date] + feature_cols
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют признаки ({len(missing)}): {missing}")
        df = df[["date"] + self.feature_cols].copy()
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        if not df["date"].is_monotonic_increasing:
            raise ValueError("date не монотонно возрастает")
        if df["date"].duplicated().any():
            raise ValueError("Дубликаты date")
        feat = df[self.feature_cols]
        n_nan = int(feat.isna().sum().sum())
        n_inf = int(np.isinf(feat.to_numpy(dtype=np.float64)).sum())
        if n_nan or n_inf:
            bad = feat.columns[feat.isna().any() | np.isinf(feat).any()].tolist()
            raise ValueError(f"NaN/inf в признаках: {bad} (NaN={n_nan}, inf={n_inf})")

    def _fill_irregular_klines(self) -> int:
        total = _run_db(db_repository.count_klines())
        if total < 2:
            return 0
        df = _run_db(db_repository.get_klines(limit=total))
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
        diffs = df["date"].diff()
        expected = pd.Timedelta(hours=4)
        gaps = df[diffs > expected]
        if gaps.empty:
            return 0
        new_rows = []
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
            return 0
        return _run_db(db_repository.save_klines(pd.DataFrame(new_rows)))

    # ---------- публичный API ----------
    def build_features(
        self,
        date_from: Optional[str | pd.Timestamp] = None,
        date_to: Optional[str | pd.Timestamp] = None,
        limit: Optional[int] = None,
        fit_range: Optional[tuple] = None,
        winsor_params: Optional[dict] = None,
        return_winsor_params: bool = False,
        validate: bool = True,
        n_keep: Optional[int] = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
        """Собирает 51 признак + winsorization."""
        df = self._build_raw(date_from=date_from, date_to=date_to, limit=limit)
        if winsor_params is not None:
            df = self._apply_saved_winsorization(df, winsor_params)
            params = winsor_params
        elif fit_range is not None:
            df, params = self._fit_and_apply_winsorization(df, fit_range)
        else:
            params = {}
        if limit is not None:
            n = n_keep if n_keep is not None else (self.lookback + 1)
            df = df.iloc[-n:].reset_index(drop=True)
        if validate:
            self._validate(df)
        if return_winsor_params:
            return df, params
        return df

    def build_features_for_chart(self, limit: int = 200) -> pd.DataFrame:
        ohlcv = _run_db(db_repository.get_klines(limit=limit))
        if ohlcv.empty:
            raise RuntimeError("В БД нет OHLCV.")
        df = ohlcv.copy()
        df = self._add_indicators(df)
        df = self._add_returns_volatility(df)
        chart_cols = ["date", "open", "high", "low", "close", "volume",
                      "sma_50", "rsi_14", "macd", "macd_signal", "macd_diff",
                      "bb_high", "bb_low"]
        available = [c for c in chart_cols if c in df.columns]
        result = df[available].copy()
        result["date"] = pd.to_datetime(result["date"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S%z")
        result = result.where(pd.notna(result), None)
        return result


feature_engineer = FeatureEngineer()