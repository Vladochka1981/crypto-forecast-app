"""
Сборщик данных из внешних API (Binance).

Поддерживает:
- get_historical: полная история с пагинацией (для первичного заполнения).
- get_latest: последние N свечей.
- get_incremental: только новые свечи после max(date) в БД.

Кэширование и fallback — через CacheManager (async-версия).
"""
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
import pandas as pd

from app.core.config import settings
from app.services.cache_manager import cache_manager

logger = logging.getLogger(__name__)

KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class DataCollector:
    """Асинхронный сборщик рыночных данных с Binance."""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.base_url = settings.binance_api_url

    # ---------- внутренние ----------

    async def _request_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
    ) -> list[list[Any]]:
        """Сырой запрос к /api/v3/klines с опциональными startTime/endTime."""
        url = f"{self.base_url}/api/v3/klines"
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Binance вернул {resp.status}: {text[:200]}"
                    )
                data = await resp.json()
                if not isinstance(data, list):
                    raise RuntimeError("Binance вернул неожиданный формат")
                return data

    @staticmethod
    def _klines_to_dataframe(raw: list[list[Any]]) -> pd.DataFrame:
        """Парсит сырой ответ Binance в DataFrame с колонками date, OHLCV."""
        if not raw:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = df.sort_values("date").reset_index(drop=True)
        return df[OHLCV_COLUMNS]

    @staticmethod
    def _df_to_records(df: pd.DataFrame) -> list[dict]:
        """DataFrame → JSON-совместимый список словарей."""
        out = df.copy()
        out["date"] = out["date"].astype(str)
        return out.to_dict(orient="records")

    @staticmethod
    def _records_to_df(records: list[dict]) -> pd.DataFrame:
        """Список словарей → DataFrame с типом datetime для date."""
        if not records:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df

    # ---------- публичные ----------

    async def get_latest(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "4h",
        limit: int = 1000,
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, bool]:
        """Последние N свечей с кэшем. Возвращает (DataFrame, is_stale)."""
        cache_key = f"binance_latest_{symbol}_{interval}_{limit}"

        async def loader() -> list[dict]:
            raw = await self._request_klines(symbol, interval, limit=limit)
            if not raw:
                raise RuntimeError("Binance вернул пустой ответ")
            df = self._klines_to_dataframe(raw)
            return self._df_to_records(df)

        records, is_stale = await cache_manager.get_or_load_async(
            cache_key, loader, force_refresh=force_refresh
        )
        return self._records_to_df(records), is_stale

    async def get_historical(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "4h",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, bool]:
        """Полная история с пагинацией. Возвращает (DataFrame, is_stale)."""
        end_date = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cache_key = f"binance_hist_{symbol}_{interval}_{start_date}_{end_date}"

        async def loader() -> list[dict]:
            start_ms = int(pd.to_datetime(start_date, utc=True).timestamp() * 1000)
            end_ms = int(pd.to_datetime(end_date, utc=True).timestamp() * 1000)
            all_raw: list[list[Any]] = []
            current_start = start_ms
            while current_start < end_ms:
                raw = await self._request_klines(
                    symbol, interval, limit=1000,
                    start_time_ms=current_start, end_time_ms=end_ms,
                )
                if not raw:
                    break
                all_raw.extend(raw)
                last_ms = int(raw[-1][0])
                if last_ms <= current_start:
                    break
                current_start = last_ms + 1
            if not all_raw:
                raise RuntimeError("Binance вернул пустой ответ")
            df = self._klines_to_dataframe(all_raw)
            return self._df_to_records(df)

        records, is_stale = await cache_manager.get_or_load_async(
            cache_key, loader, force_refresh=force_refresh
        )
        return self._records_to_df(records), is_stale

    async def get_incremental(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "4h",
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, bool, int]:
        """
        Инкрементальная загрузка: только свечи после max(date) в БД.
        Если БД пуста — грузит последние 1000 свечей.

        Возвращает:
            (DataFrame с новыми свечами, is_stale, n_new_rows)
        """
        from app.services.database_repository import db_repository

        last_date = await db_repository.get_last_date()   # ★ await
        if last_date is None:
            df, is_stale = await self.get_latest(
                symbol, interval, limit=1000, force_refresh=force_refresh
            )
            await db_repository.save_klines(df)           # ★ await
            return df, is_stale, len(df)

        start_ms = int((last_date - pd.Timedelta(days=1)).timestamp() * 1000)   # было: last_date + 1
        cache_key = f"binance_incr_{symbol}_{interval}_{start_ms}"

        async def loader() -> list[dict]:
            raw = await self._request_klines(
                symbol, interval, limit=1000, start_time_ms=start_ms,
            )
            if not raw:
                return []
            df = self._klines_to_dataframe(raw)
            return self._df_to_records(df)

        records, is_stale = await cache_manager.get_or_load_async(
            cache_key, loader, force_refresh=force_refresh
        )
        new_df = self._records_to_df(records)
        if not new_df.empty:
            await db_repository.save_klines(new_df)       # ★ await
        return new_df, is_stale, len(new_df)


# Глобальный экземпляр
data_collector = DataCollector()