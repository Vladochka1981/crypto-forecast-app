"""
Сборщик внешних данных: ончейн (Blockchain.com), Fear & Greed (Alternative.me),
макро (FRED).

Задача модуля — доставить сырые данные из API. Заполнение пропусков (ffill),
ресемплинг и мерж с OHLCV выполняются в FeatureEngineer.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import pandas as pd
from fredapi import Fred

from app.core.config import settings
from app.services.cache_manager import cache_manager

logger = logging.getLogger(__name__)

# FRED-серии
FRED_SERIES = {
    "SP500": "SP500",
    "VIXCLS": "VIXCLS",
    "DCOILWTICO": "DCOILWTICO",
    "NASDAQQGLDI": "NASDAQQGLDI",
    "DFF": "DFF",
    "GEPUCURRENT": "GEPUCURRENT",
}

# Blockchain.com chart names -> колонки
BLOCKCHAIN_CHARTS = {
    "hash-rate": "hash_rate",
    "difficulty": "difficulty",
    "n-transactions": "n_tx",
    "miners-revenue": "miners_revenue_btc",
    "trade-volume": "trade_volume_btc",
}

ONCHAIN_COLUMNS = [
    "date", "hash_rate", "difficulty", "n_tx",
    "miners_revenue_btc", "trade_volume_btc",
]
MACRO_COLUMNS = [
    "date", "SP500", "VIXCLS", "DCOILWTICO",
    "NASDAQQGLDI", "DFF", "GEPUCURRENT",
]


class ExternalDataCollector:
    """Сборщик ончейн- и макро-данных."""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.fred = Fred(api_key=settings.fred_api_key) if settings.fred_api_key else None

    # ---------- Blockchain.com ----------

    async def _fetch_blockchain_chart(
        self,
        chart: str,
        start_date: str = "2020-01-01",
        chunk_days: int = 90,
        pause_seconds: float = 0.5,
    ) -> pd.DataFrame:
        """
        Загрузка одного chart из Blockchain.com чанками по chunk_days,
        чтобы API не сэмплировал данные (daily вместо ~4-days).
        Если данных нет — возвращает пустой DataFrame (не ошибка).
        """
        from datetime import datetime as dt
        from datetime import timedelta

        start = dt.strptime(start_date, "%Y-%m-%d")
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        col = BLOCKCHAIN_CHARTS.get(chart, chart)
        frames: list[pd.DataFrame] = []

        current = start
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            while current <= end:
                chunk_end = min(current + timedelta(days=chunk_days - 1), end)
                params = {
                    "start": int(current.timestamp()),
                    "end": int(chunk_end.timestamp()),
                    "format": "json",
                    "timezone": "UTC",
                }
                url = f"{settings.blockchain_api_url}/charts/{chart}"
                try:
                    async with session.get(url, params=params) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"{chart} {resp.status}")
                        data = await resp.json()
                    values = data.get("values", [])
                    if values:
                        df = pd.DataFrame(values)
                        df["date"] = pd.to_datetime(df["x"], unit="s", utc=True)
                        df = df.rename(columns={"y": col})[["date", col]]
                        frames.append(df)
                except Exception as e:
                    logger.warning(
                        f"Blockchain.com {chart} {current.date()}-{chunk_end.date()}: {e}"
                    )
                current = chunk_end + timedelta(days=1)
                if current <= end:
                    await asyncio.sleep(pause_seconds)

        if not frames:
            return pd.DataFrame(columns=["date", col])

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
        return merged.reset_index(drop=True)

    async def get_onchain(self, force_refresh: bool = False) -> tuple[pd.DataFrame, bool]:
        """
        Полная загрузка всех ончейн-метрик.
        Возвращает сырые данные (без ffill).
        """
        cache_key = "external_onchain"

        async def loader() -> list[dict]:
            frames = []
            for chart in BLOCKCHAIN_CHARTS:
                df = await self._fetch_blockchain_chart(chart)
                if not df.empty:
                    frames.append(df)
            if not frames:
                raise RuntimeError("Blockchain.com: все чарты вернули пустой ответ")
            merged = frames[0]
            for f in frames[1:]:
                merged = merged.merge(f, on="date", how="outer")
            # гарантируем все 5 колонок, даже если часть серий пустая
            for col in BLOCKCHAIN_CHARTS.values():
                if col not in merged.columns:
                    merged[col] = float("nan")
            merged = merged.sort_values("date").reset_index(drop=True)
            out = merged[ONCHAIN_COLUMNS].copy()
            out["date"] = out["date"].astype(str)
            return out.to_dict(orient="records")

        records, is_stale = await cache_manager.get_or_load_async(
            cache_key, loader, force_refresh=force_refresh,
        )
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df, is_stale

    async def get_onchain_incremental(
        self, force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, bool, int]:
        """
        Инкрементальная загрузка ончейн-метрик с перекрытием 7 дней
        (для компенсации задержек публикации разных чартов).
        """
        from app.services.database_repository import db_repository

        last_date = await db_repository.get_onchain_last_date()   # ★ await
        if last_date is None:
            df, is_stale = await self.get_onchain(force_refresh=force_refresh)
            return df, is_stale, len(df)

        start_date = (last_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        cache_key = "onchain_incremental"

        async def loader() -> list[dict]:
            frames = []
            for chart in BLOCKCHAIN_CHARTS:
                df = await self._fetch_blockchain_chart(
                    chart, start_date=start_date, chunk_days=90,
                )
                if not df.empty:
                    frames.append(df)
            if not frames:
                return []
            merged = frames[0]
            for f in frames[1:]:
                merged = merged.merge(f, on="date", how="outer")
            for col in BLOCKCHAIN_CHARTS.values():
                if col not in merged.columns:
                    merged[col] = float("nan")
            merged = merged.sort_values("date").reset_index(drop=True)
            out = merged[ONCHAIN_COLUMNS].copy()
            out["date"] = out["date"].astype(str)
            return out.to_dict(orient="records")

        records, is_stale = await cache_manager.get_or_load_async(
            cache_key, loader,
            force_refresh=force_refresh,
            ttl_seconds=6 * 3600,
        )
        new_df = pd.DataFrame(records)
        if not new_df.empty:
            new_df["date"] = pd.to_datetime(new_df["date"], utc=True)
        return new_df, is_stale, len(new_df)

    # ---------- Alternative.me (Fear & Greed) ----------

    async def get_fear_greed(self, force_refresh: bool = False) -> tuple[pd.DataFrame, bool]:
        """Fear & Greed Index (вся история)."""
        cache_key = "external_fear_greed"

        async def loader() -> list[dict]:
            url = f"{settings.alternative_api_url}/fng/"
            params = {"limit": 0, "format": "json"}
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Alternative.me: {resp.status}")
                    data = await resp.json()
            rows = data.get("data", [])
            if not rows:
                raise RuntimeError("Alternative.me: пустой ответ")
            out = []
            for r in rows:
                out.append({
                    "date": pd.to_datetime(int(r["timestamp"]), unit="s", utc=True).isoformat(),
                    "fear_greedy_value": int(r["value"]),
                })
            return out

        records, is_stale = await cache_manager.get_or_load_async(
            cache_key, loader, force_refresh=force_refresh,
        )
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values("date").reset_index(drop=True), is_stale

    async def get_fear_greed_incremental(
        self, force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, bool, int]:
        """Инкрементальная загрузка Fear & Greed: только после max(date) в БД."""
        from app.services.database_repository import db_repository

        last_date = await db_repository.get_fear_greed_last_date()   # ★ await
        if last_date is None:
            df, is_stale = await self.get_fear_greed(force_refresh=force_refresh)
            return df, is_stale, len(df)

        cache_key = "fear_greed_incremental"

        async def loader() -> list[dict]:
            # Alternative.me не поддерживает фильтр по дате — качаем всё
            url = f"{settings.alternative_api_url}/fng/"
            params = {"limit": 0, "format": "json"}
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Alternative.me: {resp.status}")
                    data = await resp.json()
            rows = data.get("data", [])
            if not rows:
                return []
            out = []
            for r in rows:
                d = pd.to_datetime(int(r["timestamp"]), unit="s", utc=True)
                if d >= last_date - pd.Timedelta(days=3):
                    out.append({
                        "date": d.isoformat(),
                        "fear_greedy_value": int(r["value"]),
                    })
            return out

        records, is_stale = await cache_manager.get_or_load_async(
            cache_key, loader,
            force_refresh=force_refresh,
            ttl_seconds=6 * 3600,
        )
        new_df = pd.DataFrame(records)
        if not new_df.empty:
            new_df["date"] = pd.to_datetime(new_df["date"], utc=True)
        return new_df, is_stale, len(new_df)

    # ---------- FRED ----------

    def _fetch_fred_one(
        self, series_id: str, start_date: str, end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Синхронная загрузка одной FRED-серии (fredapi синхронный)."""
        if self.fred is None:
            raise RuntimeError("FRED_API_KEY не задан в .env")
        end_date = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        s = self.fred.get_series(
            series_id, observation_start=start_date, observation_end=end_date,
        )
        if s is None or s.empty:
            raise RuntimeError(f"FRED {series_id}: пустой ответ")
        df = pd.DataFrame({"date": s.index, series_id: s.values}).dropna()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.reset_index(drop=True)

    async def get_fred(
        self, start_date: str = "2020-01-01", force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, bool]:
        """
        Полная загрузка всех FRED-серий.
        Возвращает сырые данные (без ffill).
        """
        cache_key = f"external_fred_{start_date}"

        async def loader() -> list[dict]:
            loop = asyncio.get_running_loop()
            results = await asyncio.gather(*[
                loop.run_in_executor(None, self._fetch_fred_one, sid, start_date)
                for sid in FRED_SERIES.values()
            ], return_exceptions=True)
            frames = [
                r for r in results
                if isinstance(r, pd.DataFrame) and not r.empty
            ]
            if not frames:
                raise RuntimeError("FRED: все серии вернули пустой ответ")
            merged = frames[0]
            for f in frames[1:]:
                merged = merged.merge(f, on="date", how="outer")
            for sid in FRED_SERIES.values():
                if sid not in merged.columns:
                    merged[sid] = float("nan")
            merged = merged.sort_values("date").reset_index(drop=True)
            out = merged[MACRO_COLUMNS].copy()
            out["date"] = out["date"].astype(str)
            return out.to_dict(orient="records")

        records, is_stale = await cache_manager.get_or_load_async(
            cache_key, loader, force_refresh=force_refresh,
        )
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        return df, is_stale

    async def get_fred_incremental(
        self, force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, bool, int]:
        """Инкрементальная загрузка FRED с перекрытием 3 дня."""
        from app.services.database_repository import db_repository

        last_date = await db_repository.get_macro_last_date()   # ★ await
        if last_date is None:
            df, is_stale = await self.get_fred(force_refresh=force_refresh)
            return df, is_stale, len(df)

        start_date = (last_date - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        cache_key = "fred_incremental"

        async def loader() -> list[dict]:
            loop = asyncio.get_running_loop()
            results = await asyncio.gather(*[
                loop.run_in_executor(None, self._fetch_fred_one, sid, start_date)
                for sid in FRED_SERIES.values()
            ], return_exceptions=True)
            frames = [
                r for r in results
                if isinstance(r, pd.DataFrame) and not r.empty
            ]
            if not frames:
                return []
            merged = frames[0]
            for f in frames[1:]:
                merged = merged.merge(f, on="date", how="outer")
            for sid in FRED_SERIES.values():
                if sid not in merged.columns:
                    merged[sid] = float("nan")
            merged = merged.sort_values("date").reset_index(drop=True)
            out = merged[MACRO_COLUMNS].copy()
            out["date"] = out["date"].astype(str)
            return out.to_dict(orient="records")

        records, is_stale = await cache_manager.get_or_load_async(
            cache_key, loader,
            force_refresh=force_refresh,
            ttl_seconds=12 * 3600,
        )
        new_df = pd.DataFrame(records)
        if not new_df.empty:
            new_df["date"] = pd.to_datetime(new_df["date"], utc=True)
        return new_df, is_stale, len(new_df)

    # ---------- агрегатор ----------

    async def get_all_external(
        self, start_date: str = "2020-01-01", force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Все внешние данные, смерженные по date (outer join).
        Без ffill — это делает FeatureEngineer.
        """
        onchain, s1 = await self.get_onchain(force_refresh)
        fng, s2 = await self.get_fear_greed(force_refresh)
        fred, s3 = await self.get_fred(start_date, force_refresh)

        merged = onchain.merge(fng, on="date", how="outer")
        merged = merged.merge(fred, on="date", how="outer")
        merged = merged.sort_values("date").reset_index(drop=True)

        stale_flags = {"onchain": s1, "fear_greed": s2, "fred": s3}
        return merged, stale_flags


# Глобальный экземпляр
external_data_collector = ExternalDataCollector()