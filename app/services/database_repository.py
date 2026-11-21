"""
Репозиторий для работы с базой данных.

Поддерживает SQLite (aiosqlite) и PostgreSQL (asyncpg).
Выбор драйвера — по settings.db_url:
  - sqlite:///data/crypto.db       → aiosqlite
  - postgresql://...               → asyncpg

Все методы асинхронные. SQL-запросы используют ? (SQLite) или $N (Postgres)
через адаптер _sql().
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Async-репозиторий для OHLCV, ончейн, макро, Fear & Greed и прогнозов."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or getattr(settings, "db_url", None) or f"sqlite:///{settings.db_path}"
        self.is_postgres = self.db_url.startswith("postgresql")

        if self.is_postgres:
            self.pg_url = self.db_url
        else:
            self.db_path = Path(self.db_url.replace("sqlite:///", ""))
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # connection + helpers
    # ============================================================

    @asynccontextmanager
    async def _connect(self):
        if self.is_postgres:
            import asyncpg
            conn = await asyncpg.connect(self.pg_url)
            try:
                yield conn
            finally:
                await conn.close()
        else:
            import aiosqlite
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
            finally:
                await conn.close()

    @staticmethod
    def _to_pg(sql: str) -> str:
        """Заменить ? на $1, $2, ... для asyncpg."""
        i = 0
        out = []
        for ch in sql:
            if ch == "?":
                i += 1
                out.append(f"${i}")
            else:
                out.append(ch)
        return "".join(out)

    def _sql(self, sql: str) -> str:
        return self._to_pg(sql) if self.is_postgres else sql

    async def _fetchall(self, conn, sql: str, params: tuple = ()) -> list[dict]:
        q = self._sql(sql)
        if self.is_postgres:
            rows = await conn.fetch(q, *params)
            return [dict(r) for r in rows]
        cur = await conn.execute(q, params)
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def _fetchone(self, conn, sql: str, params: tuple = ()) -> dict | None:
        q = self._sql(sql)
        if self.is_postgres:
            r = await conn.fetchrow(q, *params)
            return dict(r) if r else None
        cur = await conn.execute(q, params)
        r = await cur.fetchone()
        return dict(r) if r else None

    async def _execute(self, conn, sql: str, params: tuple = ()) -> int:
        q = self._sql(sql)
        if self.is_postgres:
            await conn.execute(q, *params)
            return 0
        cur = await conn.execute(q, params)
        await conn.commit()
        return cur.lastrowid or 0

    async def _executemany(self, conn, sql: str, records: list[tuple]) -> None:
        q = self._sql(sql)
        if self.is_postgres:
            await conn.executemany(q, records)
        else:
            await conn.executemany(q, records)
            await conn.commit()

    # ============================================================
    # init schema (только для SQLite — для Postgres init_postgres.sql)
    # ============================================================

    async def _init_schema(self) -> None:
        """Создаёт таблицы (только для SQLite; Postgres — init_postgres.sql)."""
        if self.is_postgres:
            logger.info("Postgres: схема из init_postgres.sql (пропускаем _init_schema)")
            return

        async with self._connect() as conn:
            await conn.execute("""CREATE TABLE IF NOT EXISTS klines (date TEXT PRIMARY KEY, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL)""")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_klines_date ON klines(date)")
            await conn.execute("""CREATE TABLE IF NOT EXISTS onchain_metrics (date TEXT PRIMARY KEY, hash_rate REAL, difficulty REAL, n_tx REAL, miners_revenue_btc REAL, trade_volume_btc REAL)""")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_onchain_date ON onchain_metrics(date)")
            await conn.execute("""CREATE TABLE IF NOT EXISTS macro_metrics (date TEXT PRIMARY KEY, SP500 REAL, VIXCLS REAL, DCOILWTICO REAL, NASDAQQGLDI REAL, DFF REAL, GEPUCURRENT REAL)""")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_metrics(date)")
            await conn.execute("""CREATE TABLE IF NOT EXISTS fear_greed (date TEXT PRIMARY KEY, fear_greedy_value INTEGER)""")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_fear_greed_date ON fear_greed(date)")
            await conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, model TEXT NOT NULL,
                horizon INTEGER NOT NULL, anchor REAL NOT NULL, point REAL NOT NULL,
                lower REAL, upper REAL, is_stale INTEGER NOT NULL DEFAULT 0,
                q_used REAL, target_time TEXT, is_updated INTEGER NOT NULL DEFAULT 0,
                alpha_t REAL, version TEXT)""")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at)")
            await conn.execute("""CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL, version TEXT NOT NULL,
                file_path TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 0, trained_at TEXT NOT NULL,
                created_at TEXT NOT NULL, metrics_snapshot TEXT, dataset_fingerprint TEXT,
                UNIQUE (model_name, version))""")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(model_name, is_active)")
            await conn.execute("""CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, model_version_id INTEGER NOT NULL REFERENCES model_versions(id),
                started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, hyperparameters TEXT,
                metrics TEXT, dataset_fingerprint TEXT, decision TEXT, n_train INTEGER, n_val INTEGER,
                champion_run_id TEXT)""")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status, started_at DESC)")
            await conn.commit()
        logger.info(f"SQLite schema инициализирована: {self.db_path}")

    # ============================================================
    # klines
    # ============================================================

    async def save_klines(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame должен содержать колонки: {required}")
        rows = df.copy()
        rows["date"] = pd.to_datetime(rows["date"], utc=True)
        now = pd.Timestamp.now(tz="UTC")
        rows = rows[rows["date"] + pd.Timedelta(hours=4) <= now]
        if rows.empty:
            return 0
        rows["date"] = rows["date"].astype(str)
        records = [tuple(r) for r in rows[["date", "open", "high", "low", "close", "volume"]].itertuples(index=False)]
        async with self._connect() as conn:
            await self._executemany(conn, """
                INSERT INTO klines (date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume
            """, records)
        logger.info(f"Сохранено {len(rows)} свечей в klines")
        return len(rows)

    async def get_last_date(self) -> Optional[pd.Timestamp]:
        async with self._connect() as conn:
            row = await self._fetchone(conn, "SELECT MAX(date) AS m FROM klines")
        if not row or row.get("m") is None:
            return None
        return pd.to_datetime(row["m"], utc=True)

    async def get_klines(self, limit: Optional[int] = 500,
                         date_from: Optional[str | pd.Timestamp] = None,
                         date_to: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        clauses, params = [], []
        if date_from is not None:
            clauses.append("date >= ?")
            params.append(pd.Timestamp(date_from, tz="UTC").isoformat())
        if date_to is not None:
            clauses.append("date <= ?")
            params.append(pd.Timestamp(date_to, tz="UTC").isoformat())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = "ORDER BY date DESC" if limit else "ORDER BY date"
        q = f"SELECT date, open, high, low, close, volume FROM klines{where} {order}"
        if limit:
            q += " LIMIT ?"
            params.append(limit)
        async with self._connect() as conn:
            rows = await self._fetchall(conn, q, tuple(params))
        if not rows:
            return pd.DataFrame(columns=["date","open","high","low","close","volume"])
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values("date").reset_index(drop=True)

    async def count_klines(self) -> int:
        async with self._connect() as conn:
            row = await self._fetchone(conn, "SELECT COUNT(*) AS c FROM klines")
        return int(row["c"]) if row else 0

    # ============================================================
    # onchain_metrics
    # ============================================================

    async def save_onchain(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = ["date", "hash_rate", "difficulty", "n_tx", "miners_revenue_btc", "trade_volume_btc"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Нет колонок: {missing}")
        rows = df[cols].copy()
        rows["date"] = pd.to_datetime(rows["date"], utc=True).astype(str)
        rows = rows.replace({float("nan"): None})
        records = [tuple(r) for r in rows.itertuples(index=False)]
        async with self._connect() as conn:
            await self._executemany(conn, """
                INSERT INTO onchain_metrics
                    (date, hash_rate, difficulty, n_tx, miners_revenue_btc, trade_volume_btc)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    hash_rate=excluded.hash_rate, difficulty=excluded.difficulty,
                    n_tx=excluded.n_tx, miners_revenue_btc=excluded.miners_revenue_btc,
                    trade_volume_btc=excluded.trade_volume_btc
            """, records)
        logger.info(f"Сохранено {len(rows)} строк в onchain_metrics")
        return len(rows)

    async def get_onchain(self, limit: Optional[int] = None,
                          date_from: Optional[str | pd.Timestamp] = None,
                          date_to: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        clauses, params = [], []
        if date_from is not None:
            clauses.append("date >= ?"); params.append(pd.Timestamp(date_from, tz="UTC").isoformat())
        if date_to is not None:
            clauses.append("date <= ?"); params.append(pd.Timestamp(date_to, tz="UTC").isoformat())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = "ORDER BY date DESC" if limit else "ORDER BY date"
        q = ("SELECT date, hash_rate, difficulty, n_tx, miners_revenue_btc, trade_volume_btc "
             f"FROM onchain_metrics{where} {order}")
        if limit:
            q += " LIMIT ?"; params.append(limit)
        async with self._connect() as conn:
            rows = await self._fetchall(conn, q, tuple(params))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values("date").reset_index(drop=True)

    async def get_onchain_last_date(self) -> Optional[pd.Timestamp]:
        async with self._connect() as conn:
            row = await self._fetchone(conn, "SELECT MAX(date) AS m FROM onchain_metrics")
        return pd.to_datetime(row["m"], utc=True) if row and row.get("m") else None

    async def count_onchain(self) -> int:
        async with self._connect() as conn:
            row = await self._fetchone(conn, "SELECT COUNT(*) AS c FROM onchain_metrics")
        return int(row["c"]) if row else 0

    # ============================================================
    # macro_metrics
    # ============================================================

    async def save_macro(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cols = ["date", "SP500", "VIXCLS", "DCOILWTICO", "NASDAQQGLDI", "DFF", "GEPUCURRENT"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Нет колонок: {missing}")
        rows = df[cols].copy()
        rows["date"] = pd.to_datetime(rows["date"], utc=True).astype(str)
        rows = rows.replace({float("nan"): None})
        records = [tuple(r) for r in rows.itertuples(index=False)]
        async with self._connect() as conn:
            await self._executemany(conn, """
                INSERT INTO macro_metrics
                    (date, SP500, VIXCLS, DCOILWTICO, NASDAQQGLDI, DFF, GEPUCURRENT)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    SP500=excluded.SP500, VIXCLS=excluded.VIXCLS,
                    DCOILWTICO=excluded.DCOILWTICO, NASDAQQGLDI=excluded.NASDAQQGLDI,
                    DFF=excluded.DFF, GEPUCURRENT=excluded.GEPUCURRENT
            """, records)
        logger.info(f"Сохранено {len(rows)} строк в macro_metrics")
        return len(rows)

    async def get_macro(self, limit: Optional[int] = None,
                        date_from: Optional[str | pd.Timestamp] = None,
                        date_to: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        clauses, params = [], []
        if date_from is not None:
            clauses.append("date >= ?"); params.append(pd.Timestamp(date_from, tz="UTC").isoformat())
        if date_to is not None:
            clauses.append("date <= ?"); params.append(pd.Timestamp(date_to, tz="UTC").isoformat())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = "ORDER BY date DESC" if limit else "ORDER BY date"
        q = ("SELECT date, SP500, VIXCLS, DCOILWTICO, NASDAQQGLDI, DFF, GEPUCURRENT "
             f"FROM macro_metrics{where} {order}")
        if limit:
            q += " LIMIT ?"; params.append(limit)
        async with self._connect() as conn:
            rows = await self._fetchall(conn, q, tuple(params))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values("date").reset_index(drop=True)

    async def get_macro_last_date(self) -> Optional[pd.Timestamp]:
        async with self._connect() as conn:
            row = await self._fetchone(conn, "SELECT MAX(date) AS m FROM macro_metrics")
        return pd.to_datetime(row["m"], utc=True) if row and row.get("m") else None

    async def count_macro(self) -> int:
        async with self._connect() as conn:
            row = await self._fetchone(conn, "SELECT COUNT(*) AS c FROM macro_metrics")
        return int(row["c"]) if row else 0

    # ============================================================
    # fear_greed
    # ============================================================

    async def save_fear_greed(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        if "date" not in df.columns or "fear_greedy_value" not in df.columns:
            raise ValueError("Нужны колонки: date, fear_greedy_value")
        rows = df[["date", "fear_greedy_value"]].copy()
        rows["date"] = pd.to_datetime(rows["date"], utc=True).astype(str)
        rows["fear_greedy_value"] = rows["fear_greedy_value"].astype(int)
        records = [(str(r.date), int(r.fear_greedy_value)) for r in rows.itertuples(index=False)]
        async with self._connect() as conn:
            await self._executemany(conn, """
                INSERT INTO fear_greed (date, fear_greedy_value) VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET fear_greedy_value=excluded.fear_greedy_value
            """, records)
        logger.info(f"Сохранено {len(records)} строк в fear_greed")
        return len(records)

    async def get_fear_greed(self, limit: Optional[int] = None,
                             date_from: Optional[str | pd.Timestamp] = None,
                             date_to: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        clauses, params = [], []
        if date_from is not None:
            clauses.append("date >= ?"); params.append(pd.Timestamp(date_from, tz="UTC").isoformat())
        if date_to is not None:
            clauses.append("date <= ?"); params.append(pd.Timestamp(date_to, tz="UTC").isoformat())
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = "ORDER BY date DESC" if limit else "ORDER BY date"
        q = f"SELECT date, fear_greedy_value FROM fear_greed{where} {order}"
        if limit:
            q += " LIMIT ?"; params.append(limit)
        async with self._connect() as conn:
            rows = await self._fetchall(conn, q, tuple(params))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["fear_greedy_value"] = df["fear_greedy_value"].astype(int)
        return df.sort_values("date").reset_index(drop=True)

    async def get_last_fear_greed(self, n: int = 1) -> dict | None:
        df = await self.get_fear_greed(limit=n)
        if df.empty:
            return None
        r = df.iloc[-1]
        return {"date": str(r["date"]), "fear_greedy_value": int(r["fear_greedy_value"])}

    async def get_fear_greed_last_date(self) -> Optional[pd.Timestamp]:
        async with self._connect() as conn:
            row = await self._fetchone(conn, "SELECT MAX(date) AS m FROM fear_greed")
        return pd.to_datetime(row["m"], utc=True) if row and row.get("m") else None

    async def count_fear_greed(self) -> int:
        async with self._connect() as conn:
            row = await self._fetchone(conn, "SELECT COUNT(*) AS c FROM fear_greed")
        return int(row["c"]) if row else 0

    async def get_last_klines(self, n: int = 2) -> list[dict]:
        async with self._connect() as conn:
            rows = await self._fetchall(conn,
                "SELECT date, open, high, low, close, volume FROM klines ORDER BY date DESC LIMIT ?",
                (n,))
        return list(reversed(rows))

    async def get_last_close(self) -> dict | None:
        async with self._connect() as conn:
            row = await self._fetchone(conn,
                "SELECT date, close FROM klines ORDER BY date DESC LIMIT 1")
        return row

    # ============================================================
    # predictions
    # ============================================================

    async def save_prediction(self, model: str, horizon: int, anchor: float, point: float,
                              lower: Optional[float] = None, upper: Optional[float] = None,
                              is_stale: bool = False, q_used: Optional[float] = None,
                              alpha_t: Optional[float] = None, version: Optional[str] = None) -> int:
        created_at = datetime.utcnow()
        target_time = (created_at + timedelta(hours=horizon * 4)).isoformat()
        async with self._connect() as conn:
            if self.is_postgres:
                row = await conn.fetchrow("""
                    INSERT INTO predictions
                        (created_at, model, version, horizon, anchor, point,
                         lower, upper, is_stale, q_used, target_time, alpha_t)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                    RETURNING id
                """, created_at.isoformat(), model, version, horizon, anchor, point,
                     lower, upper, int(is_stale), q_used, target_time, alpha_t)
                return int(row["id"])
            else:
                cur = await conn.execute("""
                    INSERT INTO predictions
                        (created_at, model, version, horizon, anchor, point,
                         lower, upper, is_stale, q_used, target_time, alpha_t)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (created_at.isoformat(), model, version, horizon, anchor, point,
                      lower, upper, int(is_stale), q_used, target_time, alpha_t))
                await conn.commit()
                return int(cur.lastrowid or 0)

    async def get_recent_predictions(self, limit: int = 50) -> pd.DataFrame:
        async with self._connect() as conn:
            rows = await self._fetchall(conn,
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,))
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    async def get_pending_predictions(self) -> list[dict]:
        now = datetime.utcnow().isoformat()
        async with self._connect() as conn:
            return await self._fetchall(conn,
                "SELECT id, model, horizon, point, q_used, target_time "
                "FROM predictions WHERE is_updated = 0 AND target_time IS NOT NULL "
                "AND target_time <= ? ORDER BY target_time", (now,))

    async def get_close_at(self, iso_time: str) -> float | None:
        try:
            dt = datetime.fromisoformat(str(iso_time).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            iso_fmt = dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        except Exception:
            iso_fmt = str(iso_time)
        async with self._connect() as conn:
            row = await self._fetchone(conn,
                "SELECT date, close FROM klines WHERE date <= ? ORDER BY date DESC LIMIT 1",
                (iso_fmt,))
        if not row:
            return None
        try:
            d1 = datetime.fromisoformat(str(row["date"]).replace("Z", "+00:00"))
            d2 = datetime.fromisoformat(str(iso_fmt).replace("Z", "+00:00"))
            if d1.tzinfo: d1 = d1.replace(tzinfo=None)
            if d2.tzinfo: d2 = d2.replace(tzinfo=None)
            if abs((d2 - d1).total_seconds()) > 5 * 3600:
                return None
        except Exception:
            return None
        return float(row["close"])

    async def mark_prediction_updated(self, pred_id: int) -> None:
        async with self._connect() as conn:
            await self._execute(conn, "UPDATE predictions SET is_updated = 1 WHERE id = ?", (pred_id,))

    async def get_cached_predictions(self, model: str, version: str | None,
                                     last_close_iso: str, max_age_seconds: int = 3600) -> list[dict]:
        cutoff = (datetime.utcnow() - timedelta(seconds=max_age_seconds)).isoformat()
        try:
            lc = datetime.fromisoformat(str(last_close_iso).replace("Z", "+00:00"))
            if lc.tzinfo: lc = lc.replace(tzinfo=None)
            lc_iso = lc.isoformat()
        except Exception:
            lc_iso = str(last_close_iso)

        if version is None:
            v_clause = "AND version IS NULL"; params = (model, lc_iso, cutoff)
        else:
            v_clause = "AND version = ?"; params = (model, lc_iso, cutoff, version)

        q = f"""SELECT id, created_at, model, version, horizon, anchor, point, lower, upper,
                       is_stale, q_used, alpha_t
                FROM predictions
                WHERE model = ? AND created_at >= ? AND created_at >= ? {v_clause}
                ORDER BY horizon"""
        async with self._connect() as conn:
            return await self._fetchall(conn, q, params)


# Глобальный экземпляр (создаётся в main.py после инициализации схемы)
db_repository = DatabaseRepository()