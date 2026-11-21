"""Миграция данных SQLite → PostgreSQL.

Использует async DatabaseRepository (два драйвера). Идемпотентно:
повторный запуск не создаёт дубликатов (ON CONFLICT DO NOTHING/UPDATE).

Запуск (из корня crypto-forecast-app/):
    python scripts/migrate_sqlite_to_pg.py

Переменные окружения (опционально):
    SQLITE_URL (default: sqlite:///data/crypto.db)
    PG_URL     (default: postgresql://crypto:crypto@postgres:5432/crypto)
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Чтобы работал импорт app.* при запуске из корня
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.database_repository import DatabaseRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("migrate")

SQLITE_URL = "sqlite:///data/crypto.db"
PG_URL = "postgresql://crypto:crypto@postgres:5432/crypto"


# (table, columns, conflict_target)
TABLES = [
    ("klines",
     ["date", "open", "high", "low", "close", "volume"],
     "date"),
    ("fear_greed",
     ["date", "fear_greedy_value"],
     "date"),
    ("onchain_metrics",
     ["date", "hash_rate", "difficulty", "n_tx",
      "miners_revenue_btc", "trade_volume_btc"],
     "date"),
    ("macro_metrics",
     ["date", "SP500", "VIXCLS", "DCOILWTICO",
      "NASDAQQGLDI", "DFF", "GEPUCURRENT"],
     "date"),
    ("predictions",
     ["created_at", "model", "horizon", "anchor", "point",
      "lower", "upper", "is_stale", "q_used", "target_time",
      "is_updated", "alpha_t", "version"],
     None),  # id SERIAL — пропускаем, ON CONFLICT не нужен
    ("model_versions",
     ["model_name", "version", "file_path", "is_active",
      "trained_at", "created_at", "metrics_snapshot",
      "dataset_fingerprint"],
     "model_name, version"),
    ("training_runs",
     ["model_version_id", "started_at", "finished_at", "status",
      "hyperparameters", "metrics", "dataset_fingerprint",
      "decision", "n_train", "n_val", "champion_run_id"],
     None),
]


async def migrate_table(src: DatabaseRepository, dst: DatabaseRepository,
                        table: str, cols: list[str], conflict: str | None) -> tuple[int, int]:
    # 1. Читаем из SQLite
    async with src._connect() as conn:
        rows = await src._fetchall(
            conn, f"SELECT {', '.join(cols)} FROM {table}",
        )
    n_src = len(rows)
    if n_src == 0:
        log.info("%s: 0 rows (skip)", table)
        return 0, 0

    # 2. Пишем в Postgres
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    if conflict:
        updates = ", ".join(
            f"{c}=EXCLUDED.{c}" for c in cols if c not in conflict.split(", ")
        )
        sql += f" ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    else:
        sql += " ON CONFLICT DO NOTHING"

    records = [tuple(r[c] for c in cols) for r in rows]
    async with dst._connect() as conn:
        await dst._executemany(conn, sql, records)

    # 3. Считаем в Postgres
    async with dst._connect() as conn:
        row = await dst._fetchone(conn, f"SELECT COUNT(*) AS c FROM {table}")
    n_dst = int(row["c"]) if row else 0
    log.info("%s: src=%s → dst=%s (inserted/updated %s)", table, n_src, n_dst, n_src)
    return n_src, n_dst


async def main() -> int:
    log.info("SQLite: %s", SQLITE_URL)
    log.info("Postgres: %s", PG_URL)
    src = DatabaseRepository(db_url=SQLITE_URL)
    dst = DatabaseRepository(db_url=PG_URL)

    # Проверка подключения
    try:
        async with dst._connect():
            pass
    except Exception as e:
        log.error("Не удалось подключиться к Postgres: %s", e)
        return 1

    for table, cols, conflict in TABLES:
        try:
            await migrate_table(src, dst, table, cols, conflict)
        except Exception:
            log.exception("Ошибка миграции таблицы %s", table)
            return 1

    log.info("Миграция завершена.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))