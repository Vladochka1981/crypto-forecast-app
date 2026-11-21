"""Планировщик фоновых задач: сбор данных + еженедельное переобучение.

Async-версия: AsyncIOScheduler в том же event loop, что FastAPI.
Все вызовы db_repository — await.
"""
import logging
import subprocess
import sys
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.services.data_collector import data_collector
from app.services.external_data_collector import external_data_collector
from app.services.database_repository import db_repository
from app.services.att_gru_predictor import att_gru_predictor
from app.services.arimax_predictor import arimax_predictor

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def _update_aci():
    """Обновляет ACI (alpha_t) по наступившим фактам для обеих моделей."""
    try:
        pending = await db_repository.get_pending_predictions()
        if not pending:
            logger.info("scheduler: _update_aci — нет прогнозов для обновления")
            return

        n_updates = 0
        for row in pending:
            y_true = await db_repository.get_close_at(row["target_time"])
            if y_true is None:
                continue  # свеча ещё не закрыта, ждём следующего цикла

            q_used = row["q_used"]
            if q_used is None:
                await db_repository.mark_prediction_updated(row["id"])
                continue

            model = row["model"]
            h = row["horizon"]
            point = row["point"]

            if model == "attention_gru":
                att_gru_predictor.update_aci(h, y_true, point, q_used)
            elif model == "arimax_lstm":
                arimax_predictor.update_aci(h, y_true, point, q_used)

            await db_repository.mark_prediction_updated(row["id"])
            n_updates += 1

        logger.info("scheduler: _update_aci обновил %s прогнозов", n_updates)
    except Exception:
        logger.exception("scheduler: _update_aci failed")


async def _collect_klines():
    """Каждые 4 часа — свечи из Binance (incremental, save внутри)."""
    try:
        df, _, n = await data_collector.get_incremental()
        logger.info("scheduler: klines incremental: %s new rows", n)
        # Кэш price обновится при следующем /dashboard (TTL 60с)
        await _update_aci()
    except Exception:
        logger.exception("scheduler: collect_klines failed")


async def _collect_external():
    """Раз в день — F&G, onchain, macro (incremental + save)."""
    try:
        fng, _, _ = await external_data_collector.get_fear_greed_incremental()
        if not fng.empty:
            n = await db_repository.save_fear_greed(fng)
            logger.info("scheduler: fng saved %s", n)

        onchain, _, _ = await external_data_collector.get_onchain_incremental()
        if not onchain.empty:
            n = await db_repository.save_onchain(onchain)
            logger.info("scheduler: onchain saved %s", n)

        macro, _, _ = await external_data_collector.get_fred_incremental()
        if not macro.empty:
            n = await db_repository.save_macro(macro)
            logger.info("scheduler: macro saved %s", n)
    except Exception:
        logger.exception("scheduler: collect_external failed")


def _retrain_weekly():
    """Еженедельное переобучение (subprocess, не блокирует веб).

    Синхронная (subprocess.run) — обёрнута в asyncio.to_thread,
    чтобы не блокировать event loop на 2.5ч.
    """
    try:
        logger.info("scheduler: weekly retrain started")
        result = subprocess.run(
            [sys.executable, "-m", "app.services.trainer.weekly", "--model", "both"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=6 * 3600,
        )
        if result.returncode == 0:
            logger.info("scheduler: weekly retrain done")
        else:
            logger.error("scheduler: weekly retrain failed rc=%s", result.returncode)
            logger.error("scheduler: stderr: %s", result.stderr[-1000:])
    except subprocess.TimeoutExpired:
        logger.error("scheduler: weekly retrain TIMEOUT")
    except Exception:
        logger.exception("scheduler: weekly retrain failed")


async def _retrain_weekly_async():
    """Обёртка для AsyncIOScheduler: subprocess.run в отдельном потоке."""
    import asyncio
    await asyncio.to_thread(_retrain_weekly)


def start_scheduler():
    """Регистрирует задачи и запускает AsyncIOScheduler.

    Вызывается из lifespan (внутри async-контекста FastAPI).
    """
    scheduler.add_job(_collect_klines, "interval", hours=4, id="klines",
                      max_instances=1, coalesce=True)
    scheduler.add_job(_collect_external, "cron", hour=3, minute=0, id="external",
                      max_instances=1, coalesce=True)
    scheduler.add_job(_retrain_weekly_async, "cron", day_of_week="sun", hour=2, minute=0,
                      id="retrain_weekly", max_instances=1, coalesce=True)

    scheduler.start()
    logger.info("scheduler: started (klines 4h, external 03:00, retrain Sun 02:00)")


def stop_scheduler():
    """Останавливает Scheduler (вызывается из lifespan в main.py)."""
    scheduler.shutdown()
    logger.info("scheduler: stopped")