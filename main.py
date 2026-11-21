"""FastAPI приложение: роуты + статика + Scheduler."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.database_repository import db_repository
from app.services.scheduler import (
    start_scheduler, stop_scheduler,
    _collect_klines, _collect_external,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Сначала инициализируем схему (SQLite; для Postgres — no-op)
    await db_repository._init_schema()

    # 2. Потом — фоновые boot-задачи (данные уже есть куда писать)
    asyncio.create_task(_collect_klines())
    asyncio.create_task(_collect_external())

    # 3. Регулярный scheduler
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()


app = FastAPI(title="Crypto Forecast App", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")