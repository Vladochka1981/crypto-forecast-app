import logging
import math
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.services.orchestrator import orchestrator

from fastapi.responses import StreamingResponse
import io
import csv

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def _sanitize_nan(obj):
    """Рекурсивно заменяет NaN/Inf на None."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={},
    )


@router.get("/predict")
async def predict(horizon: int = 6, model: str = "attention_gru", version: str | None = None):
    try:
        result = await orchestrator.predict(horizon=horizon, model=model, version=version)
        encoded = jsonable_encoder(result)
        sanitized = _sanitize_nan(encoded)
        return JSONResponse(sanitized)
    except Exception as e:
        logger.exception("Ошибка в /predict")
        return JSONResponse({"error": str(e)}, status_code=500)
    
    
@router.get("/dashboard")
async def dashboard():
    try:
        data = await orchestrator.get_dashboard()
        return JSONResponse(_sanitize_nan(jsonable_encoder(data)))
    except Exception as e:
        logger.exception("Ошибка в /dashboard")
        return JSONResponse({"error": str(e)}, status_code=500)
    
@router.get("/metrics")
async def metrics():
    try:
        data = await orchestrator.get_metrics()
        return JSONResponse(_sanitize_nan(jsonable_encoder(data)))
    except Exception as e:
        logger.exception("Ошибка в /metrics")
        return JSONResponse({"error": str(e)}, status_code=500)
    
@router.get("/predict_all")
async def predict_all(model: str = "attention_gru", version: str | None = None):
    try:
        result = await orchestrator.predict_all_horizons(model=model, version=version)
        return JSONResponse(_sanitize_nan(jsonable_encoder(result)))
    except Exception as e:
        logger.exception("Ошибка в /predict_all")
        return JSONResponse({"error": str(e)}, status_code=500)



@router.get("/versions")
def versions():
    """Список версий для UI (из archive/) — только с aci_state.json."""
    import json
    out = {"attention_gru": [], "arimax_lstm": []}
    for mn in out:
        base = Path(settings.models_dir) / mn / "archive"
        if not base.exists():
            continue
        for p in sorted(base.iterdir(), reverse=True):
            if not (p.is_dir() and p.name.startswith(mn + "_v")):
                continue
            
            # Читаем metadata один раз
            meta_path = p / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
                
            # ★ фильтр: только версии с aci_state.json
            if not meta.get("has_aci", False):
                continue
                
            version = p.name[len(mn) + 1:]
            # ★ только deploy-версии (eval — внутренняя, для сравнения)
            if "_eval" in version:
                continue
            metrics = meta.get("metrics", {})
            out[mn].append({"version": version, "metrics": metrics})
            
    return JSONResponse(_sanitize_nan(jsonable_encoder(out)))

@router.get("/comparison")
def comparison(model: str = "attention_gru"):
    """Последний comparison.json (лог сравнения при переобучении)."""
    import json
    deploy_dir = Path(settings.models_dir) / model / "deploy"
    cmp_path = deploy_dir / "comparison.json"
    if not cmp_path.exists():
        return JSONResponse({"error": "comparison.json не найден"}, status_code=404)
    try:
        return JSONResponse(json.loads(cmp_path.read_text(encoding="utf-8")))
    except Exception as e:
        logger.exception("Ошибка в /comparison")
        return JSONResponse({"error": str(e)}, status_code=500)
    
@router.get("/debug_cache")
def debug_cache():
    from app.services.cache_manager import cache_manager
    return JSONResponse({
        "price_status": cache_manager.last_status("price"),
        "price_age": cache_manager.last_age_seconds("price"),
        "memory_keys": list(cache_manager._memory.keys()),
        "memory_timestamps": {k: v.get("timestamp") for k, v in cache_manager._memory.items()},
    })

@router.get("/backtest")
def backtest(model: str = "attention_gru", version: str | None = None):
    """Читает backtest.json для deploy (или archive/vX_deploy для version)."""
    import json
    base = Path(settings.models_dir) / model
    if version:
        bt_path = base / "archive" / f"{model}_{version}" / "backtest.json"
    else:
        bt_path = base / "deploy" / "backtest.json"
    if not bt_path.exists():
        return JSONResponse({"error": "backtest.json не найден"}, status_code=404)
    return JSONResponse(json.loads(bt_path.read_text(encoding="utf-8")))

_retrain_task = None

@router.post("/retrain")
async def trigger_retrain(model: str = "all"):
    global _retrain_task
    if _retrain_task and not _retrain_task.done():
        return JSONResponse({"status": "already_running", "message": "Переобучение уже идёт"}, status_code=409)

    from app.services.scheduler import _retrain_weekly_async
    import asyncio

    async def run_retrain():
        try:
            await _retrain_weekly_async()
            logger.info("Ручное переобучение завершено")
        except Exception as e:
            logger.exception("Ошибка при ручном переобучении: %s", e)

    _retrain_task = asyncio.create_task(run_retrain())
    return JSONResponse({"status": "started", "message": "Переобучение запущено"})

@router.get("/export_predictions")
async def export_predictions():
    """
    Выгружает все предсказания из БД в CSV.
    """
    # Получаем данные из БД (нужно узнать, как они называются)
    # Предположим, у тебя есть метод get_all_predictions()
    # Если нет — используй SQL-запрос через db_repository
    from app.services.database_repository import db_repository

    async with db_repository._connect() as conn:
        rows = await db_repository._fetchall(
            conn, "SELECT * FROM predictions ORDER BY created_at DESC"
        )

    # Формируем CSV в памяти
    output = io.StringIO()
    writer = csv.writer(output, delimiter=';')  # ; — чтобы Excel правильно открывал русский

    # Заголовки
    if rows:
        headers = list(rows[0].keys())
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[h] for h in headers])
    else:
        writer.writerow(["Нет данных"])

    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )