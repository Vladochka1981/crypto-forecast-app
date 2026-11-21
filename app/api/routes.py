import logging
import math
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.services.orchestrator import orchestrator

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
    """Отдаёт главную страницу приложения с формой прогноза."""
    return templates.TemplateResponse(
        request=request, name="index.html", context={},
    )


@router.get("/predict")
async def predict(horizon: int = 6, model: str = "attention_gru", version: str | None = None):
    """
    Возвращает точечный прогноз цены биткойна/USD на заданный горизонт.

    Используется указанная модель (attention_gru или arimax_lstm);
    если версия не передана — берётся последняя активная из таблицы model_versions.
    Прогноз сопровождается адаптивными доверительными интервалами (ACI).

    Возвращает JSON с прогнозом и служебной информацией.
    При внутренней ошибке — 500 с полем "error".
    """    
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
    """
    Возвращает JSON с агрегированными данными для дашборда:
    текущие прогнозы активных моделей, историю цен и сводку метрик.

    Используется главной страницей для отрисовки графиков.
    При внутренней ошибке — 500 с полем "error".
    """
    try:
        data = await orchestrator.get_dashboard()
        return JSONResponse(_sanitize_nan(jsonable_encoder(data)))
    except Exception as e:
        logger.exception("Ошибка в /dashboard")
        return JSONResponse({"error": str(e)}, status_code=500)
    
@router.get("/metrics")
async def metrics():
    """
    Возвращает метрики качества моделей (MAE, RMSE, MAPE и др.)
    по последним доступным версиям.

    Источник — данные обучения и история метрик из data/metrics_history.
    При внутренней ошибке — 500 с полем "error".
    """
    try:
        data = await orchestrator.get_metrics()
        return JSONResponse(_sanitize_nan(jsonable_encoder(data)))
    except Exception as e:
        logger.exception("Ошибка в /metrics")
        return JSONResponse({"error": str(e)}, status_code=500)
    
@router.get("/predict_all")
async def predict_all(model: str = "attention_gru", version: str | None = None):
    """
    Возвращает прогнозы указанной модели сразу по всем горизонтам
    (1, 3, 6, 12, 18 часов) — для сравнения динамики на разных сроках.

    Если версия не передана, используется последняя активная версия модели.
    При внутренней ошибке — 500 с полем "error".
    """
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
    """
    Отладочный эндпоинт: возвращает текущее состояние кэша внешних данных
    (цены с Binance и др.).

    В ответе:
    - price_status — статус последней загрузки цен ("ok" / "error" и т. п.);
    - price_age — возраст данных кэша в секундах;
    - memory_keys — список ключей, хранящихся в кэше в памяти;
    - memory_timestamps — временные метки каждого ключа.
    
    """
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