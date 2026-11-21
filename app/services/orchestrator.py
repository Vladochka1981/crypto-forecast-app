"""Оркестратор прогнозов: выбор модели, сборка ответа, запись в БД."""
import json
import logging

from starlette.concurrency import run_in_threadpool

from app.services.database_repository import db_repository
from app.services.feature_engineer import feature_engineer
from app.services.predictor import model_predictor
from app.services.cache_manager import cache_manager
from app.services.trainer.base import HORIZONS
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _fng_label(v: int | None) -> str:
    if v is None:
        return "—"
    if v <= 24:
        return "Extreme Fear"
    if v <= 44:
        return "Fear"
    if v <= 55:
        return "Neutral"
    if v <= 75:
        return "Greed"
    return "Extreme Greed"


class Orchestrator:
    """Координатор: выбор модели, инференс, запись в БД, дашборд."""

    def __init__(self) -> None:
        pass

    def _resolve_model(self, horizon: int, model: str) -> str:
        if model != "auto":
            return model
        from app.core.config import settings
        selector = settings.model_selector or {}
        resolved = selector.get(horizon) or selector.get(str(horizon))
        resolved = resolved or settings.default_model
        return resolved

    def _load_backtest(self, model: str, version: str | None = None) -> dict | None:
        """Загружает backtest.json для модели/версии (один раз, для переиспользования)."""
        from pathlib import Path
        from app.core.config import settings

        base = Path(settings.models_dir) / model
        if version:
            bt_path = base / "archive" / f"{model}_{version}" / "backtest.json"
        else:
            bt_path = base / "deploy" / "backtest.json"
        if not bt_path.exists():
            return None
        try:
            return json.loads(bt_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("_load_backtest: не удалось прочитать %s", bt_path)
            return None

    def _attach_signal(
        self,
        prediction: dict,
        model: str,
        version: str | None = None,
        bt: dict | None = None,
    ) -> None:
        """Добавляет trading signal (BUY/SELL/HOLD) в prediction."""
        prediction.setdefault("signal", "HOLD")
        prediction.setdefault("theta_long", None)
        prediction.setdefault("theta_short", None)
        prediction.setdefault("exp_ret", None)

        try:
            if bt is None:
                bt = self._load_backtest(model, version)
            if bt is None:
                return

            horizon = prediction["horizon"]
            h_bt = bt.get("horizons", {}).get(str(horizon), {})
            theta_long = float(h_bt.get("theta_long", 0.0))
            theta_short = float(h_bt.get("theta_short", 0.0))

            anchor = float(prediction["anchor"])
            point = float(prediction["point"])
            if anchor <= 0:
                return
            exp_ret = (point / anchor) - 1.0

            if exp_ret > theta_long:
                signal = "BUY"
            elif exp_ret < -theta_short:
                signal = "SELL"
            else:
                signal = "HOLD"

            prediction["signal"] = signal
            prediction["theta_long"] = theta_long
            prediction["theta_short"] = theta_short
            prediction["exp_ret"] = exp_ret
        except Exception:
            logger.exception("_attach_signal: не удалось применить signal")

    # ------------------------------------------------------------------
    async def predict(
        self, horizon: int, model: str = "attention_gru", version: str | None = None,
    ) -> dict:
        """Прогноз на один горизонт (с кэшем по версии и последней свече)."""
        resolved = self._resolve_model(horizon, model)
        bt = self._load_backtest(resolved, version)

        # 1. Пробуем кэш (для любой версии: None = дефолт)
        last_close = await db_repository.get_last_close()
        if last_close is not None:
            cached = await db_repository.get_cached_predictions(
                resolved, version, last_close["date"],
            )
            for r in cached:
                if r["horizon"] == horizon:
                    logger.info(
                        "cache HIT: %s h=%s version=%s, last_close=%s",
                        resolved, horizon, version, last_close["date"],
                    )
                    prediction = {
                        "model": r["model"],
                        "horizon": r["horizon"],
                        "hours": r["horizon"] * 4,
                        "point": r["point"],
                        "lower": r["lower"],
                        "upper": r["upper"],
                        "anchor": r["anchor"],
                        "q_used": r["q_used"],
                        "alpha_t": r["alpha_t"],
                        "is_stale": bool(r["is_stale"]),
                    }
                    self._attach_signal(prediction, resolved, version, bt=bt)
                    chart_df = feature_engineer.build_features_for_chart(limit=200)
                    return {
                        "prediction": prediction,
                        "chart": chart_df.to_dict(orient="records"),
                        "is_stale": prediction["is_stale"],
                    }

        # 2. Cache MISS — считаем (TF — CPU-bound, уводим в threadpool)
        logger.info("cache MISS: %s h=%s version=%s", resolved, horizon, version)
        prediction = await run_in_threadpool(
            model_predictor.predict, horizon, resolved, version,
        )
        chart_df = feature_engineer.build_features_for_chart(limit=200)

        # ★ Trading signal (для cache MISS)
        self._attach_signal(prediction, resolved, version, bt=bt)

        # 3. Сохраняем в БД (для любой версии)
        try:
            await db_repository.save_prediction(
                model=prediction["model"],
                horizon=prediction["horizon"],
                anchor=prediction["anchor"],
                point=prediction["point"],
                lower=prediction["lower"],
                upper=prediction["upper"],
                is_stale=prediction.get("is_stale", False),
                q_used=prediction.get("q_used"),
                alpha_t=prediction.get("alpha_t"),
                version=version,
            )
        except Exception:
            logger.exception("Не удалось сохранить prediction в БД")

        return {
            "prediction": prediction,
            "chart": chart_df.to_dict(orient="records"),
            "is_stale": prediction.get("is_stale", False),
        }

    # ------------------------------------------------------------------
    async def predict_all_horizons(
        self, model: str = "attention_gru", version: str | None = None,
    ) -> dict:
        """5 горизонтов для модели (с кэшем для любой версии) + chart."""
        last_close = await db_repository.get_last_close()
        if last_close is None:
            raise ValueError("Нет данных в klines")

        bt = self._load_backtest(model, version)

        # 1. Пробуем кэш (для любой версии: None = дефолт)
        cached = await db_repository.get_cached_predictions(
            model, version, last_close["date"],
        )
        if len(cached) == len(HORIZONS):
            logger.info("cache HIT: %s version=%s, last_close=%s",
                        model, version, last_close["date"])
            predictions = []
            for r in cached:
                pred = {
                    "model": r["model"],
                    "horizon": r["horizon"],
                    "hours": r["horizon"] * 4,
                    "point": r["point"],
                    "lower": r["lower"],
                    "upper": r["upper"],
                    "anchor": r["anchor"],
                    "q_used": r["q_used"],
                    "alpha_t": r["alpha_t"],
                    "is_stale": bool(r["is_stale"]),
                }
                self._attach_signal(pred, model, version, bt=bt)
                predictions.append(pred)
            chart_df = feature_engineer.build_features_for_chart(limit=200)
            return {
                "predictions": predictions,
                "chart": chart_df.to_dict(orient="records"),
            }

        # 2. Cache MISS — считаем все 5
        logger.info("cache MISS: %s version=%s, last_close=%s",
                    model, version, last_close["date"])
        predictions = []
        for h in HORIZONS:
            pred = await self.predict(h, model, version)
            predictions.append(pred["prediction"])

        chart_df = feature_engineer.build_features_for_chart(limit=200)
        return {
            "predictions": predictions,
            "chart": chart_df.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    async def get_metrics(self) -> dict:
        """Текущие rolling-метрики champion (GRU + ARIMAX) + история runs.

        Async: get_champion/get_history теперь async (registry переписан).
        """
        from pathlib import Path
        from app.core.config import settings
        from app.services.trainer.registry import get_champion, get_history

        out = {"models": {}, "history": [], "last_retrain": None}
        for model_name in ("attention_gru", "arimax_lstm"):
            champion = await get_champion(model_name)
            history = await get_history(model_name, limit=12)

            metrics = {}
            version = None
            trained_at = None
            run_id = None

            if champion:
                version = champion["version"]
                trained_at = champion["trained_at"]
                run_id = str(champion["id"])

                deploy_dir = Path(settings.models_dir) / model_name / "deploy"
                metrics_file = deploy_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                    except Exception:
                        metrics = {}
                if not metrics and champion.get("metrics_snapshot"):
                    try:
                        metrics = json.loads(champion["metrics_snapshot"])
                    except Exception:
                        metrics = {}

            if version:
                out["models"][model_name] = {
                    "run_id": run_id,
                    "version": version,
                    "trained_at": trained_at,
                    "metrics": metrics,
                }
            else:
                out["models"][model_name] = None

            if history and out["last_retrain"] is None:
                out["last_retrain"] = history[0]

        all_history = []
        for model_name in ("attention_gru", "arimax_lstm"):
            for h in await get_history(model_name, limit=12):
                h["model"] = model_name
                all_history.append(h)
        all_history.sort(key=lambda x: x.get("finished_at") or "", reverse=True)
        out["history"] = all_history[:24]
        return out

    # ------------------------------------------------------------------
    async def _load_fng(self) -> tuple[dict | None, bool]:
        """F&G с кэшем (async loader → await db_repository.get_last_fear_greed)."""
        async def loader():
            return await db_repository.get_last_fear_greed(1)

        return await cache_manager.get_or_load_async(
            "fng", loader, ttl_seconds=3600,
        )

    async def _load_price(self) -> tuple[list[dict], bool]:
        """Последние 2 свечи с кэшем (async loader)."""
        async def loader():
            return await db_repository.get_last_klines(2)

        return await cache_manager.get_or_load_async(
            "price", loader, ttl_seconds=60,
        )

    async def get_dashboard(self) -> dict:
        from datetime import datetime as _dt, timedelta as _td

        # --- 1. Fear & Greed ---
        fng, fng_stale = await self._load_fng()
        fng_value = int(fng["fear_greedy_value"]) if fng else None

        # --- 2. BTC price + Δ ---
        klines, price_stale = await self._load_price()
        if len(klines) >= 2:
            prev_close = float(klines[-2]["close"])
            last_close = float(klines[-1]["close"])
            delta_abs = last_close - prev_close
            delta_pct = (delta_abs / prev_close * 100.0) if prev_close else 0.0
            last_date = klines[-1]["date"]
            direction = "up" if delta_abs > 0 else ("down" if delta_abs < 0 else "flat")
        else:
            last_close = prev_close = delta_abs = delta_pct = None
            last_date = None
            direction = "flat"

        # --- 3. Onchain / Macro (свежесть) ---
        now = _dt.utcnow()
        onchain_last = await db_repository.get_onchain_last_date()
        macro_last = await db_repository.get_macro_last_date()

        def _freshness_status(last_date, ttl_hours: float) -> str:
            if last_date is None:
                return "—"
            try:
                dt = last_date
                if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                age = now - dt
                if age < _td(hours=ttl_hours):
                    return "live"
                if age < _td(hours=ttl_hours * 2):
                    return "cache"
                return "stale"
            except Exception:
                return "—"

        onchain_status = _freshness_status(onchain_last, ttl_hours=24)
        macro_status = _freshness_status(macro_last, ttl_hours=24)

        # --- 4. Последнее переобучение ---
        from app.services.trainer.registry import get_champion, get_history
        last_retrain = None
        trained_dates = []
        for mn in ("attention_gru", "arimax_lstm"):
            c = await get_champion(mn)
            if c and c.get("trained_at"):
                trained_dates.append(c["trained_at"])
        if trained_dates:
            last_retrain = max(trained_dates)

        # --- 5. Warnings ---
        warnings: list[str] = []
        now_utc = _dt.utcnow()

        if price_stale:
            last_str = ""
            if last_date:
                try:
                    dt = _dt.fromisoformat(str(last_date).replace("Z", "+00:00"))
                    last_str = f" от {dt.strftime('%H:%M %d.%m')}"
                except Exception:
                    pass
            warnings.append(
                f"Binance недоступен, используются кэшированные данные{last_str}"
            )

        if last_retrain:
            try:
                lr = _dt.fromisoformat(str(last_retrain).replace("Z", "+00:00"))
                if lr.tzinfo is not None:
                    lr = lr.replace(tzinfo=None)
                age_days = (now_utc - lr).days
                if age_days > 14:
                    warnings.append(
                        f"Модель не переобучалась больше 2 недель "
                        f"(последнее: {lr.strftime('%d.%m.%Y')})"
                    )
            except Exception:
                pass

        # 5d. Деградация: MAE h=1 растёт на >=10% за последние 3+ переобучения
        try:
            for mn in ("attention_gru", "arimax_lstm"):
                history = await get_history(mn, limit=5)
                if len(history) < 3:
                    continue
                recent = [h["metrics"] for h in history[:3] if h.get("metrics")]
                if len(recent) < 3:
                    continue
                maes = [r.get("h1", {}).get("mae") for r in recent
                        if r.get("h1", {}).get("mae")]
                if len(maes) < 3:
                    continue
                oldest, newest = maes[-1], maes[0]
                if oldest > 0 and newest / oldest > 1.10:
                    warnings.append(
                        f"⚠️ Деградация {mn}: MAE h=1 вырос с {oldest:.0f} "
                        f"до {newest:.0f} за последние {len(maes)} переобучений"
                    )
        except Exception:
            pass

        # 5c. Высокая неопределённость (только для свежих прогнозов, < 4ч)
        try:
            recent = await db_repository.get_recent_predictions(limit=20)
            now_utc = _dt.utcnow()
            for _, r in recent.iterrows():
                try:
                    created = _dt.fromisoformat(str(r["created_at"]).replace("Z", "+00:00"))
                    if created.tzinfo is not None:
                        created = created.replace(tzinfo=None)
                    age_hours = (now_utc - created).total_seconds() / 3600
                    if age_hours > 4:
                        continue
                except Exception:
                    continue
                point = float(r["point"]); lo = float(r["lower"]); hi = float(r["upper"])
                if point and (hi - lo) / point > 0.20:
                    warnings.append(
                        f"Высокая неопределённость прогноза: {r['model']} h={int(r['horizon'])} "
                        f"(интервал ±{(hi - lo) / 2:.0f}$)"
                    )
                    break
        except Exception:
            pass

        return {
            "fear_greed": {
                "value": fng_value,
                "label": _fng_label(fng_value),
                "date": fng["date"] if fng else None,
            },
            "price": {
                "last": last_close,
                "prev": prev_close,
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
                "direction": direction,
                "date": last_date,
            },
            "data_status": {
                "fng": {
                    "status": cache_manager.last_status("fng"),
                    "age_seconds": cache_manager.last_age_seconds("fng"),
                },
                "price": {
                    "status": cache_manager.last_status("price"),
                    "age_seconds": cache_manager.last_age_seconds("price"),
                },
                "onchain": {
                    "status": onchain_status,
                    "last_date": str(onchain_last) if onchain_last is not None else None,
                },
                "macro": {
                    "status": macro_status,
                    "last_date": str(macro_last) if macro_last is not None else None,
                },
            },
            "last_retrain": last_retrain,
            "warnings": warnings,
        }


orchestrator = Orchestrator()