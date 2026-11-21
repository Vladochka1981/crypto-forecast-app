"""
Менеджер кэширования.

Двухуровневый кэш (память + диск) с TTL и fallback:
- если данные в памяти и не просрочены — отдаём их;
- если просрочены — пытаемся обновить (вызывающий код передаёт loader);
- если обновление не удалось (API недоступен) — отдаём последнюю
  сохранённую копию с диска и помечаем её как stale.
"""
import json
import time
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Двухуровневый кэш с TTL и fallback на диск."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_seconds: Optional[int] = None):
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        self.ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, dict[str, Any]] = {}
        # для дашборда: статус последнего обращения к каждому ключу
        self._last_status: dict[str, str] = {}   # key -> "live"|"cache"|"stale"
        self._last_ts: dict[str, float] = {}      # key -> unix ts записи

    # ---------- внутренние helpers ----------

    def _disk_path(self, key: str) -> Path:
        """Безопасное имя файла для ключа."""
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.cache_dir / f"{safe}.json"

    def _is_fresh(self, entry: dict[str, Any], ttl: Optional[int] = None) -> bool:
        effective = ttl if ttl is not None else self.ttl
        return (time.time() - entry["timestamp"]) < effective

    def _read_from_disk(self, key: str) -> Optional[dict[str, Any]]:
        path = self._disk_path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Не удалось прочитать кэш с диска {path}: {e}")
            return None

    def _write_to_disk(self, key: str, entry: dict[str, Any]) -> None:
        path = self._disk_path(key)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"Не удалось записать кэш на диск {path}: {e}")

    # ---------- синхронный API ----------

    def get(self, key: str) -> Optional[Any]:
        """Вернуть значение из кэша без обновления."""
        entry = self._memory.get(key)
        if entry is None:
            entry = self._read_from_disk(key)
            if entry is not None:
                self._memory[key] = entry
        if entry is None:
            return None
        return entry["value"]

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], Any],
        force_refresh: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> tuple[Any, bool]:
        """Синхронная версия get_or_load.

        Возвращает (value, is_stale):
          - is_stale=False — данные свежие (из кэша или только что загружены);
          - is_stale=True  — loader упал, отдали последнюю копию с диска.
        Статус последнего обращения сохраняется в _last_status[key]
        ("live" | "cache" | "stale") — для /dashboard.
        """
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.ttl

        entry = self._memory.get(key)
        if entry is None:
            entry = self._read_from_disk(key)
            if entry is not None:
                self._memory[key] = entry

        if entry is not None and not force_refresh:
            age = time.time() - entry["timestamp"]
            if age < effective_ttl:
                self._last_status[key] = "cache"
                self._last_ts[key] = entry["timestamp"]
                return entry["value"], False

        try:
            value = loader()
            new_entry = {"value": value, "timestamp": time.time()}
            self._memory[key] = new_entry
            self._write_to_disk(key, new_entry)
            self._last_status[key] = "live"
            self._last_ts[key] = new_entry["timestamp"]
            return value, False
        except Exception as e:
            logger.warning(f"Ошибка загрузки для ключа '{key}': {e}. Пробуем fallback.")
            if entry is not None:
                self._last_status[key] = "stale"
                self._last_ts[key] = entry["timestamp"]
                return entry["value"], True
            raise

    # ---------- асинхронный API ----------

    async def get_or_load_async(
        self,
        key: str,
        loader: Callable[[], Awaitable[Any]],
        force_refresh: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> tuple[Any, bool]:
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.ttl

        entry = self._memory.get(key)
        if entry is None:
            entry = self._read_from_disk(key)
            if entry is not None:
                self._memory[key] = entry

        if entry is not None and not force_refresh:
            age = time.time() - entry["timestamp"]
            if age < effective_ttl:
                self._last_status[key] = "cache"
                self._last_ts[key] = entry["timestamp"]
                return entry["value"], False

        try:
            value = await loader()
            new_entry = {"value": value, "timestamp": time.time()}
            self._memory[key] = new_entry
            self._write_to_disk(key, new_entry)
            self._last_status[key] = "live"
            self._last_ts[key] = new_entry["timestamp"]
            return value, False
        except Exception as e:
            logger.warning(f"Ошибка загрузки для ключа '{key}': {e}. Пробуем fallback.")
            if entry is not None:
                self._last_status[key] = "stale"
                self._last_ts[key] = entry["timestamp"]
                return entry["value"], True
            raise

    # ---------- статус для дашборда ----------

    def last_status(self, key: str) -> str:
        """'live' | 'cache' | 'stale' | 'unknown' — по последнему обращению к ключу."""
        return self._last_status.get(key, "unknown")

    def last_age_seconds(self, key: str) -> Optional[float]:
        """Сколько секунд прошло с момента записи значения в кэш (None — ключ не трогали)."""
        ts = self._last_ts.get(key)
        if ts is None:
            return None
        return max(0.0, time.time() - ts)

    # ---------- управление ----------

    def invalidate(self, key: str) -> None:
        """Удалить ключ из памяти и с диска."""
        self._memory.pop(key, None)
        self._last_status.pop(key, None)
        self._last_ts.pop(key, None)
        path = self._disk_path(key)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Не удалось удалить {path}: {e}")

    def clear(self) -> None:
        """Полностью очистить кэш (память + диск)."""
        self._memory.clear()
        self._last_status.clear()
        self._last_ts.clear()
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
            except Exception:
                pass


# Глобальный экземпляр
cache_manager = CacheManager()