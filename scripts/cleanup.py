"""Cleanup archive: удаляет старые версии моделей, оставляет N последних.

Правила:
  - deploy/, eval/ — всегда (не трогаем).
  - archive/{model}_vX_deploy/ — оставить N последних.
  - archive/{model}_vX_eval/   — оставить N последних.
  - archive/{model}_prev_*/    — удалить ВСЕ (откат из vX_deploy).
  - archive/{model}_prev_eval_*/ — удалить ВСЕ (откат из vX_eval).

Запуск:
  python scripts/cleanup_archive.py --keep 5 --dry-run
  python scripts/cleanup_archive.py --keep 5 --apply
"""
import argparse
import logging
import shutil
from pathlib import Path

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def cleanup(model_name: str, keep_last: int = 5, dry_run: bool = True):
    base = Path(settings.models_dir) / model_name
    archive = base / "archive"
    if not archive.exists():
        log.warning("[%s] archive не найден: %s", model_name, archive)
        return

    # 1. vX_deploy / vX_eval — оставить keep_last
    for suffix in ("_deploy", "_eval"):
        versions = sorted(
            [d for d in archive.iterdir() if d.is_dir() and d.name.endswith(suffix)],
            key=lambda p: p.name, reverse=True,
        )
        to_delete = versions[keep_last:]
        log.info("[%s] %s: всего %d, оставить %d, удалить %d",
                 model_name, suffix, len(versions),
                 min(keep_last, len(versions)), len(to_delete))
        for d in to_delete:
            if dry_run:
                log.info("  [DRY] delete %s", d.name)
            else:
                shutil.rmtree(d, ignore_errors=True)
                log.info("  deleted %s", d.name)

    # 2. _prev_ / _prev_eval_ — удалить ВСЕ
    for prefix in ("_prev_", "_prev_eval_"):
        prevs = [d for d in archive.iterdir() if d.is_dir() and prefix in d.name]
        log.info("[%s] %s: удалить %d", model_name, prefix, len(prevs))
        for d in prevs:
            if dry_run:
                log.info("  [DRY] delete %s", d.name)
            else:
                shutil.rmtree(d, ignore_errors=True)
                log.info("  deleted %s", d.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=5, help="Сколько последних vX_deploy/vX_eval оставить")
    ap.add_argument("--dry-run", action="store_true", help="Показать, что удалится (по умолчанию)")
    ap.add_argument("--apply", action="store_true", help="Реально удалить")
    args = ap.parse_args()

    dry = not args.apply
    if dry:
        log.info("DRY-RUN: ничего не удаляется. Для реального удаления --apply")

    for m in ("attention_gru", "arimax_lstm"):
        cleanup(m, keep_last=args.keep, dry_run=dry)