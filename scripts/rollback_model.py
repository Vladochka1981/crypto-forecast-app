"""Откат модели на предыдущую версию из archive/vX_deploy/ по БД.

Запуск:
  python scripts/rollback_model.py attention_gru           # предыдущая (не is_active)
  python scripts/rollback_model.py attention_gru v20261114_1643  # конкретная
"""
import argparse
import logging
import shutil
import sqlite3
from pathlib import Path

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(model_name: str, target_version: str | None = None):
    db_path = settings.db_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if target_version:
        row = conn.execute(
            "SELECT version FROM model_versions WHERE model_name=? AND version=?",
            (model_name, target_version),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT version FROM model_versions WHERE model_name=? AND is_active=0 "
            "ORDER BY trained_at DESC LIMIT 1",
            (model_name,),
        ).fetchone()

    if not row:
        log.error("Нет версии для отката: %s / %s", model_name, target_version)
        return

    version = row["version"]
    src = Path(settings.models_dir) / model_name / "archive" / f"{model_name}_{version}_deploy"
    dst = Path(settings.models_dir) / model_name / "deploy"
    if not src.exists():
        log.error("Нет папки: %s", src)
        return

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    conn.execute("UPDATE model_versions SET is_active=0 WHERE model_name=?", (model_name,))
    conn.execute("UPDATE model_versions SET is_active=1 WHERE model_name=? AND version=?", (model_name, version))
    conn.commit()
    conn.close()

    log.info("[%s] откат на %s. Перезапусти uvicorn.", model_name, version)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["attention_gru", "arimax_lstm"])
    ap.add_argument("version", nargs="?", default=None)
    args = ap.parse_args()
    main(args.model, args.version)