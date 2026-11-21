"""Registry: версии моделей, metadata.json, model_versions, training_runs.

Схема хранения:
  data/models/<model>/deploy/          ← активная (deploy) версия, читает Web
  data/models/<model>/eval/            ← сестринская (eval) версия, для rolling-метрик
  data/models/<model>/archive/vX_Y/    ← прошлые deploy (для rollback)
  data/models/<model>/champion.json    ← указатель на текущий champion

  БД:
    model_versions   — метаданные (version, file_path, is_active, trained_at)
    training_runs    — аудит (metrics, hyperparameters, dataset_fingerprint, decision)

Файлы (ALLOWED) копируются в archive/vX_Y/, БД — только метаданные.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import pandas as pd
from app.services.trainer.base import dataset_hash

from app.services.database_repository import db_repository

log = logging.getLogger(__name__)

# Какие файлы копировать в archive/vX_Y/ (НЕ backtest_*, НЕ plots_*)
ALLOWED: dict[str, list[str]] = {
    "attention_gru": [
        "attention_gru.keras", "scaler_X.pkl", "scaler_y.pkl",
        "conformal_quantiles.json", "model_config.json",
        "history.csv", "timing.json", "winsor_params.json",
        "aci_state.json",
        "backtest.json",
        "val_predictions.csv", 
        "test_predictions.csv",
    ],
    "arimax_lstm": [
        "arima_result.pkl", "config.json", "val_residuals_by_horizon.csv",
        "timing.json", "winsor_params.json", "aci_state.json", "backtest.json",
        "val_predictions.csv", "test_predictions.csv",
    ],
}
ALLOWED_DIRS: dict[str, list[str]] = {
    "attention_gru": [],
    "arimax_lstm": ["models_per_h"],
}


def compute_fingerprint(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    d = df.sort_values("date").reset_index(drop=True)
    return {
        "period": {"start": str(d["date"].iloc[0]), "end": str(d["date"].iloc[-1])},
        "n_rows": int(len(d)),
        "n_features": len(feature_cols),
        "columns": list(feature_cols),
        "hash_sha256": dataset_hash(df, feature_cols),   # ← зовём base, не свой
    }


def save_version(
    model_name: str,
    version: str,
    src_dir: Path,
    archive_root: Path,
    metrics: dict[str, Any],
    dataset_fp: dict[str, Any],
    n_train: int,
    n_val: int,
    trained_at: str,
) -> Path:
    """Копирует ALLOWED-файлы src_dir → archive_root/{model}_{version}/.

    Пишет metadata.json + metrics.json в папку версии.
    Возвращает путь к папке версии.
    """
    vdir = archive_root / f"{model_name}_{version}"
    vdir.mkdir(parents=True, exist_ok=True)

    for f in ALLOWED[model_name]:
        src = src_dir / f
        if src.exists():
            shutil.copy2(src, vdir / f)
        else:
            log.warning("save_version: нет %s в %s", f, src_dir)

    for d in ALLOWED_DIRS[model_name]:
        src_d = src_dir / d
        if src_d.exists():
            shutil.copytree(src_d, vdir / d, dirs_exist_ok=True)

    (vdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    # ★ Читаем backtest.json (если есть) — чтобы положить в metadata
    backtest_data = None
    bt_path = vdir / "backtest.json"
    if bt_path.exists():
        try:
            backtest_data = json.loads(bt_path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("save_version: не удалось прочитать backtest.json")

    metadata = {
        "model": model_name,
        "version": version,
        "trained_at": trained_at,
        "n_train": n_train,
        "n_val": n_val,
        "dataset_fingerprint": dataset_fp,
        "metrics": metrics,
        "has_aci": (vdir / "aci_state.json").exists(),
        "backtest": backtest_data,     # ← NEW
    }
    
    (vdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    log.info("version saved: %s", vdir)
    return vdir


def promote(
    model_name: str,
    version_dir: Path,
    deploy_dir: Path,
    archive_root: Path,
) -> None:
    """Продвигает version_dir в deploy_dir.

    Старый deploy_dir удаляется (откат — из archive/vX_deploy/ по версии в БД).
    version_dir → deploy_dir, champion.json в archive_root.
    """
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
        log.info("promote: prev deploy removed (откат из archive/vX_deploy)")

    shutil.copytree(version_dir, deploy_dir)
    champion = {
        "model": model_name,
        "version": version_dir.name,
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "prev_backup": None,
    }
    (archive_root / "champion.json").write_text(
        json.dumps(champion, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("promote: %s → %s", version_dir, deploy_dir)



def load_champion_metrics(model_name: str, deploy_dir: Path) -> dict[str, Any] | None:
    """Читает metrics.json из deploy_dir (если есть)."""
    p = deploy_dir / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


async def save_to_db(
    model_name: str,
    version: str,
    file_path: str,
    metrics: dict[str, Any],
    dataset_fp: dict[str, Any],
    hyperparameters: dict[str, Any],
    decision: str,
    n_train: int,
    n_val: int,
    trained_at: str,
    champion_run_id: str | None = None,
) -> int:
    """Пишет model_versions + training_runs в SQLite (async).

    ★ is_active=1 только при decision=="promote" (иначе БД рассинхрон с deploy/).
    """
    now = datetime.now(timezone.utc).isoformat()
    is_active = 1 if decision == "promote" else 0

    async with db_repository._connect() as conn:
        if is_active:
            await db_repository._execute(
                conn,
                "UPDATE model_versions SET is_active = 0 WHERE model_name = ?",
                (model_name,),
            )
        await db_repository._execute(
            conn,
            """INSERT INTO model_versions
               (model_name, version, file_path, is_active, trained_at, created_at,
                metrics_snapshot, dataset_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_name, version, file_path, is_active, trained_at, now,
                json.dumps(metrics, ensure_ascii=False, default=float),
                json.dumps(dataset_fp, ensure_ascii=False, default=float),
            ),
        )
        # Получаем id последней вставки
        row = await db_repository._fetchone(
            conn, "SELECT last_insert_rowid() AS id",
        )
        version_id = int(row["id"])
        await db_repository._execute(
            conn,
            """INSERT INTO training_runs
               (model_version_id, started_at, finished_at, status,
                hyperparameters, metrics, dataset_fingerprint, decision,
                n_train, n_val, champion_run_id)
               VALUES (?, ?, ?, 'success', ?, ?, ?, ?, ?, ?, ?)""",
            (
                version_id, trained_at, now,
                json.dumps(hyperparameters, ensure_ascii=False, default=float),
                json.dumps(metrics, ensure_ascii=False, default=float),
                json.dumps(dataset_fp, ensure_ascii=False, default=float),
                decision, n_train, n_val, champion_run_id,
            ),
        )
    log.info("db: model_versions id=%s, is_active=%s, decision=%s",
             version_id, is_active, decision)
    return version_id


async def get_champion(model_name: str) -> dict[str, Any] | None:
    """Возвращает активную версию из model_versions (is_active=1)."""
    async with db_repository._connect() as conn:
        row = await db_repository._fetchone(
            conn,
            """SELECT id, version, file_path, metrics_snapshot, trained_at
               FROM model_versions WHERE model_name = ? AND is_active = 1
               LIMIT 1""",
            (model_name,),
        )
    return row  # _fetchone уже возвращает dict | None


async def get_history(model_name: str, limit: int = 12) -> list[dict[str, Any]]:
    """История training_runs (для дашборда)."""
    async with db_repository._connect() as conn:
        rows = await db_repository._fetchall(
            conn,
            """SELECT tr.id, tr.finished_at, tr.metrics, tr.decision,
                      mv.version
               FROM training_runs tr
               JOIN model_versions mv ON mv.id = tr.model_version_id
               WHERE mv.model_name = ?
               ORDER BY tr.finished_at DESC LIMIT ?""",
            (model_name, limit),
        )
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
        out.append(d)
    return out