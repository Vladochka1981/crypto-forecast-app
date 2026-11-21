"""Еженедельное переобучение: deploy + eval + evaluate + promote/keep.

Схема:
  deploy_train_end = today - 72h (18 баров)  ← свежая модель для Web
  eval_train_end   = today - 720*4h (720 баров) ← сестринская для rolling-метрик
  test (eval)      = 720 баров после eval_train_end
  val (deploy/eval)= 2000 баров

  Winsorization: fit на eval_train_end (не deploy, чтобы не было утечки в test).

  Решение promote/keep — по rolling-MAE на eval test:
    challenger (свежая) не хуже champion (старая) на majority горизонтов (>=3 из 5)
    с допуском (MARGIN=5%) И хотя бы на одном горизонте строго лучше.

  Бэктест (out-of-sample):
    θ_long/θ_short подбираются на val (eval-модели), метрики на test.
    Сохраняется в deploy_tmp/backtest.json и eval_tmp/backtest.json.
    При promote копируется в deploy/ и eval/ (через promote + явно).
    При keep — только в archive/vX_deploy/ и archive/vX_eval/.

Запуск:
  python -m app.services.trainer.weekly --model both --dry-run
  python -m app.services.trainer.weekly --model attention_gru
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings
from app.services.database_repository import db_repository
from app.services.feature_engineer import feature_engineer
from app.services.trainer.attention_gru import train_attention_gru
from app.services.trainer.base import HORIZONS, LOOKBACK, MAX_H
from app.services.trainer.arimax_lstm import train_arimax_lstm
from app.services.trainer.evaluator import evaluate
from app.services.trainer.registry import (
    compute_fingerprint, get_champion, promote, save_to_db, save_version,
)
from app.services.trainer.splits import split_by_bars
from app.services.trainer.aci_calibration import calibrate_aci
from app.services.trainer.backtest import backtest_from_predictions

log = logging.getLogger(__name__)

EVAL_TEST_BARS = 720
VAL_BARS = 2000
DEPLOY_TEST_BARS = 18   # 72h / 4h
MARGIN = 0.02           # допуск 2% (не хуже)
MAJORITY_H = 3          # >=3 из 5 горизонтов


def _cfg_gru() -> dict[str, Any]:
    return {
        "n_units": 64, "attn_units": 64, "dropout": 0.25,
        "l2_reg": 1e-3, "learning_rate": 5e-4, "batch_size": 32,
        "patience": 30, "min_delta": 1e-5, "clipnorm": 0,
    }


def _compare_metrics(
    challenger: dict[str, dict], champion: dict[str, dict] | None,
) -> dict[str, Any]:
    """Сравнивает rolling-MAE challenger vs champion по горизонтам.

    Логика "свежесть + не хуже":
      - challenger не хуже на >=3/5 горизонтов (с допуском MARGIN)
      - challenger строго лучше хотя бы на 1 горизонте
    """
    if not champion:
        return {
            "improved_h": list(challenger.keys()),
            "n_improved": len(challenger),
            "decision": "promote",
            "reason": "no champion",
            "delta_mae": {},
            "strictly_better": list(challenger.keys()),
        }

    not_worse, strictly_better, delta = [], [], {}
    for h, m in challenger.items():
        c = champion.get(h)
        if not c:
            not_worse.append(h)
            strictly_better.append(h)
            continue
        mae_new, mae_old = m["mae"], c["mae"]
        if mae_old <= 0:
            continue
        pct = (mae_new - mae_old) / mae_old
        delta[h] = pct
        if pct <= MARGIN:
            not_worse.append(h)
        if pct < 0:
            strictly_better.append(h)

    n_not_worse = len(not_worse)
    n_strictly = len(strictly_better)

    if n_not_worse >= MAJORITY_H and n_strictly >= 1:
        decision = "promote"
        reason = f"not_worse={n_not_worse}/5, strictly_better={n_strictly}"
    else:
        decision = "keep"
        reason = (f"not_worse={n_not_worse}/5 (need >={MAJORITY_H}), "
                  f"strictly_better={n_strictly} (need >=1)")

    return {
        "improved_h": not_worse,
        "n_improved": n_not_worse,
        "decision": decision,
        "reason": reason,
        "delta_mae": delta,
        "strictly_better": strictly_better,
    }


async def _run_model(
    model_name: str,
    full_df: pd.DataFrame,
    feature_cols: list[str],
    winsor_params: dict[str, Any],
    dataset_fp: dict[str, Any],
    today: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Один модель: train → calibrate → compare → promote/keep.

    Папки:
      deploy_tmp/ — свежая deploy-модель (train_end = n_total - 18)
      eval_tmp/   — свежая eval-модель (train_end = n_total - 720 - 2000)
      deploy/     — champion (Web)
      eval/       — текущая eval-модель (для сравнения)
    """
    models_root = Path(settings.models_dir) / model_name
    deploy_dir = models_root / "deploy"
    eval_dir = models_root / "eval"
    deploy_tmp = models_root / "deploy_tmp"
    eval_tmp = models_root / "eval_tmp"
    archive_root = models_root / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    n_total = len(full_df)
    deploy_split = split_by_bars(
        full_df, test_bars=0, val_bars=VAL_BARS,
        train_end_bars=n_total - DEPLOY_TEST_BARS,
    )
    eval_split = split_by_bars(full_df, test_bars=EVAL_TEST_BARS, val_bars=VAL_BARS)
    log.info("[%s] deploy: train=%s val=%s; eval: train=%s val=%s test=%s",
             model_name, len(deploy_split["train"]), len(deploy_split["val"]),
             len(eval_split["train"]), len(eval_split["val"]), len(eval_split["test"]))

    if dry_run:
        return {"model": model_name, "dry_run": True}

    # === 1. Очистка временных + deploy-модель (deploy_tmp) ===
    for d in (deploy_tmp, eval_tmp):
        if d.exists():
            shutil.rmtree(d)
    deploy_tmp.mkdir(parents=True, exist_ok=True)
    eval_tmp.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if model_name == "attention_gru":
        train_attention_gru(deploy_split["train"], deploy_split["val"],
                            feature_cols, deploy_tmp, _cfg_gru(),
                            winsor_params=winsor_params)
    else:
        train_arimax_lstm(deploy_split["train"], deploy_split["val"],
                          deploy_tmp, feature_cols, winsor_params=winsor_params)
    deploy_sec = time.time() - t0
    log.info("[%s] deploy обучен за %.0fs", model_name, deploy_sec)

    # === 2. Eval-модель (eval_tmp) ===
    t0 = time.time()
    if model_name == "attention_gru":
        train_attention_gru(eval_split["train"], eval_split["val"],
                            feature_cols, eval_tmp, _cfg_gru(),
                            winsor_params=winsor_params)
    else:
        train_arimax_lstm(eval_split["train"], eval_split["val"],
                          eval_tmp, feature_cols, winsor_params=winsor_params)
    eval_sec = time.time() - t0
    log.info("[%s] eval обучен за %.0fs", model_name, eval_sec)

    # === 3. ACI-калибровка для обеих временных ===
    res_val = None
    for name, split_dir, split_df in (
        ("deploy_tmp", deploy_tmp, deploy_split),
        ("eval_tmp", eval_tmp, eval_split),
    ):
        val_with_ctx = pd.concat([
            split_df["train"].iloc[-(MAX_H + LOOKBACK - 1):],
            split_df["val"],
        ], ignore_index=True)
        t0 = time.time()
        r = evaluate(model_name, split_dir, val_with_ctx, feature_cols, n_test=None)
        calibrate_aci(split_dir, r["predictions"])
        if name == "eval_tmp":
            res_val = r  # ★ для бэктеста (θ из eval val)
        log.info("[%s] ACI calibration (%s): n=%s, %.0fs",
                 model_name, name, r["n"], time.time() - t0)

    # === 4. Сравнение: challenger (eval_tmp) vs champion (eval/) ===
    test_with_ctx = pd.concat([
        eval_split["train"].iloc[-(MAX_H + LOOKBACK - 1):],
        eval_split["test"],
    ], ignore_index=True)

    res_challenger = evaluate(model_name, eval_tmp, test_with_ctx, feature_cols,
                              n_test=EVAL_TEST_BARS)
    log.info("[%s] challenger metrics (n=%s): %s",
             model_name, res_challenger["n"],
             {h: round(m["mae"], 1) for h, m in res_challenger["metrics"].items()})

    # ★ Бэктест (θ из eval val → test). Один результат для deploy и eval.
    try:
        bt_result = backtest_from_predictions(
            val_preds=res_val["predictions"],
            test_preds=res_challenger["predictions"],
            test_close=eval_split["test"]["close"].to_numpy(dtype=float),
        )
        bt_json = json.dumps(bt_result, indent=2, ensure_ascii=False, default=float)
        (eval_tmp / "backtest.json").write_text(bt_json, encoding="utf-8")
        (deploy_tmp / "backtest.json").write_text(bt_json, encoding="utf-8")
        log.info("[%s] backtest done: %s", model_name, {
            h: round(bt_result["horizons"][str(h)]["sharpe"], 2) for h in HORIZONS
        })
    except Exception:
        log.exception("[%s] backtest failed", model_name)

    # ★ Сохраняем predictions (для ручного backtest_eval и аудита)
    try:
        if res_val and "predictions" in res_val:
            res_val["predictions"].to_csv(
                eval_tmp / "val_predictions.csv", index=False,
            )
        if res_challenger and "predictions" in res_challenger:
            res_challenger["predictions"].to_csv(
                eval_tmp / "test_predictions.csv", index=False,
            )
        log.info("[%s] predictions saved: val + test (eval_tmp)", model_name)
    except Exception:
        log.exception("[%s] failed to save predictions", model_name)
        
    # === Оценка champion (текущий deploy) на том же test ===
    champion_metrics = None
    if (eval_dir / "metrics.json").exists():
        res_champion = evaluate(model_name, eval_dir, test_with_ctx, feature_cols,
                                n_test=EVAL_TEST_BARS)
        champion_metrics = res_champion["metrics"]
        log.info("[%s] champion metrics (n=%s): %s",
                 model_name, res_champion["n"],
                 {h: round(m["mae"], 1) for h, m in champion_metrics.items()})

    cmp = _compare_metrics(res_challenger["metrics"], champion_metrics)
    log.info("[%s] compare: decision=%s, reason=%s, delta=%s",
             model_name, cmp["decision"], cmp["reason"],
             {h: round(v * 100, 2) for h, v in cmp["delta_mae"].items()})

    # === 5. save_version для обоих архивов + comparison ===
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
    vdir_deploy = save_version(
        model_name, f"{version}_deploy", deploy_tmp, archive_root,
        metrics=res_challenger["metrics"],
        dataset_fp=dataset_fp,
        n_train=len(deploy_split["train"]),
        n_val=len(deploy_split["val"]),
        trained_at=datetime.now().isoformat(timespec="seconds"),
    )
    vdir_eval = save_version(
        model_name, f"{version}_eval", eval_tmp, archive_root,
        metrics=res_challenger["metrics"],
        dataset_fp=dataset_fp,
        n_train=len(eval_split["train"]),
        n_val=len(eval_split["val"]),
        trained_at=datetime.now().isoformat(timespec="seconds"),
    )

    comparison = {
        "version": version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "decision": cmp["decision"],
        "reason": cmp["reason"],
        "delta_mae": cmp["delta_mae"],
        "improved_h": cmp["improved_h"],
        "strictly_better": cmp["strictly_better"],
        "challenger_metrics": res_challenger["metrics"],
        "champion_metrics": champion_metrics,
    }
    for d in (vdir_deploy, vdir_eval):
        (d / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False, default=float),
            encoding="utf-8",
        )

    # === 6. promote / keep ===
    if cmp["decision"] == "promote":
        promote(model_name, deploy_tmp, deploy_dir, archive_root)
        for f in ("metrics.json", "metadata.json", "comparison.json"):
            src = vdir_deploy / f
            if src.exists():
                (deploy_dir / f).write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8",
                )

        # eval: новый eval_tmp → eval/ (откат из archive/vX_eval)
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        shutil.copytree(eval_tmp, eval_dir)
        for f in ("metrics.json", "metadata.json", "comparison.json"):
            src = vdir_eval / f
            if src.exists():
                (eval_dir / f).write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8",
                )
    # keep: deploy/ и eval/ не трогаем (в архиве уже есть vX_deploy / vX_eval)

    # === 7. БД (is_active=1 только при promote) ===
    champion = await get_champion(model_name)
    await save_to_db(
        model_name=model_name,
        version=version,
        file_path=str(deploy_dir if cmp["decision"] == "promote" else vdir_deploy),
        metrics=res_challenger["metrics"],
        dataset_fp=dataset_fp,
        hyperparameters={"model": model_name, "winsor_params": winsor_params},
        decision=cmp["decision"],
        n_train=len(deploy_split["train"]),
        n_val=len(deploy_split["val"]),
        trained_at=datetime.now().isoformat(timespec="seconds"),
        champion_run_id=str(champion["id"]) if champion else None,
    )

    # === 8. Удаление временных ===
    for tmp in (deploy_tmp, eval_tmp):
        if tmp.exists():
            shutil.rmtree(tmp)
    log.info("[%s] deploy_tmp и eval_tmp удалены", model_name)

    return {
        "model": model_name,
        "version": version,
        "decision": cmp["decision"],
        "reason": cmp["reason"],
        "n_improved": cmp["n_improved"],
        "delta_mae": cmp["delta_mae"],
        "challenger_metrics": res_challenger["metrics"],
        "champion_metrics": champion_metrics,
        "deploy_sec": deploy_sec,
        "eval_sec": eval_sec,
        "total_sec": deploy_sec + eval_sec,
    }


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(
        description="Еженедельное переобучение (deploy + eval + promote/keep).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", choices=["attention_gru", "arimax_lstm", "both"],
                    default="both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.info("weekly start: model=%s, today=%s", args.model, today)

    last_date = await db_repository.get_last_date()
    if last_date is None:
        log.error("Нет данных в klines")
        return 1

    # build_features — sync, но внутри _run_db (asyncio.run).
    # В async main() running loop есть → _run_db уйдёт в ThreadPoolExecutor.
    # Чтобы было чисто — вызываем в отдельном потоке (без running loop внутри).
    df_all = await asyncio.to_thread(
        feature_engineer.build_features,
        date_from="2020-01-01", date_to=None, validate=False,
    )
    n_total = len(df_all)
    eval_train_end_idx = n_total - EVAL_TEST_BARS - VAL_BARS
    if eval_train_end_idx <= 0:
        log.error("Слишком мало данных: n_total=%s", n_total)
        return 1
    eval_train_end_date = df_all["date"].iloc[eval_train_end_idx]

    log.info("winsorization fit на eval_train_end=%s (idx=%s из %s)",
             eval_train_end_date, eval_train_end_idx, n_total)
    full_df, winsor_params = await asyncio.to_thread(
        feature_engineer.build_features,
        date_from="2020-01-01", date_to=None,
        fit_range=(None, eval_train_end_date),
        return_winsor_params=True,
    )
    feature_cols = feature_engineer.feature_cols
    dataset_fp = compute_fingerprint(full_df, feature_cols)
    log.info("dataset fingerprint: n=%s, hash=%s",
             dataset_fp["n_rows"], dataset_fp["hash_sha256"][:12])

    models = ["attention_gru", "arimax_lstm"] if args.model == "both" else [args.model]
    results = []
    for m in models:
        try:
            r = await _run_model(m, full_df, feature_cols, winsor_params, dataset_fp,
                                 today, args.dry_run)
            results.append(r)
            log.info("[%s] DONE: %s", m, {k: v for k, v in r.items()
                                         if k not in ("challenger_metrics", "champion_metrics")})
        except Exception as e:
            log.exception("[%s] FAILED: %s", m, e)
            results.append({"model": m, "error": str(e)})

    out = Path(settings.data_dir) / "metrics_history"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"weekly_{datetime.now().strftime('%Y%m%d_%H%M')}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    log.info("weekly done: %s", [r.get("model") for r in results])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
    