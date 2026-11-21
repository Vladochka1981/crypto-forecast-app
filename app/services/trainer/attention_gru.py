"""Переобучение Attention-GRU — соответствует Colab-скрипту.

Таргеты ПЕРЕСЧИТЫВАЮТСЯ из close через make_diff_targets (НЕ из CSV!).
  target_h = close(t+h) - close(t) (diff), а не close(t+h) (цена).
  Если взять target_h из CSV (цена) — huber_delta станет ~24837 вместо ~1375,
  conformal 14000–17000 вместо 1343–3371. Это была ошибка, исправлена.

Соответствие Colab:
  - trainval = concat([train, val]) → fit на всём trainval.
  - feature_cols = переданные (51, из feature_engineer) или всё кроме date/target_*.
  - scaler_X/scaler_y — fit на trainval.
  - huber_delta = 0.8 * std(t5).
  - EarlyStopping(monitor='loss', patience=30), ReduceLROnPlateau(patience=10).
  -  - ACI — на тех же residuals (блок 8b), сохраняется в aci_state.json.
  - test — НЕ используется (оценка в evaluator.py).
  - winsor_params — сохраняется в out_dir/winsor_params.json (для Web).
  - Слой импортируется из app.services.attention_layer (без дубля).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.trainer.base import (
    HORIZONS, LOOKBACK, MAX_H, create_mimo_sequences, make_diff_targets,
)

log = logging.getLogger(__name__)

CONFORMAL_VAL_FRAC = 0.15


# ============================================================
# Model factory
# ============================================================
def _build_model(
    lookback: int, n_features: int, huber_delta: float, cfg: dict[str, Any]
):
    """Собирает Attention-GRU (MIMO, 5 голов) — как в Colab."""
    import tensorflow as tf
    from keras.layers import Concatenate, Dense, Dropout, GRU, Input
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.regularizers import l2

    import app.services.attention_layer  # noqa: F401
    from app.services.attention_layer import ArticleStyleTemporalAttention

    inp = Input(shape=(lookback, n_features), name="seq_input")
    x_att, _ = ArticleStyleTemporalAttention(
        attn_units=cfg["attn_units"], name="article_style_attention",
    )(inp)
    x_att = Dropout(0.2, name="attn_dropout")(x_att)
    x = GRU(
        cfg["n_units"], return_sequences=False, name="gru",
        kernel_regularizer=l2(cfg["l2_reg"]) if cfg["l2_reg"] > 0 else None,
    )(x_att)
    x = Dropout(cfg["dropout"], name="gru_dropout")(x)
    heads = []
    for h in HORIZONS:
        hx = Dense(
            32, activation="relu",
            kernel_regularizer=l2(cfg["l2_reg"]) if cfg["l2_reg"] > 0 else None,
            name=f"head_{h}_1",
        )(x)
        hx = Dropout(cfg["dropout"] * 0.5, name=f"head_{h}_drop")(hx)
        heads.append(Dense(1, name=f"head_{h}_2")(hx))
    out = Concatenate(name="multi_head_output")(heads)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(
            learning_rate=cfg["learning_rate"],
            clipnorm=cfg.get("clipnorm", 0),
        ),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )
    return model


# ============================================================
# Helpers
# ============================================================
def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Все колонки, кроме date и target_* — как в Colab (FEATURE_COLS)."""
    return [c for c in df.columns
            if c != "date" and not c.startswith("target_")]


def _sanitize_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """ffill по всем колонкам, кроме date (для audit-скриптов)."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in df.columns:
        if c == "date":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").ffill()
        if df[c].isna().any():
            fv = df[c].dropna().iloc[0] if df[c].notna().any() else 0.0
            df[c] = df[c].fillna(fv)
    return df


# ============================================================
# Training
# ============================================================
def train_attention_gru(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    cfg: dict[str, Any],
    winsor_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Переобучает Attention-GRU на trainval = train + val.

    - trainval = concat([train, val]) → fit на всём trainval.
    - Таргеты ПЕРЕСЧИТЫВАЮТСЯ из close через make_diff_targets (НЕ из CSV!).
      target_h = close(t+h) - close(t) (diff), а не close(t+h) (цена).
    - feature_cols = переданные (51) или всё кроме date/target_*.
    - scaler_X/scaler_y — fit на trainval.
    - huber_delta = 0.8 * std(t5).
    - EarlyStopping(monitor='loss', patience=30), ReduceLROnPlateau(patience=10).
    - conformal — на последних 15% trainval (отдельный predict).
    - test — НЕ используется (оценка в evaluator.py).
    - winsor_params — сохраняется в out_dir/winsor_params.json (для Web).

    Returns:
        dict с метриками.
    """
    import keras
    import tensorflow as tf
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler

    # === 1. trainval = train + val ===
    trainval_df = pd.concat([train_df, val_df], axis=0) \
        .sort_values("date").reset_index(drop=True)
    log.info("trainval: %s (%s..%s), val_rows=%s",
             trainval_df.shape, trainval_df["date"].iloc[0],
             trainval_df["date"].iloc[-1], len(val_df))

    # === 2. Таргеты = diff (close(t+h) - close(t)), НЕ из CSV ===
    close_tv = trainval_df["close"].to_numpy(dtype=np.float64)
    targets_all = make_diff_targets(close_tv, MAX_H)       # (n, 18)
    t5_all = targets_all[:, [h - 1 for h in HORIZONS]]     # (n, 5)
    X_raw = trainval_df[feature_cols].to_numpy(dtype=np.float32)

    mask = np.isfinite(t5_all).all(axis=1) & np.isfinite(X_raw).all(axis=1)
    X_raw, t5 = X_raw[mask], t5_all[mask]
    log.info("trainval (masked): X=%s, y=%s", X_raw.shape, t5.shape)

    # === 3. Scaler на trainval ===
    scaler_X = StandardScaler().fit(X_raw)
    scaler_y = StandardScaler().fit(t5)
    X_s = scaler_X.transform(X_raw).astype(np.float32)
    y_s = scaler_y.transform(t5).astype(np.float32)

    # === 4. HUBER_DELTA = 0.8 * std(t5) ===
    huber_delta = 0.8 * float(np.std(t5))
    log.info("HUBER_DELTA = %.4f (std=%.4f)", huber_delta, float(np.std(t5)))

    # === 5. Последовательности ===
    X_seq, y_seq = create_mimo_sequences(X_s, y_s, LOOKBACK)
    log.info("trainval seq: X=%s, y=%s", X_seq.shape, y_seq.shape)

    # === 6. Модель + fit ===
    keras.utils.set_random_seed(42)
    tf.keras.backend.clear_session()
    model = _build_model(LOOKBACK, len(feature_cols), huber_delta, cfg)

    callbacks = [
        EarlyStopping(
            monitor="loss", patience=30,
            min_delta=cfg["min_delta"],
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=10,
            min_delta=cfg["min_delta"], min_lr=1e-7, verbose=1,
        ),
    ]
    history = model.fit(
        X_seq, y_seq,
        epochs=500,
        batch_size=cfg["batch_size"],
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    epochs = int(len(history.history["loss"]))
    best_epoch = int(np.argmin(history.history["loss"]))
    log.info("fit: epochs=%s, best_epoch=%s, final_loss=%.6f",
             epochs, best_epoch, float(history.history["loss"][-1]))

    # === 7. Сохранение ===
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    model.save(str(out_dir / "attention_gru.keras"))
    joblib.dump(scaler_X, out_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, out_dir / "scaler_y.pkl")
    pd.DataFrame(history.history).to_csv(out_dir / "history.csv", index=False)
    log.info("model + scalers + history → %s", out_dir)

    # === 8. conformal (последние 15% seq) ===
    n_seq = len(X_seq)
    split = int(n_seq * (1 - CONFORMAL_VAL_FRAC))
    X_conf, y_conf_s = X_seq[split:], y_seq[split:]
    pred_conf_s = model.predict(X_conf, verbose=0)
    pred_conf = scaler_y.inverse_transform(pred_conf_s)
    y_conf = scaler_y.inverse_transform(y_conf_s)
    residuals = y_conf - pred_conf
    conformal = {
        str(h): float(np.quantile(np.abs(residuals[:, i]),
                                  1 - settings.aci_alpha_target))
        for i, h in enumerate(HORIZONS)
    }
    with open(out_dir / "conformal_quantiles.json", "w", encoding="utf-8") as f:
        json.dump({
            "q_by_h": conformal,
            "alpha": settings.aci_alpha_target,
            "space": "standardized",
            "n_conformal": int(len(X_conf)),
        }, f, indent=2)
    log.info("conformal: %s", conformal)

    # === 8b. ACI: буфер ошибок гибрида (diff = цена, anchor сокращается) ===
    from app.services.aci import AdaptiveConformal

    aci = AdaptiveConformal(
        alpha_target=settings.aci_alpha_target,
        gamma=settings.aci_gamma,
        lookback=settings.aci_lookback,
        alpha_min=settings.aci_alpha_min,
        alpha_max=settings.aci_alpha_max,
    )
    for i, h in enumerate(HORIZONS):
        errs = np.abs(residuals[:, i])
        errs = errs[np.isfinite(errs)]
        if len(errs) < settings.aci_lookback:
            raise ValueError(
                f"ACI h={h}: ошибок {len(errs)} < lookback {settings.aci_lookback}. "
                f"Увеличь CONFORMAL_VAL_FRAC (сейчас {CONFORMAL_VAL_FRAC}) "
                f"или уменьши ACI_LOOKBACK."
            )
        aci.errors[h] = [float(e) for e in errs[-settings.aci_lookback:]]
    aci.save(out_dir / "aci_state.json")
    log.info("ACI saved: %s (n per h: %s)",
             out_dir / "aci_state.json",
             {h: len(aci.errors[h]) for h in HORIZONS})

    # === 9. model_config.json ===
    cfg_out = {
        "model": "Attention-GRU (article-style), trainval, diff-target",
        "lookback": LOOKBACK,
        "horizons": HORIZONS,
        "max_h": MAX_H,
        "target": "close(t+h) - close(t) (diff, пересчитан из close)",
        "huber_delta": float(huber_delta),
        "n_units": cfg["n_units"],
        "attn_units": cfg["attn_units"],
        "dropout": cfg["dropout"],
        "l2_reg": cfg["l2_reg"],
        "learning_rate": cfg["learning_rate"],
        "batch_size": cfg["batch_size"],
        "patience": 30,
        "patience_reduce": 10,
        "min_delta": cfg["min_delta"],
        "clipnorm": cfg.get("clipnorm", 0),
        "shuffle": False,
        "epochs_max": 500,
        "epochs_actual": epochs,
        "best_epoch": best_epoch,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)

    # === 10. timing ===
    with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump({
            "epochs_actual": epochs,
            "best_epoch": best_epoch,
            "huber_delta": float(huber_delta),
        }, f, indent=2)

    # === 11. winsor_params (для Web) ===
    if winsor_params is not None:
        with open(out_dir / "winsor_params.json", "w", encoding="utf-8") as f:
            json.dump(winsor_params, f, indent=2, ensure_ascii=False)
        log.info("saved: winsor_params.json")

    # === 12. Метрики для metadata ===
    metrics = {
        "n_trainval_seq": int(n_seq),
        "n_conformal": int(len(X_conf)),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "huber_delta": float(huber_delta),
        "conformal": conformal,
        "aci_n_per_h": {h: len(aci.errors[h]) for h in HORIZONS},   # ← NEW
        "final_train_loss": float(history.history["loss"][-1]),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }
    log.info("Attention-GRU обучен: %s", {
        k: v for k, v in metrics.items() if k != "history"
    })
    return metrics


# ============================================================
# CLI
# ============================================================
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Переобучение Attention-GRU на 2 CSV (train/val).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train", default="data/processed_splits/train_arima_clean.csv")
    p.add_argument("--val", default="data/processed_splits/val_lstm_clean.csv")
    p.add_argument("--out-dir", default="data/models/attention_gru")
    p.add_argument("--backup", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--patience-reduce", type=int, default=10)
    p.add_argument("--huber-factor", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_units": 64,
        "attn_units": 64,
        "dropout": 0.25,
        "l2_reg": 1e-3,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "patience": args.patience,
        "min_delta": 1e-5,
        "clipnorm": 0,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    print("=" * 70)
    print("train_attention_gru (CLI)")
    print(f"  train:   {args.train}")
    print(f"  val:     {args.val}")
    print(f"  out-dir: {args.out_dir}")
    print(f"  epochs={args.epochs} patience={args.patience} "
          f"patience_reduce={args.patience_reduce} huber_factor={args.huber_factor}")
    print("=" * 70)

    paths = {"train": args.train, "val": args.val}
    dfs = {}
    for name, path in paths.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"[{name}] нет файла: {p.resolve()}")
        df = pd.read_csv(p)
        if "date" not in df.columns:
            raise ValueError(f"[{name}] нет колонки 'date'")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
        n_targets = sum(c.startswith("target_") for c in df.columns)
        print(f"  [{name:5s}] {p}  shape={df.shape}  "
              f"date={df['date'].min()} .. {df['date'].max()}  "
              f"target_*={n_targets}")
        dfs[name] = df

    train, val = dfs["train"], dfs["val"]

    feature_cols = _get_feature_cols(train)
    print(f"\n  feature_cols: {len(feature_cols)}")

    leak = [c for c in feature_cols if c.startswith("target_")]
    if leak:
        raise RuntimeError(f"target_* в feature_cols: {leak}")
    missing = [f"target_{h}" for h in HORIZONS if f"target_{h}" not in train.columns]
    if missing:
        raise ValueError(f"в CSV нет target_* для горизонтов: {missing}")

    if args.dry_run:
        print("\n[DRY-RUN] Данные прочитаны, обучение пропущено.")
        return 0

    out_dir = Path(args.out_dir)
    if args.backup and out_dir.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = out_dir.parent / f"{out_dir.name}_backup_{ts}"
        shutil.copytree(out_dir, backup_dir)
        print(f"\n[BACKUP] {out_dir} -> {backup_dir}")

    cfg = _cfg_from_args(args)

    t0 = time.time()
    metrics = train_attention_gru(
        train_df=train, val_df=val, feature_cols=feature_cols,
        out_dir=out_dir, cfg=cfg, winsor_params=None,
    )
    print(f"\nВремя обучения: {time.time() - t0:.1f} с")
    print(f"epochs={metrics['epochs']} best_epoch={metrics['best_epoch']} "
          f"huber_delta={metrics['huber_delta']:.2f}")
    print(f"Сохранено: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())