"""Графики для Trainer'а: loss, predictions, metrics, conformal.

Самодостаточный модуль: может работать как библиотека (make_all_plots)
и как CLI (python -m app.services.trainer.plots --model attention_gru).

Строит:
  - loss_curve.png:         train loss/mae по эпохам (из history.csv).
  - predictions.png:        pred vs actual на test (по 5 горизонтам).
  - metrics_by_h.png:       MAE/RMSE/DA по h (bar).
  - conformal_coverage.png: доля точек в conformal-интервале (по h).

Данные:
  - predictions_df — из evaluator.evaluate (колонки date, anchor,
    actual_h1/pred_h1, ..., actual_h18/pred_h18; поддержаны варианты
    y_true_h1/y_pred_h1, actual_1/pred_1).
  - history — из history.csv (GRU); для ARIMAX — None (loss_curve пропускается).
  - q_by_h — из conformal_quantiles.json (GRU/ARIMAX) или aci_state.json
    (если conformal_quantiles.json нет — считаем q = quantile(errors_tail, 1-alpha_t)).

CLI:
    python -m app.services.trainer.plots --model attention_gru
    python -m app.services.trainer.plots --model arimax_lstm
    python -m app.services.trainer.plots --model attention_gru --version v20261115_0300
    python -m app.services.trainer.plots --model attention_gru --out-dir data/models/attention_gru/deploy/plots
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # headless (без plt.show())
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.core.config import settings

log = logging.getLogger(__name__)

HORIZONS = [1, 3, 6, 12, 18]
LOOKBACK = 30
MAX_H = max(HORIZONS)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, h: int, kind: str) -> Optional[str]:
    """Ищет колонку для горизонта h и типа kind ('actual'|'pred').

    Поддерживает варианты:
      - actual_h1 / pred_h1
      - y_true_h1 / y_pred_h1
      - actual_1 / pred_1
    """
    candidates = [
        f"{kind}_h{h}", f"{kind}_{h}",
        f"y_true_h{h}" if kind == "actual" else f"y_pred_h{h}",
        f"y_true_{h}" if kind == "actual" else f"y_pred_{h}",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_series(df: pd.DataFrame, h: int, kind: str) -> Optional[np.ndarray]:
    col = _find_col(df, h, kind)
    if col is None:
        return None
    return df[col].to_numpy(dtype=np.float64)


def _load_history_from_csv(history_csv: Path) -> Optional[dict[str, list[float]]]:
    """Читает history.csv (Keras History) → dict[str, list[float]]."""
    if not history_csv.exists():
        return None
    try:
        df = pd.read_csv(history_csv)
        return {c: df[c].dropna().tolist() for c in df.columns}
    except Exception:
        log.exception("_load_history_from_csv: %s", history_csv)
        return None


def _load_q_by_h(model_dir: Path) -> dict[int, float]:
    """Читает q_by_h из conformal_quantiles.json или aci_state.json.

    Приоритет: conformal_quantiles.json (статические q).
    Если нет — aci_state.json: q = quantile(errors_tail, 1 - alpha_t).
    """
    # 1. conformal_quantiles.json
    cq = model_dir / "conformal_quantiles.json"
    if cq.exists():
        try:
            data = json.loads(cq.read_text(encoding="utf-8"))
            q = data.get("q_by_h", {})
            if q:
                return {int(k): float(v) for k, v in q.items()}
        except Exception:
            log.exception("_load_q_by_h: conformal_quantiles.json")

    # 2. aci_state.json (fallback)
    aci = model_dir / "aci_state.json"
    if aci.exists():
        try:
            data = json.loads(aci.read_text(encoding="utf-8"))
            out = {}
            for h_str, v in data.get("horizons", {}).items():
                h = int(h_str)
                errors = v.get("errors_tail", [])
                alpha_t = float(v.get("alpha_t", 0.05))
                if errors:
                    out[h] = float(np.quantile(errors, 1 - alpha_t))
            return out
        except Exception:
            log.exception("_load_q_by_h: aci_state.json")
    return {}


# ---------------------------------------------------------------------------
# 1. Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(history: Optional[dict[str, list[float]]], out_dir: Path) -> None:
    """Строит loss/mae по эпохам (train). history — dict из model.fit."""
    if not history:
        log.warning("plot_loss_curve: history пуст — пропуск")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "loss" in history:
        axes[0].plot(history["loss"], color="blue", lw=1.5, label="train loss")
        axes[0].set_title("Huber loss (train)", fontsize=12)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

    if "mae" in history:
        axes[1].plot(history["mae"], color="green", lw=1.5, label="train mae")
        axes[1].set_title("MAE (train)", fontsize=12)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("mae")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    plt.tight_layout()
    path = out_dir / "loss_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("plot_loss_curve → %s", path)


# ---------------------------------------------------------------------------
# 2. Predictions (pred vs actual) на test
# ---------------------------------------------------------------------------

def plot_predictions(out_dir: Path, predictions_df: pd.DataFrame) -> None:
    """Строит pred vs actual на test для каждого h (5 subplot)."""
    if predictions_df is None or predictions_df.empty:
        log.warning("plot_predictions: predictions_df пуст — пропуск")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(predictions_df["date"]) if "date" in predictions_df.columns else None
    if dates is None:
        log.warning("plot_predictions: нет колонки 'date' — пропуск")
        return

    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(14, 3 * len(HORIZONS)))
    for k, h in enumerate(HORIZONS):
        ax = axes[k]
        actual = _get_series(predictions_df, h, "actual")
        pred = _get_series(predictions_df, h, "pred")
        if actual is None or pred is None:
            ax.set_title(f"h={h} ({h*4}ч) | нет данных", fontsize=11)
            ax.grid(alpha=0.3)
            continue

        valid = ~np.isnan(actual)
        ax.plot(dates[valid], actual[valid], label="actual", color="blue", lw=1.2)
        ax.plot(dates[valid], pred[valid], label=f"pred h={h}", color="red",
                lw=1.0, alpha=0.85)

        if valid.sum() > 0:
            mae = float(np.mean(np.abs(pred[valid] - actual[valid])))
        else:
            mae = float("nan")
        ax.set_title(f"h={h} ({h*4}ч) | MAE=${mae:.0f}", fontsize=11)
        ax.set_ylabel("price (USD)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "predictions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("plot_predictions → %s", path)


# ---------------------------------------------------------------------------
# 3. Metrics by h (MAE/RMSE/DA)
# ---------------------------------------------------------------------------

def plot_metrics_by_h(out_dir: Path, predictions_df: pd.DataFrame) -> None:
    """Строит MAE/RMSE/DA по h (bar) из predictions_df."""
    if predictions_df is None or predictions_df.empty:
        log.warning("plot_metrics_by_h: predictions_df пуст — пропуск")
        return

    mae_list, rmse_list, da_list = [], [], []
    for h in HORIZONS:
        actual = _get_series(predictions_df, h, "actual")
        pred = _get_series(predictions_df, h, "pred")
        anchor = predictions_df["anchor"].to_numpy(dtype=np.float64) if "anchor" in predictions_df.columns else None

        if actual is None or pred is None:
            mae_list.append(np.nan); rmse_list.append(np.nan); da_list.append(np.nan)
            continue
        valid = ~np.isnan(actual)
        if valid.sum() == 0:
            mae_list.append(np.nan); rmse_list.append(np.nan); da_list.append(np.nan)
            continue
        y_true, y_pred = actual[valid], pred[valid]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        if anchor is not None and len(anchor) == len(actual):
            a = anchor[valid]
            actual_dir = np.sign(y_true - a)
            pred_dir = np.sign(y_pred - a)
            mask = actual_dir != 0
            da = float(np.mean(pred_dir[mask] == actual_dir[mask]) * 100) if mask.sum() else np.nan
        else:
            da = np.nan
        mae_list.append(mae); rmse_list.append(rmse); da_list.append(da)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(HORIZONS))
    labels = [f"h={h}" for h in HORIZONS]

    for ax, vals, title, color in [
        (axes[0], mae_list, "MAE (USD)", "red"),
        (axes[1], rmse_list, "RMSE (USD)", "darkred"),
        (axes[2], da_list, "DA (%)", "blue"),
    ]:
        ax.bar(x, vals, color=color, alpha=0.7, edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    if any(not np.isnan(v) for v in da_list):
        axes[2].axhline(50, color="black", ls="--", lw=0.8, label="random 50%")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    path = out_dir / "metrics_by_h.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("plot_metrics_by_h → %s", path)


# ---------------------------------------------------------------------------
# 4. Conformal coverage
# ---------------------------------------------------------------------------

def plot_conformal_coverage(
    out_dir: Path, predictions_df: pd.DataFrame, q_by_h: dict[int, float],
) -> None:
    """Строит долю точек в conformal-интервале (по h)."""
    if not q_by_h:
        log.warning("plot_conformal_coverage: q_by_h пуст — пропуск")
        return
    if predictions_df is None or predictions_df.empty:
        log.warning("plot_conformal_coverage: predictions_df пуст — пропуск")
        return

    coverage = []
    for h in HORIZONS:
        actual = _get_series(predictions_df, h, "actual")
        pred = _get_series(predictions_df, h, "pred")
        q = float(q_by_h.get(h, 0.0))
        if actual is None or pred is None:
            coverage.append(np.nan); continue
        valid = ~np.isnan(actual)
        if valid.sum() == 0:
            coverage.append(np.nan); continue
        lower = pred[valid] - q
        upper = pred[valid] + q
        in_interval = (actual[valid] >= lower) & (actual[valid] <= upper)
        coverage.append(float(in_interval.mean() * 100))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(HORIZONS))
    ax.bar(x, coverage, color="teal", alpha=0.7, edgecolor="black")
    ax.axhline(95, color="red", ls="--", lw=1, label="target 95% (alpha=0.05)")
    ax.set_xticks(x); ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_ylabel("coverage (%)")
    ax.set_title("Conformal coverage (доля точек в интервале)")
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(coverage):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.legend()
    plt.tight_layout()
    path = out_dir / "conformal_coverage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("plot_conformal_coverage → %s", path)


# ---------------------------------------------------------------------------
# 5. make_all_plots
# ---------------------------------------------------------------------------

def make_all_plots(
    out_dir: Path,
    predictions_df: pd.DataFrame,
    history: Optional[dict[str, list[float]]] = None,
    history_csv: Optional[Path] = None,
    q_by_h: Optional[dict[int, float]] = None,
) -> None:
    """Строит все графики: loss, predictions, metrics, conformal."""
    log.info("make_all_plots → %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if history is None and history_csv is not None:
        history = _load_history_from_csv(history_csv)

    plot_loss_curve(history, out_dir)
    try:
        plot_predictions(out_dir, predictions_df)
    except Exception as e:
        log.exception("plot_predictions: %s", e)
    try:
        plot_metrics_by_h(out_dir, predictions_df)
    except Exception as e:
        log.exception("plot_metrics_by_h: %s", e)
    try:
        plot_conformal_coverage(out_dir, predictions_df, q_by_h or {})
    except Exception as e:
        log.exception("plot_conformal_coverage: %s", e)
    log.info("make_all_plots: готово (%s)", out_dir)


# ---------------------------------------------------------------------------
# 6. CLI (самодостаточный запуск)
# ---------------------------------------------------------------------------

def _build_predictions_for_model(
    model_name: str,
    model_dir: Path,
    n_test: int = 720,
) -> pd.DataFrame:
    """Собирает predictions_df для модели через evaluate (как weekly).

    model_name: 'attention_gru' | 'arimax_lstm'
    model_dir:  data/models/<model>/deploy (или archive/vX_deploy)
    """
    from app.services.feature_engineer import feature_engineer
    from app.services.trainer.splits import split_by_bars
    from app.services.trainer.evaluator import evaluate

    log.info("[%s] build_features (date_from=2020-01-01) …", model_name)
    df_all = feature_engineer.build_features(
        date_from="2020-01-01", date_to=None, validate=False,
    )
    n_total = len(df_all)
    eval_train_end_idx = n_total - n_test - 2000  # EVAL_TEST_BARS + VAL_BARS
    if eval_train_end_idx <= 0:
        raise RuntimeError(f"Слишком мало данных: n_total={n_total}")
    eval_train_end_date = df_all["date"].iloc[eval_train_end_idx]

    log.info("[%s] build_features (fit_range=…%s) …", model_name, eval_train_end_date)
    full_df, _ = feature_engineer.build_features(
        date_from="2020-01-01", date_to=None,
        fit_range=(None, eval_train_end_date),
        return_winsor_params=True,
    )
    feature_cols = feature_engineer.feature_cols

    eval_split = split_by_bars(full_df, test_bars=n_test, val_bars=2000)
    test_with_ctx = pd.concat([
        eval_split["train"].iloc[-(MAX_H + LOOKBACK - 1):],
        eval_split["test"],
    ], ignore_index=True)

    log.info("[%s] evaluate on test (n=%s) …", model_name, len(eval_split["test"]))
    res = evaluate(model_name, model_dir, test_with_ctx, feature_cols, n_test=n_test)
    return res["predictions"]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(
        description="Графики для модели (GRU/ARIMAX) без переобучения.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", choices=["attention_gru", "arimax_lstm"],
                    default="attention_gru")
    ap.add_argument("--version", default=None,
                    help="версия (archive/vX_deploy); None → deploy/")
    ap.add_argument("--out-dir", default=None,
                    help="куда сохранять (None → <model_dir>/plots)")
    ap.add_argument("--n-test", type=int, default=720,
                    help="сколько баров test (n=720 — как в weekly)")
    args = ap.parse_args()

    model_name = args.model
    base = Path(settings.models_dir) / model_name
    if args.version:
        model_dir = base / "archive" / f"{model_name}_{args.version}"
    else:
        model_dir = base / "deploy"

    if not model_dir.exists():
        log.error("model_dir не найден: %s", model_dir)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else (model_dir / "plots")

    log.info("model=%s, dir=%s, out=%s", model_name, model_dir, out_dir)

    predictions_df = _build_predictions_for_model(
        model_name=model_name, model_dir=model_dir, n_test=args.n_test,
    )
    history_csv = model_dir / "history.csv"   # у ARIMAX может не быть
    q_by_h = _load_q_by_h(model_dir)

    make_all_plots(
        out_dir=out_dir,
        predictions_df=predictions_df,
        history_csv=history_csv if history_csv.exists() else None,
        q_by_h=q_by_h,
    )
    log.info("готово: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())