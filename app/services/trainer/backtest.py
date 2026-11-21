"""Backtest (long-short) для оценки торговой стратегии.

Перенос логики из scripts/backtest_eval.py в переиспользуемый модуль.
Вызывается из:
  - weekly.py — при переобучении (сохраняет backtest.json в deploy_tmp/eval_tmp);
  - scripts/run_backtest.py — ручной запуск (читает готовые CSV из eval/).

Схема:
  1. Грид θ_long/θ_short на val (objective = Sharpe).
  2. Бэктест long-short на test с θ из val.
  3. Buy&hold benchmark.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.trainer.base import HORIZONS

log = logging.getLogger(__name__)

# --- config ---
INITIAL_EQUITY = 10000.0
FEE_BPS = 10
SLIPPAGE_BPS = 10
POSITION_SIZE = 0.10
THRESH_LONG_GRID  = np.linspace(0.000, 0.030, 31)
THRESH_SHORT_GRID = np.linspace(0.000, 0.030, 31)
OBJECTIVE = "sharpe"


# --- metrics ---
def max_drawdown(equity_curve):
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    return float(np.min(eq / peaks - 1.0))


def sharpe_ratio(period_returns, ann_factor):
    pr = np.asarray(period_returns, dtype=float)
    pr = pr[np.isfinite(pr)]
    if len(pr) < 2:
        return 0.0
    mu = pr.mean(); sigma = pr.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float((mu / sigma) * np.sqrt(ann_factor))


def profit_factor(trade_returns):
    tr = np.asarray(trade_returns, dtype=float)
    gains = tr[tr > 0].sum()
    losses = np.abs(tr[tr < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summarize_backtest(equity_curve, trade_returns, ann_factor=6 * 365):
    eq = np.asarray(equity_curve, dtype=float)
    period_returns = eq[1:] / eq[:-1] - 1.0 if len(eq) > 1 else np.array([])
    tr = np.asarray(trade_returns, dtype=float)
    return {
        "final_equity": float(eq[-1]) if len(eq) else 0.0,
        "total_return": float(eq[-1] / eq[0] - 1.0) if len(eq) else 0.0,
        "sharpe": sharpe_ratio(period_returns, ann_factor),
        "max_drawdown": max_drawdown(eq),
        "win_rate": float(np.mean(tr > 0)) if len(tr) else 0.0,
        "profit_factor": profit_factor(tr) if len(tr) else 0.0,
        "n_trades": int(len(tr)),
    }


# --- returns ---
def net_return_long(entry, exit, fee_bps, slippage_bps):
    fee = fee_bps / 1e4; slip = slippage_bps / 1e4
    entry_exec = entry * (1 + slip); exit_exec = exit * (1 - slip)
    return float(((1 - fee) ** 2) * (exit_exec / entry_exec) - 1.0)


def net_return_short(entry, exit, fee_bps, slippage_bps):
    fee = fee_bps / 1e4; slip = slippage_bps / 1e4
    entry_exec = entry * (1 - slip); exit_exec = exit * (1 + slip)
    return float(((1 - fee) ** 2) * (entry_exec / exit_exec) - 1.0)


# --- backtest ---
def event_driven_backtest(df_rows, horizon, theta_long, theta_short,
                          initial_equity=INITIAL_EQUITY,
                          fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS,
                          position_size=POSITION_SIZE):
    df = df_rows.copy().sort_values("idx").reset_index(drop=True)
    df["origin_idx"] = df["idx"].astype(int) - horizon + 1
    equity = float(initial_equity)
    equity_curve = [equity]
    trade_returns = []
    trade_log = []
    pending = None
    for i, row in df.iterrows():
        if pending is not None and i == pending["exit_row"]:
            equity *= (1 + pending["net_return"] * position_size)
            pending["status"] = "closed"
            trade_returns.append(pending["net_return"])
            trade_log.append(pending)
            pending = None
        if pending is None:
            entry = float(row["anchor"]); exit_ = float(row["actual"])
            forecast = float(row["pred"])
            exp_ret = forecast / entry - 1.0
            direction = 0
            if np.isfinite(exp_ret):
                if exp_ret > theta_long: direction = 1
                elif exp_ret < -theta_short: direction = -1
            if direction != 0:
                if direction == 1:
                    nr = net_return_long(entry, exit_, fee_bps, slippage_bps)
                else:
                    nr = net_return_short(entry, exit_, fee_bps, slippage_bps)
                pending = {"direction": direction, "entry_row": int(i),
                           "exit_row": int(i + horizon), "net_return": float(nr),
                           "status": "open"}
        equity_curve.append(equity)
    if pending is not None:
        equity *= (1 + pending["net_return"])
        pending["status"] = "closed_forced_end"
        trade_returns.append(pending["net_return"])
        trade_log.append(pending)
        equity_curve[-1] = equity
    eq = np.asarray(equity_curve, dtype=float)
    s = summarize_backtest(eq, trade_returns)
    s["n_long"] = sum(1 for t in trade_log if t["direction"] == 1)
    s["n_short"] = sum(1 for t in trade_log if t["direction"] == -1)
    return {"equity_curve": eq, "trade_returns": trade_returns,
            "trade_log": trade_log, "summary": s}


def buy_and_hold(close_series, start_idx, end_idx,
                 initial_equity=INITIAL_EQUITY, fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS):
    close = np.asarray(close_series, dtype=float)
    start_idx = max(0, start_idx); end_idx = min(len(close) - 1, end_idx)
    if end_idx <= start_idx:
        return {"summary": {"total_return": 0.0, "sharpe": 0.0,
                            "max_drawdown": 0.0, "n_trades": 0}}
    fee = fee_bps / 1e4; slip = slippage_bps / 1e4
    entry = close[start_idx] * (1 + slip); exit_ = close[end_idx] * (1 - slip)
    final_eq = float(initial_equity * ((1 - fee) ** 2) * (exit_ / entry))
    segment = close[start_idx:end_idx + 1]
    bh_curve = initial_equity * ((1 - fee) ** 2) * (segment / entry)
    bh_curve[-1] = final_eq
    trade_returns = np.array([final_eq / initial_equity - 1.0])
    return {"summary": summarize_backtest(bh_curve, trade_returns)}


def optimize_thresholds(df_val, horizon, theta_long_grid, theta_short_grid,
                        objective="sharpe", min_trades=5):
    best = None; best_score = -1e18
    for th_l in theta_long_grid:
        for th_s in theta_short_grid:
            bt = event_driven_backtest(df_val, horizon, float(th_l), float(th_s))
            s = bt["summary"]
            score = s[objective]
            if not np.isfinite(score) or s["n_trades"] < min_trades:
                score = -1e18
            if score > best_score:
                best_score = score
                best = {"theta_long": float(th_l), "theta_short": float(th_s), **s}
    return best


def backtest_from_predictions(
    val_preds: pd.DataFrame,
    test_preds: pd.DataFrame,
    test_close: np.ndarray,
    horizons: list[int] | None = None,
    position_size: float = POSITION_SIZE,
    objective: str = OBJECTIVE,
) -> dict:
    """Бэктест на test с θ из val (out-of-sample).

    Args:
        val_preds: DataFrame из evaluate()["predictions"] на val
            (колонки: horizon, anchor, actual, pred).
        test_preds: DataFrame из evaluate()["predictions"] на test.
        test_close: массив close на test (для buy&hold).
        horizons: список горизонтов (по умолчанию HORIZONS).
        position_size: доля капитала на сделку.
        objective: что максимизировать при подборе θ (sharpe).

    Returns:
        dict: {"computed_at": ..., "horizons": {h: {...}}}
    """
    from datetime import datetime
    horizons = horizons or list(HORIZONS)

    out = {
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "horizons": {},
    }
    for h in horizons:
        val_h = val_preds[val_preds["horizon"] == h].copy()
        val_h["idx"] = np.arange(len(val_h))
        test_h = test_preds[test_preds["horizon"] == h].copy()
        test_h["idx"] = np.arange(len(test_h))

        best = optimize_thresholds(val_h, h, THRESH_LONG_GRID, THRESH_SHORT_GRID,
                                   objective=objective)
        if best is None:
            continue
        th_l, th_s = best["theta_long"], best["theta_short"]

        bt = event_driven_backtest(test_h, h, th_l, th_s, position_size=position_size)
        s = bt["summary"]

        start_idx = int(test_h["idx"].min())
        end_idx = int(test_h["idx"].max())
        bh = buy_and_hold(test_close, start_idx, end_idx)["summary"]

        out["horizons"][str(h)] = {
            "theta_long": th_l, "theta_short": th_s,
            "sharpe": s["sharpe"], "total_return": s["total_return"],
            "max_drawdown": s["max_drawdown"], "win_rate": s["win_rate"],
            "n_trades": s["n_trades"], "n_long": s["n_long"], "n_short": s["n_short"],
            "buy_hold_return": bh["total_return"], "buy_hold_sharpe": bh["sharpe"],
        }
    return out


def backtest_for_model(
    model_name: str,
    version: str | None = None,
    force: bool = False,
) -> dict:
    """Ручной бэктест для модели из уже сохранённых predictions.

    Читает eval/val_predictions.csv + test_predictions.csv (или
    archive/vX_eval/... для конкретной версии). Если backtest.json есть
    и не force — возвращает его. Если predictions нет — пытается
    восстановить через evaluate (медленно). Иначе — ошибка.

    Args:
        model_name: "attention_gru" | "arimax_lstm".
        version: конкретная версия (например, v20261115_1643_eval).
        force: пересчитать, даже если backtest.json есть.

    Returns:
        dict (как backtest_from_predictions).
    """
    from app.core.config import settings

    base = Path(settings.models_dir) / model_name
    eval_dir = (
        base / "archive" / f"{model_name}_{version}"
        if version else base / "eval"
    )
    bt_path = eval_dir / "backtest.json"
    if bt_path.exists() and not force:
        log.info("backtest_for_model: читаем %s (без пересчёта)", bt_path)
        return json.loads(bt_path.read_text(encoding="utf-8"))

    val_path = eval_dir / "val_predictions.csv"
    test_path = eval_dir / "test_predictions.csv"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Нет predictions в {eval_dir}. "
            f"Запусти weekly (он сохранит predictions) или force=True."
        )

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # test_close для buy&hold
    from app.services.feature_engineer import feature_engineer
    from app.services.trainer.splits import split_by_bars
    full_df, _ = feature_engineer.build_features(
        date_from="2020-01-01", date_to=None,
    )
    split = split_by_bars(full_df, test_bars=720, val_bars=2000)
    test_close = split["test"]["close"].to_numpy(dtype=float)

    bt = backtest_from_predictions(val_df, test_df, test_close)
    bt_json = json.dumps(bt, indent=2, ensure_ascii=False, default=float)
    (eval_dir / "backtest.json").write_text(bt_json, encoding="utf-8")
    if not version:
        (base / "deploy" / "backtest.json").write_text(bt_json, encoding="utf-8")
    log.info("backtest_for_model: saved %s", eval_dir / "backtest.json")
    return bt