import pandas as pd
from app.services.feature_engineer import feature_engineer
from app.services.trainer.base import MAX_H, LOOKBACK
from app.services.trainer.evaluator import evaluate
from app.services.trainer.splits import split_by_bars
from pathlib import Path

full = feature_engineer.build_features(date_from="2020-01-01", validate=True)
feature_cols = feature_engineer.feature_cols
eval_split = split_by_bars(full, test_bars=720, val_bars=200)
test_with_ctx = pd.concat([
    eval_split["train"].iloc[-(MAX_H + LOOKBACK):],
    eval_split["test"],
], ignore_index=True)

res = evaluate("attention_gru", Path("data/models/attention_gru/eval"), test_with_ctx, feature_cols)
preds = res["predictions"]

# Смотрим только h=1
h1 = preds[preds["horizon"] == 1].sort_values("date")
print("Первые 10 строк h=1:")
print(h1[["date", "anchor", "actual", "pred"]].head(10))
print("\nПоследние 10 строк h=1:")
print(h1[["date", "anchor", "actual", "pred"]].tail(10))
print("\nДубликаты дат:", h1["date"].duplicated().sum())