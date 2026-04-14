from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.models.gradient_boost import build_lgbm, select_threshold_for_recall


@dataclass
class TrainOutput:
    models: list
    oof_pr_auc: float
    threshold: float
    threshold_precision: float
    threshold_recall: float
    feature_columns: list[str]


def train_lightgbm(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_cols: list[str],
    params: dict,
    target_recall: float,
    category_levels: dict[str, list[str]] | None = None,
) -> TrainOutput:
    data = df.copy()
    X = data[feature_columns].copy()
    y = data["is_delayed"].astype(int)

    for col in categorical_cols:
        if category_levels and col in category_levels:
            X[col] = pd.Categorical(X[col].astype(str), categories=category_levels[col])
        else:
            X[col] = X[col].astype("category")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(data))
    models = []

    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = float(neg / max(pos, 1))

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = build_lgbm(params, scale_pos_weight)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc")
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        models.append(model)

    pr_auc = float(average_precision_score(y, oof))
    threshold, precision, recall = select_threshold_for_recall(y, oof, target_recall)
    return TrainOutput(models=models, oof_pr_auc=pr_auc, threshold=threshold, threshold_precision=precision, threshold_recall=recall, feature_columns=feature_columns)
