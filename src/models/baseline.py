from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

DEFAULT_CLASSIFICATION_THRESHOLD = 0.5


@dataclass
class BaselineResult:
    name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


def evaluate_baselines(X_train, y_train, X_test, y_test, categorical, numerical):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numerical),
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            max_depth=4,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )

    results: list[BaselineResult] = []
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= DEFAULT_CLASSIFICATION_THRESHOLD).astype(int)
        results.append(
            BaselineResult(
                name=name,
                precision=float(precision_score(y_test, preds, zero_division=0)),
                recall=float(recall_score(y_test, preds, zero_division=0)),
                f1=float(f1_score(y_test, preds, zero_division=0)),
                roc_auc=float(roc_auc_score(y_test, probs)),
                pr_auc=float(average_precision_score(y_test, probs)),
            )
        )
    return results
