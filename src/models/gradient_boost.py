from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

try:
    from lightgbm import LGBMClassifier
except Exception as exc:  # pragma: no cover
    raise ImportError("lightgbm is required for the primary model") from exc


def build_lgbm(params: dict, scale_pos_weight: float) -> LGBMClassifier:
    return LGBMClassifier(**params, scale_pos_weight=scale_pos_weight)


def select_threshold_for_recall(y_true, y_score, min_recall: float) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, 0.0, 0.0
    valid = np.where(recall[:-1] >= min_recall)[0]
    if valid.size == 0:
        best_idx = int(np.argmax(recall[:-1]))
    else:
        best_idx = int(valid[np.argmax(precision[:-1][valid])])
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx])
