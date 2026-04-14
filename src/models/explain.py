from __future__ import annotations

import pandas as pd


def simple_feature_explanations(model, sample: pd.DataFrame, top_n: int = 3) -> list[str]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    ordered = sorted(zip(sample.columns, importances), key=lambda x: x[1], reverse=True)[:top_n]
    return [f"{name} contributed to elevated risk" for name, _ in ordered]
