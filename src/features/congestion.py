from __future__ import annotations

import pandas as pd


def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["demand_capacity_ratio"] = data["demand"] / data["capacity"].clip(lower=1)
    baseline = data.groupby(["origin", data["hour_ts"].dt.hour])["demand_capacity_ratio"].transform("mean")
    data["ratio_vs_hour_norm"] = data["demand_capacity_ratio"] - baseline
    return data
