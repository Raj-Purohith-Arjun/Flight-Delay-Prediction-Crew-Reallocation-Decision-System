from __future__ import annotations

import numpy as np
import pandas as pd


def add_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    hour = data["departure_ts"].dt.hour
    data["departure_bin"] = pd.cut(hour, bins=[-1, 5, 11, 17, 23], labels=["overnight", "morning", "afternoon", "evening"])
    data["day_of_week"] = data["departure_ts"].dt.dayofweek
    data["month"] = data["departure_ts"].dt.month
    data["route_type"] = np.where(data["origin"].eq(data["destination"]), "same_station", "hub_to_hub")
    data["turnaround_pressure"] = np.maximum(0, 45 - data["turnaround_min"])
    return data
