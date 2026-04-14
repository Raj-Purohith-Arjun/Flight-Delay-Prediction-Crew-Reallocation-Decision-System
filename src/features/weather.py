from __future__ import annotations

import pandas as pd


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.sort_values(["origin", "hour_ts"]).copy()
    data["weather_worsening_3h"] = (
        data.groupby("origin")["weather_severity"].rolling(window=3, min_periods=1).max().reset_index(level=0, drop=True) - data["weather_severity"]
    )
    data["weather_worsening_3h"] = data["weather_worsening_3h"].clip(lower=0)
    return data
