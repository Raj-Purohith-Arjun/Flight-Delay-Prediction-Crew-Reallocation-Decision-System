from __future__ import annotations

import pandas as pd


def merge_inputs(flights: pd.DataFrame, weather: pd.DataFrame, congestion: pd.DataFrame) -> pd.DataFrame:
    data = flights.copy()
    data["hour_ts"] = data["departure_ts"].dt.floor("h")
    data = data.merge(
        weather,
        how="left",
        left_on=["origin", "hour_ts"],
        right_on=["airport", "hour_ts"],
        suffixes=("", "_wx"),
    ).drop(columns=["airport"])
    data = data.merge(
        congestion,
        how="left",
        left_on=["origin", "hour_ts"],
        right_on=["airport", "hour_ts"],
        suffixes=("", "_cong"),
    ).drop(columns=["airport"])
    return data
