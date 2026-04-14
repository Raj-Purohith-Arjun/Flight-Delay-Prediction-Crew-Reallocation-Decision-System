from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "flight_id",
    "carrier",
    "origin",
    "destination",
    "aircraft_type",
    "departure_ts",
    "is_delayed",
}


def validate_flights(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["departure_ts"].isna().any():
        raise ValueError("departure_ts cannot contain nulls")
