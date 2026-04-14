from __future__ import annotations

import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.sort_values("departure_ts").copy()
    data["flight_date"] = data["departure_ts"].dt.date

    daily_origin = data.groupby(["origin", "flight_date"])['is_delayed'].mean().rename("daily_delay_rate").reset_index()
    daily_origin["prev_day_origin_delay_rate"] = daily_origin.groupby("origin")["daily_delay_rate"].shift(1)
    daily_origin["rolling_7d_origin_delay_rate"] = (
        daily_origin.groupby("origin")["daily_delay_rate"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    data = data.merge(daily_origin[["origin", "flight_date", "prev_day_origin_delay_rate", "rolling_7d_origin_delay_rate"]], on=["origin", "flight_date"], how="left")

    route_daily = data.groupby(["carrier", "origin", "destination", "flight_date"])['is_delayed'].mean().rename("route_daily_delay").reset_index()
    route_daily["rolling_30d_route_delay_rate"] = (
        route_daily.groupby(["carrier", "origin", "destination"])["route_daily_delay"].rolling(30, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True)
    )
    data = data.merge(
        route_daily[["carrier", "origin", "destination", "flight_date", "rolling_30d_route_delay_rate"]],
        on=["carrier", "origin", "destination", "flight_date"],
        how="left",
    )

    fill_cols = ["prev_day_origin_delay_rate", "rolling_7d_origin_delay_rate", "rolling_30d_route_delay_rate"]
    data[fill_cols] = data[fill_cols].fillna(data["is_delayed"].mean())
    return data
