from __future__ import annotations

import pandas as pd

from src.reallocation.rules import has_qualification, meets_cumulative_limits, meets_fdp, meets_rest


def eligible_crew(crew_df: pd.DataFrame, flight_row: pd.Series, constraints: dict) -> pd.DataFrame:
    projected_hours = max(float(flight_row.get("distance_miles", 500)) / 500.0, 1.0)

    def is_eligible(row: pd.Series) -> bool:
        return (
            has_qualification(row["qualifications"], flight_row["aircraft_type"])
            and meets_rest(float(row["rest_hours"]), constraints["min_rest_hours"])
            and meets_fdp(float(row["duty_hours_today"]), projected_hours, constraints["max_fdp_hours"])
            and meets_cumulative_limits(float(row["hours_7d"]), float(row["hours_28d"]), constraints["max_7day_hours"], constraints["max_28day_hours"])
        )

    mask = crew_df.apply(is_eligible, axis=1)
    return crew_df[mask].copy()
