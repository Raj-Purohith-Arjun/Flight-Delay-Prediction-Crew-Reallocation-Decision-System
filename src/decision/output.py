from __future__ import annotations

import pandas as pd


def format_recommendation_output(flight_id: str, ranked: pd.DataFrame) -> dict:
    return {
        "flight_id": flight_id,
        "options": ranked.to_dict(orient="records"),
    }
