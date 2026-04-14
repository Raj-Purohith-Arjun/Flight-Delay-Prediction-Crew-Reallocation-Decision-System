from __future__ import annotations

import pandas as pd


HUB_HOP_COST = {
    ("ORD", "ORD"): 0,
    ("IAH", "IAH"): 0,
    ("DEN", "DEN"): 0,
    ("LAX", "LAX"): 0,
    ("EWR", "EWR"): 0,
}


def _hop_distance(origin: str, location: str) -> int:
    if origin == location:
        return 0
    return 1


def score_crew_candidates(candidates: pd.DataFrame, flight_row: pd.Series, weights: dict) -> pd.DataFrame:
    data = candidates.copy()
    origin = flight_row["origin"]
    data["deadhead_hops"] = data["location"].map(lambda loc: _hop_distance(origin, loc))
    data["duty_cushion"] = (9 - data["duty_hours_today"]).clip(lower=0)
    data["cost_score"] = (
        data["is_reserve"].astype(int) * weights["reserve_bonus"]
        + data["deadhead_hops"] * weights["deadhead_cost_per_hub_hop"]
        + data["downstream_risk"] * weights["downstream_disruption"]
        + (1 / data["duty_cushion"].replace(0, 0.1)) * weights["duty_cushion_penalty"]
    )
    data["readiness_minutes"] = data["deadhead_hops"] * 75 + data["rest_hours"].rsub(10).clip(lower=0) * 60
    data["deadhead_required"] = data["deadhead_hops"] > 0
    return data.sort_values("cost_score").reset_index(drop=True)


def top_k_recommendations(scored: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    cols = ["crew_id", "cost_score", "readiness_minutes", "deadhead_required", "is_reserve", "location"]
    return scored[cols].head(k).copy()
