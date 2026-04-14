import pandas as pd

from src.reallocation.optimizer import score_crew_candidates, top_k_recommendations
from src.reallocation.roster import eligible_crew


CONSTRAINTS = {
    "min_rest_hours": 10,
    "max_fdp_hours": 9,
    "max_7day_hours": 60,
    "max_28day_hours": 190,
}


def test_reserve_crew_prioritized():
    crew = pd.DataFrame(
        [
            {
                "crew_id": "C1",
                "location": "ORD",
                "qualifications": "A320",
                "is_reserve": True,
                "duty_hours_today": 2.0,
                "rest_hours": 12.0,
                "hours_7d": 30.0,
                "hours_28d": 100.0,
                "downstream_risk": 0.2,
            },
            {
                "crew_id": "C2",
                "location": "ORD",
                "qualifications": "A320",
                "is_reserve": False,
                "duty_hours_today": 2.0,
                "rest_hours": 12.0,
                "hours_7d": 30.0,
                "hours_28d": 100.0,
                "downstream_risk": 0.2,
            },
        ]
    )
    flight = pd.Series({"origin": "ORD", "aircraft_type": "A320", "distance_miles": 500})
    scored = score_crew_candidates(
        crew,
        flight,
        {
            "reserve_bonus": -15,
            "deadhead_cost_per_hub_hop": 25,
            "downstream_disruption": 35,
            "duty_cushion_penalty": 20,
        },
    )
    top = top_k_recommendations(scored, k=1)
    assert top.iloc[0]["crew_id"] == "C1"


def test_duty_limit_violation_blocked():
    crew = pd.DataFrame(
        [
            {
                "crew_id": "C1",
                "location": "ORD",
                "qualifications": "A320",
                "is_reserve": True,
                "duty_hours_today": 8.5,
                "rest_hours": 12.0,
                "hours_7d": 40.0,
                "hours_28d": 120.0,
                "downstream_risk": 0.5,
            }
        ]
    )
    flight = pd.Series({"origin": "ORD", "aircraft_type": "A320", "distance_miles": 800})
    eligible = eligible_crew(crew, flight, CONSTRAINTS)
    assert eligible.empty


def test_qualification_mismatch_blocked():
    crew = pd.DataFrame(
        [
            {
                "crew_id": "C1",
                "location": "ORD",
                "qualifications": "B737",
                "is_reserve": True,
                "duty_hours_today": 2.0,
                "rest_hours": 12.0,
                "hours_7d": 40.0,
                "hours_28d": 120.0,
                "downstream_risk": 0.5,
            }
        ]
    )
    flight = pd.Series({"origin": "ORD", "aircraft_type": "A320", "distance_miles": 500})
    eligible = eligible_crew(crew, flight, CONSTRAINTS)
    assert eligible.empty
