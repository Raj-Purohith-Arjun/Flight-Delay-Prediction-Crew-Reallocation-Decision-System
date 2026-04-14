from __future__ import annotations


QUALIFICATION_EQUIVALENCE = {
    "A320": {"A320"},
    "B737": {"B737"},
    "B777": {"B777"},
}


def has_qualification(crew_quals: str, aircraft_type: str) -> bool:
    quals = set(filter(None, (q.strip() for q in crew_quals.split(","))))
    required = QUALIFICATION_EQUIVALENCE.get(aircraft_type, {aircraft_type})
    return bool(quals.intersection(required))


def meets_rest(rest_hours: float, min_rest_hours: float) -> bool:
    return rest_hours >= min_rest_hours


def meets_fdp(duty_hours_today: float, projected_flight_hours: float, max_fdp_hours: float) -> bool:
    return duty_hours_today + projected_flight_hours <= max_fdp_hours


def meets_cumulative_limits(hours_7d: float, hours_28d: float, max_7day: float, max_28day: float) -> bool:
    return hours_7d <= max_7day and hours_28d <= max_28day
