from __future__ import annotations

import numpy as np
import pandas as pd

HUBS = ["ORD", "IAH", "DEN", "LAX", "EWR"]
CARRIERS = ["UA", "AA", "DL", "WN"]
AIRCRAFT = ["A320", "B737", "B777"]


def generate_flights(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    departure_ts = start + pd.to_timedelta(rng.integers(0, 90 * 24, n), unit="h")
    origin = rng.choice(HUBS, n)
    destination = rng.choice(HUBS, n)
    distance = rng.integers(220, 2800, n)
    inbound_delay = rng.integers(0, 120, n)
    turnaround = rng.integers(25, 95, n)
    arr_delay = (0.35 * inbound_delay + np.maximum(0, 45 - turnaround) + rng.normal(0, 15, n)).astype(int)
    df = pd.DataFrame(
        {
            "flight_id": [f"F{i:05d}" for i in range(n)],
            "carrier": rng.choice(CARRIERS, n),
            "origin": origin,
            "destination": destination,
            "aircraft_type": rng.choice(AIRCRAFT, n),
            "departure_ts": departure_ts,
            "distance_miles": distance,
            "inbound_delay_min": inbound_delay,
            "turnaround_min": turnaround,
            "arr_delay_min": arr_delay,
        }
    )
    df["is_delayed"] = (df["arr_delay_min"] >= 15).astype(int)
    return df


def generate_weather(hours: int = 90 * 24, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    hours_index = pd.date_range("2023-01-01", periods=hours, freq="h")
    rows: list[dict[str, object]] = []
    for airport in HUBS:
        base = rng.normal(1.5, 0.8, hours)
        trend = np.clip(base + rng.normal(0, 0.5, hours), 0, 4)
        for ts, sev in zip(hours_index, trend):
            rows.append(
                {
                    "airport": airport,
                    "hour_ts": ts,
                    "weather_severity": float(sev),
                    "wind_kts": int(rng.integers(4, 45)),
                    "visibility_miles": float(np.clip(rng.normal(6, 2), 0.5, 10)),
                    "precip_mm": float(np.clip(rng.normal(2, 3), 0, 25)),
                    "ceiling_ft": int(rng.integers(200, 8000)),
                }
            )
    return pd.DataFrame(rows)


def generate_congestion(hours: int = 90 * 24, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    hours_index = pd.date_range("2023-01-01", periods=hours, freq="h")
    rows: list[dict[str, object]] = []
    for airport in HUBS:
        for ts in hours_index:
            demand = int(rng.integers(35, 90))
            capacity = int(rng.integers(40, 85))
            rows.append(
                {
                    "airport": airport,
                    "hour_ts": ts,
                    "demand": demand,
                    "capacity": capacity,
                    "gdp_flag": int(demand / max(capacity, 1) > 0.95),
                }
            )
    return pd.DataFrame(rows)


def generate_crew(n_per_hub: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3)
    rows: list[dict[str, object]] = []
    for hub in HUBS:
        for idx in range(n_per_hub):
            qual = rng.choice(AIRCRAFT, size=int(rng.integers(1, 4)), replace=False)
            rows.append(
                {
                    "crew_id": f"{hub}-{idx:03d}",
                    "home_hub": hub,
                    "location": rng.choice(HUBS),
                    "qualifications": ",".join(sorted(qual.tolist())),
                    "is_reserve": bool(rng.integers(0, 2)),
                    "duty_hours_today": float(rng.uniform(0, 8.5)),
                    "rest_hours": float(rng.uniform(8, 18)),
                    "hours_7d": float(rng.uniform(20, 65)),
                    "hours_28d": float(rng.uniform(80, 195)),
                    "downstream_risk": float(rng.uniform(0, 1)),
                }
            )
    return pd.DataFrame(rows)
