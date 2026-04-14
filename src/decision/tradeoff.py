from __future__ import annotations

import pandas as pd


def combine_tradeoff(risk_score: float, recommendations: pd.DataFrame, simulation_summary: dict) -> pd.DataFrame:
    data = recommendations.copy()
    reliability_gain = simulation_summary["reduction_pct"]
    data["risk_score"] = risk_score
    data["reliability_gain_pct"] = reliability_gain
    data["tradeoff_score"] = (risk_score * 100) - data["cost_score"] + reliability_gain * 100
    return data.sort_values(["tradeoff_score", "cost_score"], ascending=[False, True]).reset_index(drop=True)
