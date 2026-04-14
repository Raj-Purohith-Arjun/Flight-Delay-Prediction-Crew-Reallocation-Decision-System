from __future__ import annotations

import numpy as np
import pandas as pd


def run_delay_propagation(base_delay_min: float, iterations: int, carry_over_factor: float, noise_low: float, noise_high: float, legs: int = 4, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(iterations):
        delay = float(base_delay_min)
        total = delay
        for _ in range(legs - 1):
            noise = rng.uniform(noise_low, noise_high)
            delay = max(0.0, delay * (carry_over_factor + noise))
            total += delay
        results.append(total)
    return np.array(results)


def summarize_baseline_vs_action(base_delay_min: float, reduced_delay_min: float, sim_cfg: dict) -> dict:
    baseline = run_delay_propagation(base_delay_min=base_delay_min, **sim_cfg)
    intervention = run_delay_propagation(base_delay_min=reduced_delay_min, **sim_cfg)
    reduction_pct = float((baseline.mean() - intervention.mean()) / max(baseline.mean(), 1e-6))
    return {
        "mean_downstream_delay_baseline": float(baseline.mean()),
        "mean_downstream_delay_action": float(intervention.mean()),
        "reduction_pct": reduction_pct,
        "baseline_p5": float(np.percentile(baseline, 5)),
        "baseline_p95": float(np.percentile(baseline, 95)),
        "action_p5": float(np.percentile(intervention, 5)),
        "action_p95": float(np.percentile(intervention, 95)),
    }
