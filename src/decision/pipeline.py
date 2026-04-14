from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from src.data.ingest.synthetic import generate_congestion, generate_crew, generate_flights, generate_weather
from src.data.merge.join import merge_inputs
from src.data.validate.schema import validate_flights
from src.decision.output import format_recommendation_output
from src.decision.tradeoff import combine_tradeoff
from src.features.congestion import add_congestion_features
from src.features.flight import add_flight_features
from src.features.temporal import add_temporal_features
from src.features.weather import add_weather_features
from src.models.baseline import evaluate_baselines
from src.models.evaluate import evaluate_predictions
from src.models.explain import simple_feature_explanations
from src.models.train import train_lightgbm
from src.reallocation.optimizer import score_crew_candidates, top_k_recommendations
from src.reallocation.roster import eligible_crew
from src.reallocation.simulator import summarize_baseline_vs_action
from src.utils.config import load_yaml
from src.utils.io import CONFIG_DIR, DATA_PROCESSED, DATA_SIMULATED, ensure_dirs


ARTIFACT_MODEL = DATA_PROCESSED / "lightgbm_model.pkl"
ARTIFACT_META = DATA_PROCESSED / "model_meta.pkl"
ARTIFACT_FEATURES = DATA_PROCESSED / "features.parquet"


def _prepare_model_input(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    category_levels: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    X = df[feature_columns].copy()
    for col in categorical_columns:
        levels = (category_levels or {}).get(col)
        if levels is not None:
            X[col] = pd.Categorical(X[col].astype(str), categories=levels)
        else:
            X[col] = X[col].astype("category")
    return X


def build_dataset() -> pd.DataFrame:
    ensure_dirs()
    flights = generate_flights()
    weather = generate_weather()
    congestion = generate_congestion()
    merged = merge_inputs(flights, weather, congestion)
    validate_flights(merged)
    featured = add_temporal_features(add_congestion_features(add_weather_features(add_flight_features(merged))))
    featured.to_parquet(ARTIFACT_FEATURES, index=False)
    return featured


def train_models() -> dict:
    cfg = load_yaml(CONFIG_DIR / "model_config.yaml")
    data = pd.read_parquet(ARTIFACT_FEATURES)
    feature_columns = cfg["categorical_features"] + cfg["numerical_features"]
    category_levels = {col: sorted(data[col].astype(str).unique().tolist()) for col in cfg["categorical_features"]}

    train_output = train_lightgbm(
        df=data,
        feature_columns=feature_columns,
        categorical_cols=cfg["categorical_features"],
        params=cfg["lightgbm"],
        target_recall=float(cfg["target_recall"]),
        category_levels=category_levels,
    )

    split = int(len(data) * 0.8)
    baseline = evaluate_baselines(
        data.iloc[:split][feature_columns],
        data.iloc[:split]["is_delayed"],
        data.iloc[split:][feature_columns],
        data.iloc[split:]["is_delayed"],
        cfg["categorical_features"],
        cfg["numerical_features"],
    )

    with ARTIFACT_MODEL.open("wb") as handle:
        pickle.dump(train_output.models, handle)
    with ARTIFACT_META.open("wb") as handle:
        pickle.dump(
            {
                "threshold": train_output.threshold,
                "feature_columns": train_output.feature_columns,
                "oof_pr_auc": train_output.oof_pr_auc,
                "threshold_precision": train_output.threshold_precision,
                "threshold_recall": train_output.threshold_recall,
                "baseline": baseline,
                "category_levels": category_levels,
            },
            handle,
        )

    return {
        "oof_pr_auc": train_output.oof_pr_auc,
        "threshold": train_output.threshold,
        "threshold_precision": train_output.threshold_precision,
        "threshold_recall": train_output.threshold_recall,
    }


def evaluate_model() -> dict:
    with ARTIFACT_MODEL.open("rb") as handle:
        models = pickle.load(handle)
    with ARTIFACT_META.open("rb") as handle:
        meta = pickle.load(handle)

    data = pd.read_parquet(ARTIFACT_FEATURES)
    cfg_model = load_yaml(CONFIG_DIR / "model_config.yaml")
    X = _prepare_model_input(
        data,
        meta["feature_columns"],
        cfg_model["categorical_features"],
        meta.get("category_levels"),
    )

    prob_matrix = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    y_score = prob_matrix.mean(axis=1)
    report = evaluate_predictions(data["is_delayed"], y_score, float(meta["threshold"]))
    explanations = simple_feature_explanations(models[0], X.head(1), top_n=3)
    return {
        "metrics": report.__dict__,
        "threshold": meta["threshold"],
        "explanations_example": explanations,
    }


def run_reallocation_and_simulation(top_n_flights: int = 5) -> list[dict]:
    with ARTIFACT_MODEL.open("rb") as handle:
        models = pickle.load(handle)
    with ARTIFACT_META.open("rb") as handle:
        meta = pickle.load(handle)

    cfg_model = load_yaml(CONFIG_DIR / "model_config.yaml")
    cfg_realloc = load_yaml(CONFIG_DIR / "reallocation_config.yaml")
    crew = generate_crew()
    data = pd.read_parquet(ARTIFACT_FEATURES).sort_values("departure_ts").copy()
    lookahead_end = data["departure_ts"].min() + pd.Timedelta(hours=3)
    window = data[data["departure_ts"] <= lookahead_end].copy()

    X = _prepare_model_input(
        window,
        meta["feature_columns"],
        cfg_model["categorical_features"],
        meta.get("category_levels"),
    )
    risk = np.column_stack([m.predict_proba(X)[:, 1] for m in models]).mean(axis=1)
    window["risk_score"] = risk
    flagged = window[window["risk_score"] >= float(meta["threshold"])].sort_values(["risk_score", "departure_ts"], ascending=[False, True]).head(top_n_flights)

    outputs = []
    for _, flight in flagged.iterrows():
        eligible = eligible_crew(crew, flight, cfg_realloc["constraints"])
        if eligible.empty:
            outputs.append({"flight_id": flight["flight_id"], "escalation": "No eligible crew available"})
            continue
        scored = score_crew_candidates(eligible, flight, cfg_realloc["cost_weights"])
        topk = top_k_recommendations(scored, k=3)

        base_delay = float(max(0.0, flight.get("arr_delay_min", 20)))
        reduced_delay = float(max(0.0, base_delay * 0.82))
        sim_cfg = {
            "iterations": int(cfg_realloc["simulation"]["iterations"]),
            "carry_over_factor": float(cfg_realloc["simulation"]["carry_over_factor"]),
            "noise_low": float(cfg_realloc["simulation"]["noise_low"]),
            "noise_high": float(cfg_realloc["simulation"]["noise_high"]),
        }
        sim_summary = summarize_baseline_vs_action(base_delay, reduced_delay, sim_cfg)
        ranked = combine_tradeoff(float(flight["risk_score"]), topk, sim_summary)
        outputs.append(format_recommendation_output(flight["flight_id"], ranked))

    pd.DataFrame(outputs).to_json(DATA_SIMULATED / "recommendations.json", orient="records", indent=2)
    return outputs
