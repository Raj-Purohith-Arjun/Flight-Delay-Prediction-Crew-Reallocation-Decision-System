# Flight Delay Prediction & Crew Reallocation Decision System

A data science project that treats flight delays as an **operational decision problem**, not just a prediction task. Instead of stopping at "this flight will probably be late," the system connects delay risk to actionable crew reallocation strategies — the kind of decision an ops control center actually has to make under time pressure.

---

## The Problem

Every airline operations control center (OCC) deals with some version of this: a flight is 90 minutes out from departure, weather is deteriorating at the origin airport, and the inbound aircraft is running 40 minutes behind. The scheduler has to decide — now — whether to hold the crew, pull a reserve, or let it ride and deal with the downstream consequences later.

The tools most OCCs actually use are a combination of experience, tribal knowledge, and spreadsheets. The delay is usually obvious in retrospect. The hard part is knowing early enough to do something about it.

This project builds two connected pieces:

1. **A delay risk classifier** that flags high-risk flights 2–3 hours before departure using weather, congestion, and schedule data — all of which are available well before wheels-up
2. **A crew reallocation engine** that, for any flagged flight, identifies the best eligible crew options while respecting FAA duty hour limits, rest requirements, and downstream pairing obligations

The two pieces feed into a decision pipeline that produces ranked, explainable recommendations an OCC coordinator can act on in minutes.

---

## Why This Framing Matters

Most flight delay prediction work stops at the model. You get an accuracy number and a feature importance chart, and then it's unclear what to do with it. The actual operational challenge is not predicting the delay — it's deciding what to do fast enough for the decision to matter.

By pairing prediction with a constraint-aware reallocation layer, this project approximates the actual decision loop an ops team runs. The model output becomes an input to an action, not just a report.

The trade-offs in this system also reflect real operational trade-offs: reserve crew are cheaper to mobilize than deadheading line crew from another station, but there are only so many. Pulling a crew from their next scheduled pairing saves this flight but creates a problem three hours later. These are the kinds of constraints that make airline operations interesting as a decision problem.

---

## How It Works

### Data Pipeline

Three data sources are combined:

**BTS On-Time Performance** (Bureau of Transportation Statistics)
Historical flight-level records for US domestic routes from 2018–2023. This is the backbone — it provides the delay ground truth and all the schedule-level features. Pulled directly from the BTS TRANSTATS portal, covering ~6 million flights per year.

**NOAA Integrated Surface Database (METAR)**
Hourly weather observations for the top 50 US airports by flight volume. METAR codes are parsed and encoded into an ordinal severity scale (thunderstorm → freezing rain → snow/fog → rain → clear). A 3-hour rolling max is computed for each airport to capture deteriorating conditions, not just the snapshot at departure.

**FAA ASPM (Aviation System Performance Metrics)**
Airport-level demand and capacity data, including active Ground Delay Program (GDP) flags and arrival/departure acceptance rates. When the demand-to-capacity ratio exceeds ~0.85 at a hub, queuing delays begin to compound — this ratio is a strong predictor of systemic delay events.

The three sources are merged on airport + date + hour. The final dataset has roughly 500 features per flight after engineering.

### Feature Engineering

The feature set is built around a central question: what can you actually know at T-3 hours before a flight that predicts whether it will be delayed?

**Schedule-derived features** capture structural risk — tight turn times (under 45 minutes between inbound and outbound), peak-hour departures, hub-to-hub routes, holiday proximity. These don't change between now and departure.

**Weather features** encode current and near-term conditions at the origin: wind speed, visibility, ceiling height, precipitation type, and the 3-hour severity trend. The trend matters more than the snapshot — a deteriorating situation is more disruptive than a steady-state bad condition because the system hasn't had time to adapt.

**Congestion features** capture the airport's operational health at the time of departure: how loaded is the system relative to its normal capacity? Is a GDP active? Is the demand-to-capacity ratio above the queuing threshold?

**Lag and rolling features** capture chronic patterns: the previous day's delay rate at the origin airport, the rolling 7-day delay rate, and the route-specific 30-day delay rate for the same airline. Airports with structural delay problems show up clearly in these signals.

### Modeling Approach

**Baseline models** (logistic regression, decision tree) were trained first and used as a floor. Both perform reasonably on accuracy but poorly on recall — they tend to miss delayed flights because the class is imbalanced (~25% of flights are delayed by 15+ minutes).

**Primary model: LightGBM** was selected for its performance on imbalanced data, inference speed (sub-100ms per flight, which matters for real-time use), and native support for categorical features. Class imbalance is handled via `scale_pos_weight` rather than resampling — oversampling techniques like SMOTE introduce artificial patterns that don't generalize well to new airports.

The classifier is trained with 5-fold stratified cross-validation, evaluating on Precision-Recall AUC rather than ROC-AUC. On imbalanced data, ROC-AUC is an overly optimistic metric — a model can achieve 0.85 AUC while missing 40% of actual delays. PR-AUC forces the model to do well on the minority class.

The operating threshold is calibrated to a **minimum recall of 75%** — operations would rather deal with some false alarms than miss real disruptions. At this threshold, the model flags roughly 1 in 3 departing flights as elevated risk, of which about 55–60% are actually delayed (precision). The improvement in prediction accuracy vs the logistic baseline was approximately 25% measured on PR-AUC.

**SHAP explanations** are generated for every flagged flight. The OCC coordinator sees not just a risk score but: "This flight is flagged because weather severity at ORD is at 3/4 and rising, the demand-capacity ratio is 1.12 (above the queuing threshold), and this route has had a 42% delay rate in the last 30 days." That's actionable. A score alone is not.

### Crew Reallocation Engine

The reallocation engine takes a flagged flight and returns a ranked list of eligible crew members who could cover it. The engine applies constraints in sequence — failing fast on the most common exclusion reasons:

1. **Qualification check** — does the crew hold the rating for this aircraft type? Widebody-qualified crew can cover narrowbody routes; the reverse is not true.
2. **Rest requirement** — has the crew completed the minimum 10-hour rest period (FAA Part 117)?
3. **Duty period projection** — would assigning this flight push them over the 9-hour maximum flight duty period for short-haul operations?
4. **Cumulative hour limits** — 60 hours in 7 days, 190 hours in 28 days

Eligible crew are then scored on a cost function that weighs: reserve status (reserve crew are purpose-built for this and should be used first), deadhead cost if the crew needs to reposition from another station, downstream disruption risk (pulling crew from a tight subsequent pairing creates a cascade), and remaining duty time cushion.

The system returns the top 3 options with a full cost breakdown — not a single "best" answer. The OCC coordinator picks. There are seniority rules, crew welfare considerations, and union constraints the model doesn't know about.

### Downstream Impact Simulation

A Monte Carlo simulation propagates delay through the subsequent legs on the same aircraft rotation. This matters because a 30-minute delay on the first flight of the day can become a 90-minute delay by the fourth leg if nothing intervenes.

The simulation runs 500 iterations per scenario, drawing operational noise from a bounded uniform distribution around the base propagation factor (~65% delay carry-over per leg, consistent with the disruption management literature). Results are compared: baseline (no intervention) vs. with the top reallocation action.

Simulated results across test scenarios showed the reallocation recommendations reduced downstream delay by approximately 18% on average — with larger gains on hub-to-hub routes where propagation chains are longest.

### Decision Trade-off Layer

The final output is a cost-vs-reliability matrix for each flagged flight. It shows:
- Risk score and confidence interval
- Top crew options with cost estimates
- Projected downstream delay avoided under each option
- Estimated time to crew readiness
- A flagged escalation if no eligible crew is available

This is designed to reduce decision latency — the goal was to get a coordinator from "alert received" to "decision made" in under 5 minutes for straightforward cases.

---

## Results

| Metric | Logistic Baseline | LightGBM |
|---|---|---|
| PR-AUC | 0.41 | 0.63 |
| Recall (at operating threshold) | 0.58 | 0.76 |
| Precision (at operating threshold) | 0.41 | 0.57 |
| False alarm rate | 0.31 | 0.19 |
| Inference time (per flight) | ~5ms | ~45ms |

| Simulation Metric | Baseline (No Action) | With Reallocation |
|---|---|---|
| Mean downstream delay (minutes) | 94.2 | 77.1 |
| Reduction | — | ~18% |
| P5 / P95 range | 22 / 198 | 14 / 159 |

---

## Project Structure

```
flight-ops-decision-system/
├── data/
│   ├── raw/            # BTS, NOAA, FAA ASPM source files
│   ├── processed/      # Cleaned and merged parquet files
│   └── simulated/      # Synthetic crew rosters and disruption scenarios
├── notebooks/
│   ├── 01_eda_flight_delays.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_evaluation.ipynb
│   ├── 04_crew_reallocation_simulation.ipynb
│   └── 05_decision_system_demo.ipynb
├── src/
│   ├── data/           # Ingestion and validation
│   ├── features/       # Feature builders for each data source
│   ├── models/         # Training, evaluation, SHAP explanations
│   ├── reallocation/   # Duty rules, optimizer, downstream simulator
│   ├── decision/       # End-to-end pipeline and output formatting
│   └── utils/
├── tests/
├── configs/
│   ├── model_config.yaml
│   └── reallocation_config.yaml
├── requirements.txt
├── Makefile
└── README.md
```

---

## Getting Started

**Requirements:** Python 3.10+, ~8GB RAM for full dataset training

### Setup

```bash
git clone https://github.com/yourusername/flight-ops-decision-system.git
cd flight-ops-decision-system
pip install -r requirements.txt
python setup.py develop
```

### Pull data

```bash
make data
```

This downloads BTS on-time data for 2022–2023 and NOAA ISD weather for the top 50 US airports by flight volume. Expect ~2GB download and 15–20 minutes depending on connection speed.

### Build features

```bash
make features
```

Runs the full feature engineering pipeline and saves the processed dataset to `data/processed/`. Takes about 8 minutes on a standard laptop.

### Train the model

```bash
make train
```

5-fold stratified CV. Prints per-fold AUC and overall OOF metrics. Saves the 5 fold models to `models/`. Takes 10–15 minutes.

### Evaluate

```bash
make evaluate
```

Runs the full evaluation suite on the held-out test set (2023 data): metrics table, confusion matrix, PR curve, SHAP summary plot.

### Run the reallocation simulation

```bash
make simulate
```

Runs the Monte Carlo downstream delay simulation across the predefined disruption scenarios in `data/simulated/`. Outputs the before/after summary.

### Run all tests

```bash
make test
```

### Walk through the full demo

Open `notebooks/05_decision_system_demo.ipynb`. This runs the end-to-end pipeline on a sample of 500 flights from the test set and shows the OCC-style output for each flagged flight.

---

## Configuration

Model behavior is controlled by `configs/model_config.yaml`. The most important parameter for operational use is `target_recall` — it sets the minimum fraction of real delays the model must catch, which drives the classification threshold. Lowering it raises precision and reduces false alarms; raising it catches more delays at the cost of more false flags.

Reallocation behavior is controlled by `configs/reallocation_config.yaml`. The `cost_weights` section lets you tune how the optimizer balances deadhead cost vs. downstream risk vs. reserve preference.

---

## Limitations & Honest Caveats

The delay prediction model was trained on 2018–2023 BTS data and then tested on held-out 2023 flights. The COVID years (2020–2021) were included but the disruption patterns during that period are not representative of normal operations — the model may underweight certain kinds of systemic disruption it didn't see much of in training.

The crew reallocation engine uses a simplified version of FAA Part 117. The real regulatory picture is substantially more complex (different rules for different crew compositions, augmented crew rules for ultra-long-haul, carrier-specific CBA provisions). This engine is a functional approximation, not a compliance tool.

The downstream delay simulation uses a fixed propagation factor calibrated to the literature rather than airline-specific turn time data. Real propagation rates vary significantly by carrier, hub, and aircraft type.

This system is a decision support tool. It is not designed to make assignments autonomously, and shouldn't be. The reallocation recommendations surface trade-offs and options — the OCC coordinator makes the call.

---

## Business Context

Flight delays cost US airlines approximately $28–33 billion annually in direct operational costs, accounting for fuel burn, crew overtime, passenger compensation, and slot violations. The indirect costs — customer satisfaction, loyalty, downstream cancellation chains — are harder to measure but comparable in scale.

The specific operational value this kind of system targets is **decision latency**. When an operations coordinator identifies a likely disruption 2.5 hours before departure instead of 45 minutes, the menu of available responses is dramatically larger. Crew can be repositioned, passengers rebooked proactively, and downstream pairings adjusted. The 18% simulated reduction in downstream delay minutes reflects this window.

At scale across a major carrier's network, reducing average downstream propagation by even 10 minutes per disrupted flight event represents tens of millions of dollars in annual avoidable cost — before counting the softer benefits in on-time performance rankings and customer re-booking rates.

---

## Tech Stack

- **Python 3.10** — core language
- **LightGBM** — primary delay classifier
- **scikit-learn** — preprocessing, cross-validation, baseline models
- **SHAP** — model explanation
- **pandas / numpy** — data manipulation
- **pytest** — testing
- **PyYAML** — configuration

No cloud dependencies. Everything runs locally.

---

## License

MIT
