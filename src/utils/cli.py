from __future__ import annotations

import json
import sys

from src.decision.pipeline import build_dataset, evaluate_model, run_reallocation_and_simulation, train_models


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m src.utils.cli [data|features|train|evaluate|simulate]")

    command = sys.argv[1].strip().lower()
    if command in {"data", "features"}:
        df = build_dataset()
        print(json.dumps({"rows": len(df), "status": "ok"}, indent=2))
        return
    if command == "train":
        print(json.dumps(train_models(), indent=2))
        return
    if command == "evaluate":
        print(json.dumps(evaluate_model(), indent=2))
        return
    if command == "simulate":
        out = run_reallocation_and_simulation()
        print(json.dumps({"recommendations": len(out)}, indent=2))
        return

    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
