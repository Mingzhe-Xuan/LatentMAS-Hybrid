#!/usr/bin/env python3
"""Average numeric fields from repeated run.py JSON summaries."""

import argparse
import json
from pathlib import Path
from typing import Any


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def average_values(values: list[Any]) -> Any:
    """Recursively average numeric leaves and retain identical metadata."""
    if values and all(is_number(value) for value in values):
        return round(sum(values) / len(values), 6)

    if values and all(isinstance(value, dict) for value in values):
        common_keys = set(values[0])
        for value in values[1:]:
            common_keys.intersection_update(value)
        return {
            key: average_values([value[key] for value in values])
            for key in values[0]
            if key in common_keys
        }

    if values and all(value == values[0] for value in values[1:]):
        return values[0]

    return values


def build_average(documents: list[dict[str, Any]], source_files: list[Path]) -> dict[str, Any]:
    if not documents:
        raise ValueError("No repeated experiment summaries were found")

    averaged = average_values(documents)
    seeds = [document.get("run", {}).get("seed") for document in documents]
    if isinstance(averaged.get("run"), dict):
        averaged["run"].pop("seed", None)

    return {
        "aggregation": {
            "repetitions": len(documents),
            "seeds": seeds,
            "source_files": [str(path) for path in source_files],
        },
        "average": averaged,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-repetitions", type=int)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    documents = [json.loads(path.read_text(encoding="utf-8-sig")) for path in args.inputs]
    if args.expected_repetitions is not None and len(documents) != args.expected_repetitions:
        raise ValueError(
            f"Expected {args.expected_repetitions} repeated summaries, found {len(documents)}"
        )

    output = build_average(documents, args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Averaged {len(documents)} runs into {args.output}")


if __name__ == "__main__":
    main()