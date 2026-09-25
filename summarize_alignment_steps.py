#!/usr/bin/env python3
"""Summarize batched alignment latency from repeated run.py JSON outputs."""

import argparse
import json
import statistics
from pathlib import Path


def per_step_ms(document: dict, generate_bs: int) -> tuple[float, float, float]:
    timing = document["timing"]["model_phases"]["alignment_seconds"]
    latent = document["results"]["tokens"]["latent_output"]
    alignment_seconds = float(timing["total"])
    latent_output_steps = float(latent["total"])
    effective_steps = latent_output_steps / generate_bs
    if effective_steps <= 0:
        raise ValueError("latent_output.total / generate_bs must be positive")
    return 1000.0 * alignment_seconds / effective_steps, alignment_seconds, effective_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-bs", type=int, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.generate_bs <= 0:
        parser.error("--generate-bs must be positive")

    repeats = []
    values = []
    for path in args.inputs:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        value, alignment_seconds, effective_steps = per_step_ms(
            document, args.generate_bs
        )
        values.append(value)
        repeats.append(
            {
                "source": str(path),
                "seed": document.get("run", {}).get("seed"),
                "alignment_seconds": alignment_seconds,
                "effective_steps": effective_steps,
                "milliseconds_per_step": value,
            }
        )

    mean = statistics.fmean(values)
    population_std = statistics.pstdev(values)
    output = {
        "metric": "text_alignment_milliseconds_per_step",
        "formula": (
            "1000 * alignment_seconds.total / "
            "(latent_output.total / generate_bs)"
        ),
        "generate_bs": args.generate_bs,
        "repetitions": len(values),
        "mean_milliseconds_per_step": mean,
        "population_std_milliseconds_per_step": population_std,
        "reported": f"{mean:.5f} +/- {population_std:.5f} ms/step",
        "repeat_values": repeats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output["reported"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
