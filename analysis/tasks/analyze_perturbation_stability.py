#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tasks._analysis import analyze_receiver_table
from analysis.tasks._common import parser

def selected(c):
    return (int(c.get("k", -1)) in {4, 40} and c.get("alignment") in {"kernel", "soft", "linear"}
            and c.get("receiver_model_id") == "Qwen/Qwen3-8B")

if __name__ == "__main__":
    raise SystemExit(analyze_receiver_table(parser(analysis=True).parse_args(),
        "analyze_perturbation_stability", selected, "alpha", "alignment"))
