#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tasks._common import parser
from analysis.tasks._evaluate import run_evaluation
if __name__ == "__main__":
    raise SystemExit(run_evaluation(parser().parse_args(), "evaluate_perturbation_stability"))
