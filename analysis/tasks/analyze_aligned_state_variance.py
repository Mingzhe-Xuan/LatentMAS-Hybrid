#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tasks._analysis import analyze_variance
from analysis.tasks._common import parser
if __name__ == "__main__":
    raise SystemExit(analyze_variance(parser(analysis=True).parse_args()))
