#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tasks._analysis import analyze_receiver_table
from analysis.tasks._common import parser

def selected(c):
    return (c.get("alignment") == "kernel" and float(c.get("alpha", 0)) == 0
            and c.get("receiver_model_id") == "Qwen/Qwen3-8B"
            and (c.get("sender_model_id") in ("Qwen/Qwen3-8B", "receiver-only")))

if __name__ == "__main__":
    raise SystemExit(analyze_receiver_table(parser(analysis=True).parse_args(),
        "analyze_kernel_scaling", selected, "k", "alignment"))
