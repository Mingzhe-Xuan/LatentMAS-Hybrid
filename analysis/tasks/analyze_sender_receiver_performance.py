#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.tasks._analysis import analyze_receiver_table
from analysis.tasks._common import parser

def selected(c):
    if (int(c.get("k", -1)) not in {0, 4, 40} or c.get("alignment") != "kernel"
            or float(c.get("alpha", 0)) != 0):
        return False
    sender = c.get("sender_model_id", "receiver-only").replace("Qwen/Qwen3-", "")
    receiver = c.get("receiver_model_id", "").replace("Qwen/Qwen3-", "")
    c["model_pair"] = f"{sender}->{receiver}"
    return True

if __name__ == "__main__":
    raise SystemExit(analyze_receiver_table(parser(analysis=True).parse_args(),
        "analyze_sender_receiver_performance", selected, "k", "model_pair"))
