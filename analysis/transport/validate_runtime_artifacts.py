#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.core.config import load_stt_config
from analysis.core.stt import (STTArtifactSpec, load_stt_artifact,
                               transport_tokenizer_fingerprint)
from analysis.transport.complete_reverse_support import normalize_runtime_tokenizer


def validate(config_path: str | Path) -> dict:
    from transformers import AutoTokenizer

    config = load_stt_config(config_path).raw
    tokenizers = {
        key: normalize_runtime_tokenizer(AutoTokenizer.from_pretrained(
            model_id,
            revision=config["model_revisions"][key],
            use_fast=True,
            token=False,
            trust_remote_code=False,
        ))
        for key, model_id in config["models"].items()
    }
    fingerprints = {
        key: transport_tokenizer_fingerprint(tokenizer)
        for key, tokenizer in tokenizers.items()
    }
    artifacts = {}
    for name, declaration in config["transport"]["artifacts"].items():
        source, target = declaration["source"], declaration["target"]
        artifact = load_stt_artifact(
            STTArtifactSpec(
                Path(declaration["path"]), declaration["sha256"], source, target,
                declaration["source_revision"], declaration["target_revision"],
            ),
            source_vocab_size=len(tokenizers[source]),
            target_vocab_size=len(tokenizers[target]),
            source_fingerprint=fingerprints[source],
            target_fingerprint=fingerprints[target],
        )
        artifacts[name] = {
            "path": str(artifact.spec.path),
            "sha256": artifact.spec.sha256,
            "shape": list(artifact.shape),
            "source_vocab_size": len(tokenizers[source]),
            "target_vocab_size": len(tokenizers[target]),
            "max_column_mass_error": artifact.max_column_mass_error,
        }
    return {
        "status": "pass",
        "tokenizers": {
            key: {
                "model_id": config["models"][key],
                "revision": config["model_revisions"][key],
                "vocab_size": len(tokenizer),
                "pad_token_id": tokenizer.pad_token_id,
                "fingerprint": fingerprints[key],
            }
            for key, tokenizer in tokenizers.items()
        },
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate both runtime STT artifacts against locked tokenizers.")
    parser.add_argument("--config", default="analysis/configs/bidirectional_stt.yaml")
    args = parser.parse_args()
    print(json.dumps(validate(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
