"""Launch the experiments configured by :mod:`exp.sh`.

The script accepts the same command-line options and environment variables as
``exp.sh``.  It uses the active Python interpreter, so activate the project
virtual environment before running or submitting it.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


TARGETS = ("approximator", "latent_cot", "latent_comm")


def _env(name: str, default: str = "") -> str:
    """Return an environment variable, treating an empty value as unset."""
    return os.environ.get(name) or default


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a plan_v2 experiment (Python equivalent of exp.sh)."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--approximator", dest="target_flag", action="store_const", const="approximator"
    )
    target.add_argument(
        "--latent_cot", dest="target_flag", action="store_const", const="latent_cot"
    )
    target.add_argument(
        "--latent_comm", dest="target_flag", action="store_const", const="latent_comm"
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default=None,
        help="Alternative to the three target flags (default: EXP_TARGET or approximator).",
    )
    parser.add_argument("--study", default=None)
    parser.add_argument("--model-pair", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--agent-models",
        default=None,
        help="One or four space-separated model names.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--orf-seed", default=None)
    parser.add_argument("--m", default=None)
    parser.add_argument("--tau", default=None)
    parser.add_argument("--probe-seed", default=None)
    parser.add_argument("--max-questions", default=None)
    parser.add_argument("--latent-steps", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--extra-args",
        default=None,
        help="Additional flags passed to the selected experiment entry point.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without launching the experiment.",
    )
    return parser


def _normalise_cuda_devices(environment: dict[str, str]) -> None:
    """Convert PBS GPU UUID assignments to local zero-based CUDA indices."""
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if "GPU-" not in visible:
        return
    gpu_count = len([item for item in visible.split(",") if item.strip()])
    if gpu_count:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(gpu_count)))


def build_command(
    namespace: argparse.Namespace,
    environment: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Resolve settings and return ``(command, child_environment, summary)``."""
    source_env = dict(os.environ if environment is None else environment)
    target = namespace.target_flag or namespace.target or source_env.get("EXP_TARGET")
    target = target or "approximator"
    if target not in TARGETS:
        raise ValueError(
            f"invalid EXP_TARGET {target!r}; choose one of: {', '.join(TARGETS)}"
        )

    def resolve(cli_name: str, env_name: str, default: str = "") -> str:
        cli_value = getattr(namespace, cli_name)
        if cli_value is not None:
            return cli_value
        return source_env.get(env_name) or default

    m = resolve("m", "M", "2048")
    tau = resolve("tau", "TAU", "1.0")
    orf_seed = resolve("orf_seed", "ORF_SEED", "101")
    probe_seed = resolve("probe_seed", "PROBE_SEED", "42")
    device = resolve("device", "DEVICE", "cuda")
    kernel_chunk_size = source_env.get("KERNEL_CHUNK_SIZE") or "4096"
    max_states = source_env.get("MAX_STATES_PER_QUESTION") or "50"
    max_reply_tokens = source_env.get("MAX_REPLY_TOKENS") or "512"
    generation_seed = source_env.get("GENERATION_SEED") or "77"

    if target == "approximator":
        study = resolve("study", "STUDY", "all")
        model_pair = resolve("model_pair", "MODEL_PAIR", "x1")
        dataset = resolve("dataset", "DATASET", "arc_easy")
        split = resolve("split", "SPLIT", "test")
        method = resolve("method", "METHOD", "kernel")
        max_questions = resolve("max_questions", "MAX_QUESTIONS", "50")
        latent_steps = resolve("latent_steps", "LATENT_STEPS", "50")
        agent_models_text = resolve(
            "agent_models", "AGENT_MODELS", "Qwen/Qwen3-4B"
        )
        agent_models = shlex.split(agent_models_text)
        if not agent_models:
            raise ValueError("AGENT_MODELS/--agent-models must contain a model name")
        entry = "exp/approximator/run.py"
        args = [
            "--study", study,
            "--agent_models", *agent_models,
            "--dataset", dataset,
            "--split", split,
            "--kernel_features", m,
            "--kernel_temperature", tau,
            "--kernel_seed", orf_seed,
            "--probe_seed", probe_seed,
            "--max_questions", max_questions,
            "--max_states_per_question", max_states,
            "--max_new_tokens", max_reply_tokens,
            "--latent_steps", latent_steps,
            "--kernel_chunk_size", kernel_chunk_size,
            "--device", device,
        ]
        model_summary = agent_models_text
    elif target == "latent_cot":
        study = resolve("study", "STUDY", "c0")
        model_pair = resolve("model_pair", "MODEL_PAIR", "c0")
        mas_study = study in {"c1", "c2"}
        model_name = resolve(
            "model_name", "MODEL_NAME",
            "Qwen/Qwen3-8B" if mas_study else "Qwen/Qwen3-4B",
        )
        dataset = resolve(
            "dataset", "DATASET", "mbppplus" if mas_study else "all"
        )
        split = resolve("split", "SPLIT", "test")
        method = resolve("method", "METHOD")
        max_questions = resolve(
            "max_questions", "MAX_QUESTIONS", "30" if mas_study else "50"
        )
        latent_steps = resolve("latent_steps", "LATENT_STEPS", "150")
        latent_step_values = shlex.split(
            source_env.get("LATENT_STEP_VALUES")
            or "20 40 60 80 100 120 140 160 180"
        )
        alignments = shlex.split(
            source_env.get("ALIGNMENTS") or "identical linear soft kernel"
        )
        entry = "exp/latent_cot/run.py"
        args = [
            "--study", study,
            "--model_name", model_name,
            "--dataset", dataset,
            "--split", split,
            "--probe_seed", probe_seed,
            "--max_questions", max_questions,
            "--latent_steps", latent_steps,
            "--device", device,
        ]
        if mas_study:
            mas_reply_tokens = (
                source_env.get("LATENT_COT_MAX_NEW_TOKENS")
                or source_env.get("MAX_REPLY_TOKENS")
                or "4096"
            )
            args.extend(["--latent_step_values", *latent_step_values])
            args.extend(["--alignments", *alignments])
            args.extend(
                [
                    "--max_new_tokens", mas_reply_tokens,
                    "--generation_seed", generation_seed,
                ]
            )
        model_summary = model_name
    else:
        study = resolve("study", "STUDY", "m0")
        model_pair = resolve("model_pair", "MODEL_PAIR", "x1")
        dataset = resolve("dataset", "DATASET", "communication_probe")
        split = resolve("split", "SPLIT", "test")
        method = resolve("method", "METHOD", "all")
        max_questions = resolve("max_questions", "MAX_QUESTIONS", "50")
        latent_steps = resolve("latent_steps", "LATENT_STEPS", "4")
        entry = "exp/latent_comm/run.py"
        args = [
            "--study", study,
            "--model_pair", model_pair,
            "--dataset", dataset,
            "--split", split,
            "--method", method,
            "--orf_seed", orf_seed,
            "--m", m,
            "--tau", tau,
            "--latent_steps", latent_steps,
            "--generation_seed", generation_seed,
            "--device", device,
        ]
        model_summary = model_pair

    extra_text = resolve("extra_args", "EXP_EXTRA_ARGS")
    extra_args = shlex.split(extra_text) if extra_text else []
    child_env = source_env.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["HF_HOME"] = source_env.get("HF_HOME") or "/home/n2501945g/.cache/huggingface"
    child_env["HF_HUB_CACHE"] = (
        source_env.get("HF_HUB_CACHE") or f"{child_env['HF_HOME']}/hub"
    )
    child_env["HF_DATASETS_CACHE"] = (
        source_env.get("HF_DATASETS_CACHE") or f"{child_env['HF_HOME']}/datasets"
    )
    _normalise_cuda_devices(child_env)

    summary = {
        "target": target,
        "study": study,
        "model": model_summary,
        "dataset": dataset,
        "split": split,
        "method": method,
        "m": m,
        "tau": tau,
        "orf_seed": orf_seed,
        "latent_steps": (
            " ".join(latent_step_values)
            if target == "latent_cot" and study in {"c1", "c2"}
            else latent_steps
        ),
        "entry": entry,
        "max_questions": max_questions,
    }
    return [sys.executable, "-u", entry, *args, *extra_args], child_env, summary


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def _print_summary(summary: Mapping[str, str]) -> None:
    print("=" * 72)
    print(f"PBS job       : {_env('PBS_JOBID', 'interactive')}")
    print(f"Target/study  : {summary['target']}/{summary['study']}")
    label = (
        "Agent models"
        if summary["target"] == "approximator"
        else "Model" if summary["target"] == "latent_cot" else "Model pair"
    )
    print(f"{label:<14}: {summary['model']}")
    print(f"Dataset/split : {summary['dataset']}/{summary['split']}")
    print(f"Method        : {summary['method']}")
    print(
        "ORF (m,tau,seed): "
        f"{summary['m']}, {summary['tau']}, {summary['orf_seed']}"
    )
    print(f"Latent steps  : {summary['latent_steps']}")
    print(f"Host          : {socket.gethostname()}")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "-L"], check=False)
    else:
        print("nvidia-smi    : unavailable")
    print("=" * 72)


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    try:
        command, child_env, summary = build_command(namespace)
    except (ValueError, OSError) as exc:
        _parser().error(str(exc))

    workdir = Path(_env("PBS_O_WORKDIR", str(Path(__file__).resolve().parent)))
    if not (workdir / "exp").is_dir():
        print(
            f"ERROR: repository root expected at {workdir} (exp/ was not found).",
            file=sys.stderr,
        )
        return 1

    _print_summary(summary)
    entry_path = workdir / summary["entry"]
    if not entry_path.is_file():
        print(
            f"ERROR: {summary['entry']} does not exist yet; "
            "no experiment was launched.",
            file=sys.stderr,
        )
        return 2

    rendered = _display_command(command)
    if namespace.dry_run:
        print(f"Dry run: {rendered}")
        return 0

    default_log = workdir / "exp_state.txt"
    log_path = Path(_env("EXP_STATE_LOG", str(default_log))).expanduser()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    job_id = _env("PBS_JOBID", "interactive")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{timestamp}] PBS job {job_id}: {rendered}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=workdir,
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode:
        print(
            f"Experiment failed with exit code {result.returncode}; "
            f"output appended to: {log_path}",
            file=sys.stderr,
        )
        return result.returncode
    print(f"Completed successfully; progress and output appended to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
