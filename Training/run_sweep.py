"""
Training/run_sweep.py — Phase 1 multi-seed sweep runner (105 runs).
===================================================================
Builds the full ``(algo, config_id, seed)`` run queue and launches each run as
an isolated subprocess (so one crash can't take down the queue, and memory is
reclaimed between runs). Designed for an overnight Colab session with output on
mounted Google Drive.

Queue (105 runs):
  - 9 main configs (DQN Exp 1-5, PPO Exp 1-4) x MAIN_SEEDS (10) = 90
  - 3 baselines (Double/Dueling DQN, A2C) x EXPLORATORY_SEEDS (5) = 15

Ordering (front-loaded so the H1 evidence lands first):
  1. ppo_exp1 + ppo_exp4, all 10 seeds  (the symmetric-vs-inverted comparison)
  2. the rest of the main PPO/DQN configs
  3. the 3 baselines last

Resumability: deterministic run ids (NO --fresh) + resume="allow", plus a
per-run ".done" marker. Re-running the sweep skips finished runs; partial runs
resume from their latest 25k checkpoint via the trainer's own resume logic.

This module does NOT launch anything on import or with --dry-run; --dry-run just
prints the ordered queue.

    python -m Training.run_sweep --output-dir /content/drive/MyDrive/arch-ablations --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from Training.seeds import RunSpec, run_specs
from Training.trainer_common import parse_config_file

# Ordered config groups (front-loaded).
FRONT_LOAD = ["ppo_exp1", "ppo_exp4"]                       # H1 comparison, first
REST_MAIN = ["ppo_exp2", "ppo_exp3",
             "dqn_exp1", "dqn_exp2", "dqn_exp3", "dqn_exp4", "dqn_exp5"]
BASELINES = ["double_dqn", "dueling_dqn", "a2c"]            # exploratory, last

# algo -> trainer entry module (each exposes `--config --seed --output-dir ...`).
TRAINER_MODULE: dict[str, str] = {
    "ppo": "Training.ppo_training",
    "dqn": "Training.dqn_training",
    "double_dqn": "Training.baselines.double_dqn",
    "dueling_dqn": "Training.baselines.dueling_dqn",
    "a2c": "Training.baselines.a2c",
}


# ---------------------------------------------------------------------------
# Queue construction (pure, testable)
# ---------------------------------------------------------------------------

def algo_of(config_id: str, configs_dir: str | Path = "configs") -> str:
    """Authoritative algo for a config: read it from configs/{config_id}.yaml."""
    return parse_config_file(Path(configs_dir) / f"{config_id}.yaml")["algo"]


def build_queue(configs_dir: str | Path = "configs") -> list[RunSpec]:
    """Build the ordered 105-run queue (front-loaded; baselines last)."""
    queue: list[RunSpec] = []
    for config_id in FRONT_LOAD + REST_MAIN:        # 9 main configs @ 10 seeds
        queue += run_specs(algo_of(config_id, configs_dir), config_id, "main")
    for config_id in BASELINES:                     # 3 baselines @ 5 seeds
        queue += run_specs(algo_of(config_id, configs_dir), config_id, "exploratory")
    return queue


def format_queue(queue: list[RunSpec]) -> str:
    """Human-readable, grouped view of the queue for --dry-run."""
    lines, last = [], None
    for i, spec in enumerate(queue, 1):
        key = (spec.algo, spec.config_id)
        if key != last:
            n = sum(1 for s in queue if (s.algo, s.config_id) == key)
            lines.append(f"  [{spec.config_id}]  ({spec.algo}, {n} seeds)")
            last = key
        lines.append(f"      {i:3d}. {spec.run_name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Completion markers (resumability)
# ---------------------------------------------------------------------------

def marker_path(output_dir: str | Path, run_name: str) -> Path:
    return Path(output_dir) / "markers" / f"{run_name}.done"


def is_run_complete(output_dir: str | Path, run_name: str) -> bool:
    return marker_path(output_dir, run_name).is_file()


def mark_complete(output_dir: str | Path, run_name: str) -> None:
    p = marker_path(output_dir, run_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("done\n")


# ---------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------

def build_command(spec: RunSpec, args: argparse.Namespace) -> list[str]:
    """The subprocess command for one run (deterministic id; NO --fresh)."""
    return [
        sys.executable, "-m", TRAINER_MODULE[spec.algo],
        "--config", str(Path(args.configs_dir) / f"{spec.config_id}.yaml"),
        "--seed", str(spec.seed),
        "--output-dir", args.output_dir,
        "--wandb-mode", args.wandb_mode,
    ]


def run_sweep(args: argparse.Namespace) -> int:
    queue = build_queue(args.configs_dir)
    if args.limit:
        queue = queue[: args.limit]

    if args.dry_run:
        print(f"Phase 1 sweep - {len(queue)} runs (front-loaded; baselines last):\n")
        print(format_queue(queue))
        print(f"\n(dry run - nothing launched). Output dir: {args.output_dir}")
        return 0

    failures: list[str] = []
    for i, spec in enumerate(queue, 1):
        if is_run_complete(args.output_dir, spec.run_name):
            print(f"[sweep {i}/{len(queue)}] SKIP (done): {spec.run_name}")
            continue
        print(f"[sweep {i}/{len(queue)}] RUN: {spec.run_name}")
        rc = subprocess.run(build_command(spec, args)).returncode
        if rc == 0:
            mark_complete(args.output_dir, spec.run_name)
        else:
            failures.append(f"{spec.run_name} (rc={rc})")
            if not args.keep_going:
                print(f"[sweep] run failed: {spec.run_name} (rc={rc}). Stopping. "
                      "Re-run the sweep to resume (finished runs are skipped).")
                return rc
            print(f"[sweep] run failed: {spec.run_name} (rc={rc}). --keep-going set; "
                  "continuing.")

    if failures:
        print(f"\n[sweep] completed with {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[sweep] all runs complete.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 multi-seed sweep runner (105 runs; resumable)."
    )
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Root for checkpoints/markers/tb — point at mounted "
                             "Google Drive for Colab (e.g. /content/drive/MyDrive/...).")
    parser.add_argument("--configs-dir", type=str, default="configs",
                        help="Directory holding the per-config YAMLs.")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="Passed through to each trainer.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the ordered queue and exit (launch nothing).")
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue past a failed run instead of stopping.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only consider the first N runs (debug).")
    return parser


if __name__ == "__main__":
    raise SystemExit(run_sweep(build_arg_parser().parse_args()))
