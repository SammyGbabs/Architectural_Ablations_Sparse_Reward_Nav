"""
Training/run_phase2_sweep.py — Phase 2 claim-grade multi-seed queue (symmetric-only).
===================================================================================
Promotes the load-bearing Phase 2 ladder cells from single-seed gates to 10-seed,
claim-grade results. SYMMETRIC arch only (pi=vf=[256,256], Exp 1 verbatim): these
establish the task ceiling/fracture for a STANDARD agent — a statement about the
TASK, not an architecture comparison. This is NOT the inverted-vs-symmetric sweep
(the ladder concluded there is no headroom for it).

Cells (execution order = front-loaded by argument-criticality x cost):
  1. p2_strict_sym     A-STRICT 13-D            200k   base-of-ladder ceiling (load-bearing)
  2. p2_alias_sym      aliasing 10-D            200k   Track-1-directed ceiling
  3. p2_flicker08_sym  flicker p=0.8 52-D       200k   fracture claim (distinct)
  4. p2_proxnoise_sym  prox-noise q=0.3 13-D    200k   closing rung
  5. p2_flicker07_sym  flicker p=0.7 52-D       500k   COMPUTE SINK — HOLD (needs --include-hold)

Reuses the Phase 1 sweep machinery (deterministic ids, resume="allow", .done
markers, per-run subprocess) via Training.run_sweep. The expensive p=0.7 x 500k
cell is EXCLUDED by default; pass --include-hold only after explicit confirmation.

    python -m Training.run_phase2_sweep --output-dir /content/drive/MyDrive/arch-ablations --dry-run
"""

from __future__ import annotations

import argparse
import sys

from Training.seeds import RunSpec, run_specs
from Training.run_sweep import (
    build_command,
    format_queue,
    is_run_complete,
    mark_complete,
    subprocess,
)

# Execution order: cheap load-bearing cells first, compute sink last.
PHASE2_ORDER = [
    "p2_strict_sym",
    "p2_alias_sym",
    "p2_flicker08_sym",
    "p2_proxnoise_sym",
    "p2_flicker07_sym",   # HOLD — 500k x 10, gated behind --include-hold
]
HOLD_CELLS = {"p2_flicker07_sym"}
ALGO = "ppo"              # all Phase 2 cells are symmetric PPO


def build_queue(include_hold: bool = False) -> list[RunSpec]:
    """Ordered 10-seed queue over the Phase 2 cells (hold cells excluded unless asked)."""
    queue: list[RunSpec] = []
    for cid in PHASE2_ORDER:
        if cid in HOLD_CELLS and not include_hold:
            continue
        queue += run_specs(ALGO, cid, "main")      # MAIN tier = 10 seeds
    return queue


def run_sweep(args: argparse.Namespace) -> int:
    queue = build_queue(include_hold=args.include_hold)
    if args.limit:
        queue = queue[: args.limit]

    held = "" if args.include_hold else f"  (HOLD excluded: {sorted(HOLD_CELLS)})"
    if args.dry_run:
        print(f"Phase 2 claim-grade sweep - {len(queue)} runs (symmetric-only){held}:\n")
        print(format_queue(queue))
        tag = getattr(args, "sweep_tag", None)
        print(f"\nW&B namespace tag: {tag!r};  output dir: {args.output_dir}")
        print("(dry run — nothing launched)")
        return 0

    failures: list[str] = []
    for i, spec in enumerate(queue, 1):
        if is_run_complete(args.output_dir, spec.run_name):
            print(f"[p2-sweep {i}/{len(queue)}] SKIP (done): {spec.run_name}")
            continue
        print(f"[p2-sweep {i}/{len(queue)}] RUN: {spec.run_name}")
        rc = subprocess.run(build_command(spec, args)).returncode
        if rc == 0:
            mark_complete(args.output_dir, spec.run_name)
        else:
            failures.append(f"{spec.run_name} (rc={rc})")
            if not args.keep_going:
                print(f"[p2-sweep] run failed: {spec.run_name} (rc={rc}). Stopping; "
                      "re-run to resume (finished runs skip).")
                return rc
            print(f"[p2-sweep] run failed: {spec.run_name} (rc={rc}); --keep-going set.")

    if failures:
        print(f"\n[p2-sweep] completed with {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[p2-sweep] all runs complete.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 2 claim-grade multi-seed sweep (symmetric-only; resumable).")
    p.add_argument("--output-dir", type=str, default="runs",
                   help="Root for checkpoints/markers/csv — mounted Drive on Colab.")
    p.add_argument("--configs-dir", type=str, default="configs")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--sweep-tag", type=str, default="p2v1",
                   help="W&B namespace tag (distinct from Phase 1's). '' disables.")
    p.add_argument("--include-hold", action="store_true",
                   help="Include the held compute-sink cell (p2_flicker07_sym, 500k x 10). "
                        "Only pass this after explicit confirmation.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the ordered queue and exit (launch nothing).")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p


if __name__ == "__main__":
    raise SystemExit(run_sweep(build_arg_parser().parse_args()))
