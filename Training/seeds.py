"""
Training/seeds.py — the single source of truth for run identity and seeding.
=============================================================================
Every training run in this project is uniquely identified by the triple
``(algo, config_id, seed)``. This module owns two things and nothing else:

1. ``RunSpec`` — the canonical, validated ``(algo, config_id, seed)`` identity,
   which produces the W&B run name ``{algo}_{config_id}_seed{N}`` and the
   matching checkpoint/CSV stem (see CLAUDE.md § Conventions).
2. ``seed_everything`` — deterministically seeds Python ``random``, NumPy, and
   PyTorch from one integer, so a run is reproducible from its ``seed`` alone.

Design notes
------------
- This module deliberately has **no Gymnasium / SB3 import at module scope** so
  it can be imported and unit-tested in isolation (and so importing it never
  drags in the heavy RL stack). ``torch`` is imported lazily inside
  ``seed_everything`` and treated as optional, and ``seed_env`` operates on any
  object exposing Gymnasium-style ``action_space`` / ``observation_space``.
- The canonical seed budgets (10 main / 5 exploratory / 3 sanity) live here so
  every script draws from the same definition rather than hardcoding ``range(10)``.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------

# Algorithms this project trains. A controlled set so a typo in a config never
# silently produces a new, untracked run-name namespace.
VALID_ALGOS: frozenset[str] = frozenset(
    {"dqn", "ppo", "a2c", "double_dqn", "dueling_dqn"}
)

# Canonical seed budgets (CLAUDE.md § Statistical reporting:
# "10 seeds for main results, 5 for exploratory, 3 only for sanity checks").
MAIN_SEEDS: tuple[int, ...] = tuple(range(10))        # 0..9
EXPLORATORY_SEEDS: tuple[int, ...] = tuple(range(5))  # 0..4
SANITY_SEEDS: tuple[int, ...] = tuple(range(3))       # 0..2

SEED_TIERS: dict[str, tuple[int, ...]] = {
    "main": MAIN_SEEDS,
    "exploratory": EXPLORATORY_SEEDS,
    "sanity": SANITY_SEEDS,
}


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """
    The validated identity of a single training run.

    Frozen (immutable + hashable) so a set of ``RunSpec`` is the natural way to
    deduplicate a training queue, and so a spec can be used as a dict key.

    Parameters
    ----------
    algo : str
        One of ``VALID_ALGOS``.
    config_id : str
        The config identifier, matching a file ``configs/{config_id}.yaml``.
        Must be non-empty and contain no whitespace or path separators, so the
        derived run name is safe as a filename and a W&B run name.
    seed : int
        Non-negative integer seed.
    """

    algo: str
    config_id: str
    seed: int

    def __post_init__(self) -> None:
        if self.algo not in VALID_ALGOS:
            raise ValueError(
                f"algo {self.algo!r} is not one of {sorted(VALID_ALGOS)}"
            )
        if not isinstance(self.config_id, str) or not self.config_id:
            raise ValueError("config_id must be a non-empty string")
        if any(c.isspace() for c in self.config_id) or any(
            sep in self.config_id for sep in ("/", "\\")
        ):
            raise ValueError(
                f"config_id {self.config_id!r} must contain no whitespace or "
                "path separators"
            )
        # bool is a subclass of int — reject it explicitly so True/False can't
        # masquerade as a seed.
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {type(self.seed).__name__}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    @property
    def run_name(self) -> str:
        """W&B run name: ``{algo}_{config_id}_seed{N}`` (CLAUDE.md convention)."""
        return f"{self.algo}_{self.config_id}_seed{self.seed}"

    def checkpoint_stem(self) -> str:
        """Filesystem-safe stem for this run's checkpoints (== run_name)."""
        return self.run_name


def run_specs(
    algo: str, config_id: str, tier: str = "main"
) -> list[RunSpec]:
    """
    Build the list of ``RunSpec`` for one (algo, config_id) across a seed tier.

    Parameters
    ----------
    algo, config_id : str
        Passed through to ``RunSpec`` (and validated there).
    tier : {"main", "exploratory", "sanity"}
        Which canonical seed budget to use.
    """
    if tier not in SEED_TIERS:
        raise ValueError(f"tier {tier!r} must be one of {sorted(SEED_TIERS)}")
    return [RunSpec(algo, config_id, seed) for seed in SEED_TIERS[tier]]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int, *, deterministic_torch: bool = True) -> int:
    """
    Seed Python ``random``, NumPy, and (if installed) PyTorch from one integer.

    This is the project-wide entry point for reproducibility. Call it once at
    the start of every script, before constructing the env or the model.

    Parameters
    ----------
    seed : int
        Non-negative seed.
    deterministic_torch : bool
        If True and torch is installed, request deterministic cuDNN behaviour
        (disables the autotuner). Slightly slower but reproducible on GPU.

    Returns
    -------
    int
        The seed, for convenient chaining/logging.

    Notes
    -----
    Env seeding is intentionally *not* done here — Gymnasium envs are seeded via
    ``env.reset(seed=...)`` plus ``seed_env`` for their spaces, which the caller
    owns. Keeping torch optional lets this module be imported without the RL
    stack present.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed


def seed_env(env, seed: int):
    """
    Seed a Gymnasium-style env's sampling spaces deterministically.

    Seeds ``action_space`` and ``observation_space`` so that ``.sample()`` is
    reproducible. The caller is still responsible for passing ``seed`` to
    ``env.reset(seed=seed)`` for the episode/layout RNG — this function does not
    call ``reset`` so it has no side effect on episode state.

    Returns the env for chaining.
    """
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect run identity and demonstrate seeding."
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        help=f"one of {sorted(VALID_ALGOS)}")
    parser.add_argument("--config", type=str, default="ppo_exp4",
                        help="config_id (matches configs/{config_id}.yaml)")
    parser.add_argument("--seed", type=int, default=0, help="run seed")
    parser.add_argument("--tier", type=str, default="main",
                        choices=sorted(SEED_TIERS))
    args = parser.parse_args()

    spec = RunSpec(args.algo, args.config, args.seed)
    print(f"RunSpec        : {spec}")
    print(f"run_name       : {spec.run_name}")
    print(f"checkpoint stem: {spec.checkpoint_stem()}")
    print(f"{args.tier} tier seeds : {SEED_TIERS[args.tier]}")

    seed_everything(args.seed)
    draw = np.random.random(3)
    print(f"seeded np draw : {draw}")
    print("[OK] seeds.py smoke test complete.")