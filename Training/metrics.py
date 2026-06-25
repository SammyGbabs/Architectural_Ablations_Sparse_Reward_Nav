"""
Training/metrics.py — rich per-run evaluation logging for the Phase 1 sweep.
============================================================================
The single-seed result was a tie on reward, so any architecture effect likely
lives in the *other* metrics. This module logs all of them per run:

- success rate, collision rate, mean episode length, wait-action frequency
- per-room success rate AND per-room mean episode length (kitchen / bedroom /
  bathroom — the far bedroom at ~22 steps may discriminate where near rooms tie)
- eval return mean AND IQM (per eval, over the eval episodes)
- sample efficiency: env-steps to reach 90% of the run's asymptotic eval IQM
- a one-row-per-seed CSV under results/csv/ for rliable IQM + CIs later

Path Optimality Ratio (POR) is NOT logged here — it is computed post-hoc (see
the module note at the bottom and the Stage-2 recommendation). We log per-room
mean episode length, which is the only ingredient POR needs on the fixed map.

Everything except ``RichEvalCallback`` is import-safe (no SB3/Gymnasium), so the
pure helpers are unit-tested without the RL stack. The callback derives all
metrics from the env's existing ``info`` dict and the eval actions — it does NOT
modify the environment.
"""

from __future__ import annotations

import csv as _csv
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# SB3 is optional at import time: fall back to ``object`` so the pure helpers
# (and their tests) import without the RL stack. The callback is only ever
# instantiated inside a trainer where SB3 is installed.
try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
except Exception:  # pragma: no cover - exercised only where SB3 is absent
    _BaseCallback = object  # type: ignore[assignment, misc]


WAIT_ACTION = 4  # ACTION_NAMES[4] == "Wait" in Environment/custom_env.py
DEFAULT_TARGET_ROOMS = ("kitchen", "bedroom", "bathroom")  # living is the spawn room
EVAL_SEED_BASE = 10_000  # eval episode seeds, disjoint from training seeds 0..9

# Stable CSV schema (one row per seed). POR columns intentionally omitted —
# POR is post-hoc; len_<room> below is the ingredient it needs.
CSV_COLUMNS = [
    "phase", "config_id", "algo", "seed", "env_steps",
    "eval_return_iqm", "eval_return_mean",
    "success_rate", "collision_rate", "mean_ep_len", "wait_freq",
    "sr_kitchen", "sr_bedroom", "sr_bathroom",
    "len_kitchen", "len_bedroom", "len_bathroom",
    "sample_eff_steps_90", "n_eval_episodes", "wandb_run_name",
]

# Metric keys produced by aggregate_episode_metrics (subset of CSV_COLUMNS).
_AGG_KEYS = [
    "eval_return_iqm", "eval_return_mean",
    "success_rate", "collision_rate", "mean_ep_len", "wait_freq",
    "sr_kitchen", "sr_bedroom", "sr_bathroom",
    "len_kitchen", "len_bedroom", "len_bathroom",
]


# ---------------------------------------------------------------------------
# Pure helpers (no SB3 / Gymnasium)
# ---------------------------------------------------------------------------

def iqm(values: list[float]) -> float:
    """Interquartile mean: mean of the middle 50% (trims the top/bottom 25%)."""
    a = np.sort(np.asarray(list(values), dtype=float))
    n = a.size
    if n == 0:
        return float("nan")
    lo = int(math.floor(n * 0.25))
    hi = int(math.ceil(n * 0.75))
    trimmed = a[lo:hi] if hi > lo else a
    return float(np.mean(trimmed))


def aggregate_episode_metrics(
    episodes: list[dict[str, Any]],
    rooms: tuple[str, ...] = DEFAULT_TARGET_ROOMS,
) -> dict[str, float]:
    """
    Aggregate a list of per-episode dicts into the scalar metrics we log.

    Each episode dict needs: ``return``, ``length``, ``success`` (bool),
    ``collision`` (bool), ``wait_freq`` (float), ``target_room`` (str).
    """
    if not episodes:
        return {}
    rets = [e["return"] for e in episodes]
    out: dict[str, float] = {
        "eval_return_mean": float(np.mean(rets)),
        "eval_return_iqm": iqm(rets),
        "success_rate": float(np.mean([bool(e["success"]) for e in episodes])),
        "collision_rate": float(np.mean([bool(e["collision"]) for e in episodes])),
        "mean_ep_len": float(np.mean([e["length"] for e in episodes])),
        "wait_freq": float(np.mean([e["wait_freq"] for e in episodes])),
        "n_eval_episodes": float(len(episodes)),
    }
    for room in rooms:
        sub = [e for e in episodes if e["target_room"] == room]
        out[f"sr_{room}"] = (
            float(np.mean([bool(e["success"]) for e in sub])) if sub else float("nan")
        )
        out[f"len_{room}"] = (
            float(np.mean([e["length"] for e in sub])) if sub else float("nan")
        )
    return out


def steps_to_fraction_of_asymptote(
    history: list[tuple[int, float]],
    frac: float = 0.90,
    asymptote_window: int = 3,
) -> Optional[int]:
    """
    Sample efficiency: the first env-step at which the eval metric reaches
    ``frac`` of the run's asymptotic value.

    ``history`` is a list of ``(env_step, value)`` in increasing step order
    (value = per-eval IQM return). The asymptote is the mean of the last
    ``asymptote_window`` evals. Returns the crossing step, or None if the
    asymptote is non-positive (threshold ill-defined) or it never crosses.
    """
    if not history:
        return None
    window = max(1, min(asymptote_window, len(history)))
    asymptote = float(np.mean([v for _, v in history[-window:]]))
    if asymptote <= 0:
        return None  # 90% of a non-positive asymptote is not meaningful
    target = frac * asymptote
    for step, value in history:
        if value >= target:
            return int(step)
    return None


def run_eval_episodes(
    model,
    env,
    n_episodes: int,
    base_seed: int = EVAL_SEED_BASE,
    wait_action: int = WAIT_ACTION,
) -> list[dict[str, Any]]:
    """
    Roll out ``n_episodes`` deterministic episodes on a raw (non-vectorised)
    Gymnasium env, collecting the per-episode metric dict. Episodes are seeded
    ``base_seed + i`` so the eval set is fixed across a run (improvement is
    measured on the same episodes) and reproducible across runs.
    """
    episodes: list[dict[str, Any]] = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=base_seed + i)
        done = False
        ret, length, waits = 0.0, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            a = int(action)
            obs, reward, terminated, truncated, info = env.step(a)
            ret += float(reward)
            length += 1
            if a == wait_action:
                waits += 1
            done = bool(terminated or truncated)
        episodes.append({
            "return": ret,
            "length": length,
            "success": bool(info.get("reached_target", False)),
            "collision": bool(info.get("collision", False)),
            "wait_freq": waits / max(length, 1),
            "target_room": info.get("target_room", "unknown"),
        })
    return episodes


def upsert_run_row(csv_path: str | Path, row: dict[str, Any]) -> None:
    """
    Write one per-seed row to ``csv_path``, replacing any existing row with the
    same (config_id, seed) so a re-run overwrites rather than duplicates.
    Creates the file (with header) if absent.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, str]] = []
    if path.exists():
        with open(path, newline="") as f:
            existing = list(_csv.DictReader(f))

    key = (str(row.get("config_id")), str(row.get("seed")))
    existing = [r for r in existing
                if (r.get("config_id"), r.get("seed")) != key]
    existing.append({k: _fmt(row.get(k, "")) for k in CSV_COLUMNS})

    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return repr(round(v, 6))
    return str(v)


# ---------------------------------------------------------------------------
# SB3 callback
# ---------------------------------------------------------------------------

class RichEvalCallback(_BaseCallback):
    """
    Periodic rich evaluation. Every ``eval_freq`` env steps it runs
    ``n_eval_episodes`` deterministic episodes on its own raw eval env and logs
    the full metric set under ``eval/*`` (which sync_tensorboard forwards to
    W&B). Tracks the per-eval IQM history for the sample-efficiency calculation
    and exposes ``last_agg`` for the end-of-run CSV row.
    """

    def __init__(
        self,
        eval_env_fn: Callable[[], Any],
        eval_freq: int,
        n_eval_episodes: int = 30,
        base_seed: int = EVAL_SEED_BASE,
        rooms: tuple[str, ...] = DEFAULT_TARGET_ROOMS,
        best_model_path: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._eval_env_fn = eval_env_fn
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.base_seed = int(base_seed)
        self.rooms = rooms
        self.best_model_path = best_model_path
        self.history: list[tuple[int, float]] = []
        self.last_agg: dict[str, float] = {}
        self._best_iqm = -float("inf")
        self._eval_env = None

    def _init_callback(self) -> None:
        self._eval_env = self._eval_env_fn()

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        episodes = run_eval_episodes(
            self.model, self._eval_env, self.n_eval_episodes, self.base_seed
        )
        agg = aggregate_episode_metrics(episodes, self.rooms)
        self.history.append((int(self.num_timesteps), agg["eval_return_iqm"]))
        self.last_agg = agg

        for key, value in agg.items():
            self.logger.record(f"eval/{key}", value)
        se = steps_to_fraction_of_asymptote(self.history)
        self.logger.record("eval/sample_eff_steps_90", -1 if se is None else se)

        if self.best_model_path and agg["eval_return_iqm"] > self._best_iqm:
            self._best_iqm = agg["eval_return_iqm"]
            self.model.save(self.best_model_path)

    def sample_efficiency(self) -> Optional[int]:
        """Env-steps to reach 90% of the asymptotic eval IQM (None if never)."""
        return steps_to_fraction_of_asymptote(self.history)


def finalize_run_csv(
    csv_path: str | Path,
    *,
    phase: str,
    config_id: str,
    algo: str,
    seed: int,
    env_steps: int,
    callback: "RichEvalCallback",
    wandb_run_name: str = "",
) -> dict[str, Any]:
    """Assemble and write the per-seed CSV row from a finished RichEvalCallback."""
    agg = dict(callback.last_agg)
    se = callback.sample_efficiency()
    row: dict[str, Any] = {
        "phase": phase,
        "config_id": config_id,
        "algo": algo,
        "seed": seed,
        "env_steps": env_steps,
        "sample_eff_steps_90": -1 if se is None else se,
        "n_eval_episodes": int(agg.get("n_eval_episodes", 0)),
        "wandb_run_name": wandb_run_name,
    }
    row.update({k: agg.get(k) for k in _AGG_KEYS})
    upsert_run_row(csv_path, row)
    return row


# ---------------------------------------------------------------------------
# POR note (post-hoc; see Stage-2 recommendation)
# ---------------------------------------------------------------------------
# Path Optimality Ratio = actual_path_length / shortest_path_length. On the
# fixed Phase 1 map with a fixed (0,0) spawn, the shortest path to the nearest
# cell of each target room is a CONSTANT (BFS/A* agree): kitchen 11, bedroom 11,
# bathroom 22. So POR is computed POST-HOC as len_<room> / optimal_<room> from
# the per-room episode lengths logged above — no A* solver is needed for Phase 1.
# Analysis/astar.py (general solver) is Phase 2 work, when procedural maps make
# per-map shortest paths non-constant.
PHASE1_OPTIMAL_STEPS = {"kitchen": 11, "bedroom": 11, "bathroom": 22}
