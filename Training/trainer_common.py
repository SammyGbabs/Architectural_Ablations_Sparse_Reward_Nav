"""
Training/trainer_common.py — shared trainer machinery for PPO and DQN.
=====================================================================
Both ``Training/ppo_training.py`` and ``Training/dqn_training.py`` import from
here so the W&B / checkpoint / evaluation / CSV logic exists in exactly ONE
place and the two trainers cannot drift apart. Each trainer only supplies the
algorithm-specific pieces (config validation + how to build/load the SB3 model)
and calls :func:`run_training`.

Heavy dependencies (Stable-Baselines3, Gymnasium, wandb, torch.nn) are imported
lazily inside functions so ``--help`` and config parsing work without the RL
stack installed. The pure helpers here are import-safe and unit-tested.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from Training.seeds import RunSpec, seed_everything

# ---------------------------------------------------------------------------
# Project constants (one source of truth for both trainers)
# ---------------------------------------------------------------------------

WANDB_PROJECT = "arch-ablations-sparse-reward"
CHECKPOINT_EVERY_STEPS = 25_000   # CLAUDE.md: checkpoint every 25k env steps
PHASE = "p1"                      # {phase}_{config_id}.csv
# Per-seed CSVs are written under <output_dir>/csv/ (so they persist on Drive
# alongside checkpoints across Colab disconnects — see run_training). This is the
# repo canonical dir for the *committed* copies the user syncs back from Drive.
RESULTS_CSV_DIR = Path("results/csv")
CSV_SUBDIR = "csv"                # <output_dir>/csv/  (the live write location)

# Activation-function names accepted in configs (resolved to torch.nn lazily).
SUPPORTED_ACTIVATIONS = ("ReLU", "LeakyReLU", "Tanh", "ELU", "GELU")

# Type aliases for the algorithm-specific callables a trainer plugs in.
ValidateFn = Callable[[dict[str, Any]], None]
BuildModelFn = Callable[[dict[str, Any], Any, int, str], Any]
LoadModelFn = Callable[[Path, Any, str, int, str], Any]


# ---------------------------------------------------------------------------
# Config loading (generic) + activation resolution
# ---------------------------------------------------------------------------

def parse_config_file(config_path: str | Path) -> dict[str, Any]:
    """Read a YAML config into a dict (no algo-specific validation)."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    return cfg


def resolve_activation(name: str | None):
    """Map an activation name string to a ``torch.nn`` class (lazy import)."""
    if name is None:
        return None
    import torch.nn as nn

    table = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "Tanh": nn.Tanh,
        "ELU": nn.ELU,
        "GELU": nn.GELU,
    }
    if name not in table:
        raise ValueError(f"unsupported activation_fn: {name!r}")
    return table[name]


# ---------------------------------------------------------------------------
# Shared config validation + policy_kwargs / kwargs extraction
# ---------------------------------------------------------------------------
# Two algorithm FAMILIES share validation + policy_kwargs shape:
#   * actor-critic (ppo, a2c): net_arch is a {pi, vf} dict.
#   * value-based (dqn, double_dqn, dueling_dqn): net_arch is a flat list.
# All five trainers call these so the rules can't drift apart.

# config_key -> SB3 kwarg for the value-based (DQN) family (shared by dqn +
# double_dqn + dueling_dqn). The actor-critic maps live in their trainers.
DQN_KWARG_MAP: dict[str, str] = {
    "lr": "learning_rate",
    "buffer_size": "buffer_size",
    "batch_size": "batch_size",
    "gamma": "gamma",
    "learning_starts": "learning_starts",
    "target_update_interval": "target_update_interval",
    "train_freq": "train_freq",
    "gradient_steps": "gradient_steps",
    "exploration_fraction": "exploration_fraction",
    "exploration_initial_eps": "exploration_initial_eps",
    "exploration_final_eps": "exploration_final_eps",
}


def select_kwargs(cfg: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Pull SB3 constructor kwargs from the config via a config_key->sb3_key map."""
    return {sb3_key: cfg[cfg_key]
            for cfg_key, sb3_key in mapping.items() if cfg_key in cfg}


def _require_keys(cfg: dict[str, Any]) -> None:
    for key in ("config_id", "net_arch", "env_steps"):
        if key not in cfg:
            raise ValueError(f"config missing required key: {key!r}")


def _check_activation(cfg: dict[str, Any]) -> None:
    act = cfg.get("activation_fn")
    if act is not None and act not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"activation_fn {act!r} not in supported {SUPPORTED_ACTIVATIONS}"
        )


def validate_actor_critic_config(cfg: dict[str, Any], expected_algo: str) -> None:
    """Validate a ppo/a2c config: algo, required keys, net_arch {pi, vf} dict."""
    if cfg.get("algo") != expected_algo:
        raise ValueError(f"expected algo: {expected_algo}, got {cfg.get('algo')!r}")
    _require_keys(cfg)
    net_arch = cfg["net_arch"]
    if not isinstance(net_arch, dict) or "pi" not in net_arch or "vf" not in net_arch:
        raise ValueError(
            f"{expected_algo} net_arch must be a dict with 'pi' and 'vf' lists, "
            f"got {net_arch!r}"
        )
    _check_activation(cfg)


def validate_value_based_config(cfg: dict[str, Any], expected_algo: str) -> None:
    """Validate a dqn-family config: algo, required keys, flat-list net_arch."""
    if cfg.get("algo") != expected_algo:
        raise ValueError(f"expected algo: {expected_algo}, got {cfg.get('algo')!r}")
    _require_keys(cfg)
    net_arch = cfg["net_arch"]
    if not isinstance(net_arch, list) or not all(isinstance(n, int) for n in net_arch):
        raise ValueError(
            f"{expected_algo} net_arch must be a flat list of ints, got {net_arch!r}"
        )
    _check_activation(cfg)


def build_actor_critic_policy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """policy_kwargs for ppo/a2c: {pi, vf} net_arch + activation + ortho_init."""
    policy_kwargs: dict[str, Any] = {
        "net_arch": {"pi": list(cfg["net_arch"]["pi"]),
                     "vf": list(cfg["net_arch"]["vf"])},
    }
    act = resolve_activation(cfg.get("activation_fn"))
    if act is not None:
        policy_kwargs["activation_fn"] = act
    if "ortho_init" in cfg:
        policy_kwargs["ortho_init"] = bool(cfg["ortho_init"])
    return policy_kwargs


def build_value_based_policy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """policy_kwargs for the dqn family: flat-list net_arch + optional activation."""
    policy_kwargs: dict[str, Any] = {"net_arch": list(cfg["net_arch"])}
    act = resolve_activation(cfg.get("activation_fn"))
    if act is not None:
        policy_kwargs["activation_fn"] = act
    return policy_kwargs


# ---------------------------------------------------------------------------
# W&B planning (decided without importing wandb)
# ---------------------------------------------------------------------------

def resolve_wandb_run_id(run_name: str, fresh: bool) -> str:
    """Deterministic id (== run_name) by default; timestamp-suffixed if ``fresh``."""
    if not fresh:
        return run_name
    return f"{run_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def plan_wandb(
    run_name: str,
    group: str,
    config: dict[str, Any],
    mode: str,
    fresh: bool,
) -> dict[str, Any]:
    """
    Decide all W&B behaviour without importing wandb. ``disabled`` skips
    wandb.init entirely; ``offline`` forces ``WANDB_MODE=offline`` so the server
    is never contacted; the deterministic id always carries ``resume="allow"`` so
    a re-run resumes the same run instead of erroring "run ID in use".
    """
    if mode == "disabled":
        return {"enabled": False, "env": {}, "init_kwargs": None, "run_id": None}

    env = {"WANDB_MODE": "offline"} if mode == "offline" else {}
    run_id = resolve_wandb_run_id(run_name, fresh)
    init_kwargs = {
        "project": WANDB_PROJECT,
        "name": run_name,
        "id": run_id,
        "resume": "allow",
        "group": group,
        "config": config,
        "sync_tensorboard": True,
        "mode": mode,
    }
    return {"enabled": True, "env": env, "init_kwargs": init_kwargs, "run_id": run_id}


# ---------------------------------------------------------------------------
# Checkpoint discovery / resume
# ---------------------------------------------------------------------------

def parse_checkpoint_steps(filename: str, run_name: str) -> Optional[int]:
    """Extract the step count from ``{run_name}_{steps}_steps.zip`` (else None)."""
    m = re.fullmatch(rf"{re.escape(run_name)}_(\d+)_steps\.zip", filename)
    return int(m.group(1)) if m else None


def find_latest_checkpoint(ckpt_dir: str | Path, run_name: str) -> Optional[Path]:
    """Return the highest-step model checkpoint for this run, or None."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for p in ckpt_dir.glob(f"{run_name}_*_steps.zip"):
        steps = parse_checkpoint_steps(p.name, run_name)
        if steps is not None and (best is None or steps > best[0]):
            best = (steps, p)
    return best[1] if best else None


def replay_buffer_path_for(checkpoint_path: Path, run_name: str) -> Optional[Path]:
    """
    Given a model checkpoint ``{run_name}_{steps}_steps.zip``, return the sibling
    replay-buffer path ``{run_name}_replay_buffer_{steps}_steps.pkl`` that
    CheckpointCallback(save_replay_buffer=True) writes (or None if it can't be
    derived). Used by off-policy (DQN) resume.
    """
    steps = parse_checkpoint_steps(checkpoint_path.name, run_name)
    if steps is None:
        return None
    return checkpoint_path.parent / f"{run_name}_replay_buffer_{steps}_steps.pkl"


# ---------------------------------------------------------------------------
# Env construction (shared; same env for both algorithms)
# ---------------------------------------------------------------------------

def make_vec_env(seed: int):
    """Monitor-wrapped, single-env DummyVecEnv training env, seeded."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from Environment.custom_env import ResidentialGridEnv
    from Training.seeds import seed_env

    def _factory():
        env = ResidentialGridEnv()        # T_max=150 default (experimental contract)
        env = seed_env(env, seed)
        return Monitor(env)

    vec_env = DummyVecEnv([_factory])
    vec_env.seed(seed)
    return vec_env


def raw_eval_env():
    """A raw (non-vectorised) eval env for RichEvalCallback."""
    from Environment.custom_env import ResidentialGridEnv
    return ResidentialGridEnv()


# ---------------------------------------------------------------------------
# CLI (shared by both trainers)
# ---------------------------------------------------------------------------

def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a YAML config (e.g. configs/dqn_exp5.yaml).")
    parser.add_argument("--seed", type=int, required=True,
                        help="Run seed (the only per-run-varying experimental input).")
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Root dir for checkpoints/ and tb/ (default: runs).")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="W&B mode. Use 'offline'/'disabled' for local smoke tests.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoints and start training from scratch.")
    parser.add_argument("--fresh", action="store_true",
                        help="Clean run: timestamp-suffix the W&B id (new run, never "
                             "resumes the existing one) and ignore checkpoints.")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override env_steps for a quick smoke test (debug only).")
    return parser


# ---------------------------------------------------------------------------
# Orchestration (identical for PPO and DQN; algo bits are injected)
# ---------------------------------------------------------------------------

def run_training(
    args: argparse.Namespace,
    *,
    cfg: dict[str, Any],
    build_model_fn: BuildModelFn,
    load_model_fn: LoadModelFn,
    save_replay_buffer: bool = False,
    tag: str = "trainer",
) -> None:
    """
    Run one ``(algo, config_id, seed)`` training job end to end: seed → W&B plan
    → env → resume-or-build → callbacks (checkpoint + wandb + RichEvalCallback) →
    learn → save final → per-seed metrics CSV. ``cfg`` must already be validated
    by the caller.
    """
    from Training.metrics import RichEvalCallback, finalize_run_csv

    spec = RunSpec(algo=cfg["algo"], config_id=cfg["config_id"], seed=args.seed)
    run_name = spec.run_name
    total_steps = int(args.total_steps) if args.total_steps else int(cfg["env_steps"])

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    tb_dir = out_dir / "tb"
    csv_dir = out_dir / CSV_SUBDIR     # CSVs live with checkpoints (persist on Drive)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(spec.seed)
    print(f"[{tag}] run_name = {run_name}")
    print(f"[{tag}] total_steps = {total_steps:,}  (config env_steps={cfg['env_steps']:,})")

    # ---- Weights & Biases -------------------------------------------------
    plan = plan_wandb(
        run_name=run_name,
        group=cfg["config_id"],
        config={**cfg, "seed": spec.seed, "total_steps": total_steps},
        mode=args.wandb_mode,
        fresh=args.fresh,
    )
    wandb_run = None
    wandb_callback = None
    if plan["enabled"]:
        for key, value in plan["env"].items():
            os.environ[key] = value          # set BEFORE importing/initing wandb
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb_run = wandb.init(**plan["init_kwargs"])
        wandb_callback = WandbCallback(verbose=1)
        print(f"[{tag}] wandb: mode={args.wandb_mode} id={plan['run_id']} resume=allow")
    else:
        print(f"[{tag}] wandb disabled (--wandb-mode disabled): skipping wandb.init")

    # ---- Env --------------------------------------------------------------
    env = make_vec_env(spec.seed)

    # ---- Model: resume from latest checkpoint, or build fresh -------------
    resume_checkpoints = not (args.no_resume or args.fresh)
    if args.fresh:
        print(f"[{tag}] --fresh: new wandb id and ignoring existing checkpoints")
    latest = find_latest_checkpoint(ckpt_dir, run_name) if resume_checkpoints else None
    if latest is not None:
        print(f"[{tag}] resuming from checkpoint: {latest.name}")
        model = load_model_fn(latest, env, str(tb_dir), spec.seed, run_name)
    else:
        print(f"[{tag}] starting a fresh run (no checkpoint found)")
        model = build_model_fn(cfg, env, spec.seed, str(tb_dir))

    remaining = total_steps - model.num_timesteps
    if remaining <= 0:
        print(f"[{tag}] already at {model.num_timesteps:,} >= {total_steps:,} steps; "
              "saving final and exiting.")
        model.save(ckpt_dir / f"{run_name}_final")
        if wandb_run is not None:
            wandb_run.finish()
        return

    # ---- Callbacks --------------------------------------------------------
    from stable_baselines3.common.callbacks import CheckpointCallback

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY_STEPS,          # 1 env => every 25k env steps
        save_path=str(ckpt_dir),
        name_prefix=run_name,
        save_replay_buffer=save_replay_buffer,     # True for off-policy (DQN) resume
    )
    callbacks: list[Any] = [checkpoint_cb]
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    rich_eval_cb = None
    if "eval_freq" in cfg:
        rich_eval_cb = RichEvalCallback(
            eval_env_fn=raw_eval_env,
            eval_freq=int(cfg["eval_freq"]),
            n_eval_episodes=int(cfg.get("eval_episodes", 15)),
            best_model_path=str(ckpt_dir / f"{run_name}_best"),
            verbose=1,
        )
        callbacks.append(rich_eval_cb)

    # ---- Train (no early-stopping: fixed step budget for Phase 1) ----------
    print(f"[{tag}] training for {remaining:,} more steps "
          f"(from {model.num_timesteps:,} to {total_steps:,})")
    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name=run_name,
        progress_bar=True,
    )

    # ---- Save final + per-seed metrics CSV --------------------------------
    final_path = ckpt_dir / f"{run_name}_final"
    model.save(final_path)
    print(f"[{tag}] saved final model -> {final_path}.zip")

    if rich_eval_cb is not None and rich_eval_cb.last_agg:
        # Under --output-dir (Drive on Colab), NOT repo-relative, so it survives
        # disconnects. finalize_run_csv mkdir's the parent and upserts by
        # (config_id, seed), so re-runs/resumes overwrite rather than duplicate.
        csv_path = csv_dir / f"{PHASE}_{cfg['config_id']}.csv"
        finalize_run_csv(
            csv_path,
            phase=PHASE,
            config_id=cfg["config_id"],
            algo=cfg["algo"],
            seed=spec.seed,
            env_steps=int(model.num_timesteps),
            callback=rich_eval_cb,
            wandb_run_name=run_name,
        )
        print(f"[{tag}] wrote per-seed metrics row -> {csv_path}")

    if wandb_run is not None:
        wandb_run.finish()
