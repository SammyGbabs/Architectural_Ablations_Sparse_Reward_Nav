"""
Training/ppo_training.py — Phase 1 PPO trainer (config-driven, multi-seed).
===========================================================================
Trains a single PPO run identified by ``(algo, config_id, seed)``:

    python -m Training.ppo_training --config configs/ppo_exp4.yaml --seed 0

What this script guarantees (CLAUDE.md § Conventions):
- **All hyperparameters come from the YAML config** — nothing is hardcoded here.
- **`--seed` is the only thing that varies per invocation** (plus output paths /
  debug toggles). Seeding goes through ``Training.seeds`` (the source of truth).
- **`MlpPolicy` with a `net_arch` dict** (flat 16-d Box obs => MlpPolicy), so the
  inverted-asymmetry ``pi``/``vf`` split in the config is honoured.
- **Logs to Weights & Biases**, project ``arch-ablations-sparse-reward``, run name
  ``{algo}_{config_id}_seed{N}``.
- **Checkpoints every 25k env steps** and **resumes from the latest checkpoint** —
  Colab sessions die; resume-from-checkpoint must work.

Heavy dependencies (Stable-Baselines3, Gymnasium, wandb, torch.nn) are imported
lazily inside functions so that ``--help`` and config parsing work even where the
full RL stack is not installed. The pure helpers below are import-safe and unit
-tested in Training/test_ppo_training.py.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from Training.seeds import RunSpec, seed_env, seed_everything

# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

WANDB_PROJECT = "arch-ablations-sparse-reward"
CHECKPOINT_EVERY_STEPS = 25_000  # CLAUDE.md: checkpoint every 25k env steps
PHASE = "p1"                     # results/csv/{phase}_{config_id}.csv
RESULTS_CSV_DIR = Path("results/csv")

# PPO constructor keys read from the config, mapped to SB3 PPO kwargs.
# (config_key -> sb3_kwarg). Anything not here stays at the SB3 default.
_PPO_KWARG_MAP: dict[str, str] = {
    "lr": "learning_rate",
    "n_steps": "n_steps",
    "batch_size": "batch_size",
    "n_epochs": "n_epochs",
    "gamma": "gamma",
    "gae_lambda": "gae_lambda",
    "clip_range": "clip_range",
    "ent_coef": "ent_coef",
}

# Activation-function names (config strings) -> resolver. Resolved lazily so the
# module imports without torch present in a docs-only environment.
_SUPPORTED_ACTIVATIONS = ("ReLU", "LeakyReLU", "Tanh", "ELU", "GELU")


# ---------------------------------------------------------------------------
# Pure, import-safe helpers (no SB3 / Gymnasium / wandb)
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and lightly validate a PPO YAML config."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    validate_ppo_config(cfg)
    return cfg


def validate_ppo_config(cfg: dict[str, Any]) -> None:
    """Raise if the config is not a usable PPO config (fail loud, not silent)."""
    if cfg.get("algo") != "ppo":
        raise ValueError(f"expected algo: ppo, got {cfg.get('algo')!r}")
    for key in ("config_id", "net_arch", "env_steps"):
        if key not in cfg:
            raise ValueError(f"config missing required key: {key!r}")
    net_arch = cfg["net_arch"]
    if not isinstance(net_arch, dict) or "pi" not in net_arch or "vf" not in net_arch:
        raise ValueError(
            "PPO net_arch must be a dict with 'pi' and 'vf' lists, got "
            f"{net_arch!r}"
        )
    act = cfg.get("activation_fn")
    if act is not None and act not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"activation_fn {act!r} not in supported {_SUPPORTED_ACTIVATIONS}"
        )


def extract_ppo_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the SB3 PPO constructor kwargs out of the config (no policy_kwargs)."""
    return {sb3_key: cfg[cfg_key]
            for cfg_key, sb3_key in _PPO_KWARG_MAP.items()
            if cfg_key in cfg}


def parse_checkpoint_steps(filename: str, run_name: str) -> Optional[int]:
    """
    Extract the timestep count from a CheckpointCallback file name, which is
    ``{run_name}_{steps}_steps.zip``. Returns None if it doesn't match.
    """
    m = re.fullmatch(rf"{re.escape(run_name)}_(\d+)_steps\.zip", filename)
    return int(m.group(1)) if m else None


def find_latest_checkpoint(ckpt_dir: str | Path, run_name: str) -> Optional[Path]:
    """Return the highest-timestep checkpoint for this run, or None."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for p in ckpt_dir.glob(f"{run_name}_*_steps.zip"):
        steps = parse_checkpoint_steps(p.name, run_name)
        if steps is not None and (best is None or steps > best[0]):
            best = (steps, p)
    return best[1] if best else None


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


def resolve_wandb_run_id(run_name: str, fresh: bool) -> str:
    """
    The W&B run id. Deterministic (== run_name) by default so a re-run resumes
    the same run; with ``fresh`` a timestamp suffix makes a brand-new run id.
    """
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
    Decide all W&B behaviour *without importing wandb*, so the decision logic is
    import-safe and unit-testable (and ``disabled`` never touches wandb at all).

    Returns a plan dict:
      - ``enabled``: False only for mode ``disabled`` (caller skips wandb.init).
      - ``env``: os.environ overrides to apply *before* wandb.init. For ``offline``
        this forces ``WANDB_MODE=offline`` so wandb can never reach the server,
        belt-and-suspenders with the ``mode`` kwarg below.
      - ``init_kwargs``: kwargs for ``wandb.init`` (None when disabled). Always
        carries ``resume="allow"`` alongside the deterministic ``id`` so a re-run
        of a seed resumes its existing run instead of erroring "run ID in use".
      - ``run_id``: the resolved id (None when disabled).
    """
    if mode == "disabled":
        return {"enabled": False, "env": {}, "init_kwargs": None, "run_id": None}

    env = {"WANDB_MODE": "offline"} if mode == "offline" else {}
    run_id = resolve_wandb_run_id(run_name, fresh)
    init_kwargs = {
        "project": WANDB_PROJECT,
        "name": run_name,
        "id": run_id,
        "resume": "allow",            # re-run resumes the same run, never errors
        "group": group,
        "config": config,
        "sync_tensorboard": True,     # SB3 tensorboard scalars flow into W&B
        "mode": mode,                 # "online" or "offline"
    }
    return {"enabled": True, "env": env, "init_kwargs": init_kwargs, "run_id": run_id}


def build_policy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Assemble SB3 ``policy_kwargs`` from the config (resolves activation)."""
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


# ---------------------------------------------------------------------------
# Env / model construction (lazy heavy imports)
# ---------------------------------------------------------------------------

def make_env(seed: int, *, monitor_path: Optional[str] = None):
    """Build the Monitor-wrapped, vectorised training env, seeded deterministically."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from Environment.custom_env import ResidentialGridEnv

    def _factory():
        env = ResidentialGridEnv()           # T_max=150 default (experimental contract)
        env = seed_env(env, seed)            # seed action/observation spaces
        return Monitor(env, filename=monitor_path)

    vec_env = DummyVecEnv([_factory])
    vec_env.seed(seed)
    return vec_env


def build_model(cfg: dict[str, Any], env, seed: int, tensorboard_log: str):
    """Construct a fresh PPO model from the config."""
    from stable_baselines3 import PPO

    model = PPO(
        policy=cfg.get("policy", "MlpPolicy"),
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=build_policy_kwargs(cfg),
        **extract_ppo_kwargs(cfg),
    )
    return model


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    spec = RunSpec(algo=cfg["algo"], config_id=cfg["config_id"], seed=args.seed)
    run_name = spec.run_name

    total_steps = int(args.total_steps) if args.total_steps else int(cfg["env_steps"])

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    tb_dir = out_dir / "tb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # ---- Reproducibility: seed global RNGs before building anything --------
    seed_everything(spec.seed)

    print(f"[ppo] run_name = {run_name}")
    print(f"[ppo] total_steps = {total_steps:,}  (config env_steps={cfg['env_steps']:,})")

    # ---- Weights & Biases -------------------------------------------------
    # Decide everything up front without importing wandb; 'disabled' skips it
    # entirely (works with wandb uninstalled); 'offline' forces WANDB_MODE so no
    # server is ever contacted.
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
            os.environ[key] = value          # must be set BEFORE importing/initing wandb
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb_run = wandb.init(**plan["init_kwargs"])
        wandb_callback = WandbCallback(verbose=1)
        print(f"[ppo] wandb: mode={args.wandb_mode} id={plan['run_id']} resume=allow")
    else:
        print("[ppo] wandb disabled (--wandb-mode disabled): skipping wandb.init")

    # ---- Env --------------------------------------------------------------
    env = make_env(spec.seed)

    # ---- Model: resume from latest checkpoint, or build fresh -------------
    from stable_baselines3 import PPO

    # --fresh implies a clean run: don't resume old checkpoints either.
    resume_checkpoints = not (args.no_resume or args.fresh)
    if args.fresh:
        print("[ppo] --fresh: new wandb id and ignoring existing checkpoints")
    latest = find_latest_checkpoint(ckpt_dir, run_name) if resume_checkpoints else None
    if latest is not None:
        print(f"[ppo] resuming from checkpoint: {latest.name}")
        model = PPO.load(latest, env=env, tensorboard_log=str(tb_dir))
        model.set_random_seed(spec.seed)
    else:
        print("[ppo] starting a fresh run (no checkpoint found)")
        model = build_model(cfg, env, spec.seed, str(tb_dir))

    remaining = total_steps - model.num_timesteps
    if remaining <= 0:
        print(f"[ppo] already at {model.num_timesteps:,} >= {total_steps:,} steps; "
              "saving final and exiting.")
        model.save(ckpt_dir / f"{run_name}_final")
        if wandb_run is not None:
            wandb_run.finish()
        return

    # ---- Callbacks --------------------------------------------------------
    from stable_baselines3.common.callbacks import CheckpointCallback

    from Training.metrics import RichEvalCallback, finalize_run_csv

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY_STEPS,          # 1 env => every 25k env steps
        save_path=str(ckpt_dir),
        name_prefix=run_name,
        save_replay_buffer=False,                  # PPO is on-policy: no buffer
    )
    callbacks = [checkpoint_cb]
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    # Rich deterministic evaluation: logs success/collision/wait-freq/per-room SR
    # and per-room episode length under eval/*, tracks per-eval IQM for sample
    # efficiency, and saves the best-by-IQM model. No early-stopping — Phase 1
    # compares at a FIXED step budget, so eval never cuts training short.
    rich_eval_cb = None
    if "eval_freq" in cfg:
        def _raw_eval_env():
            from Environment.custom_env import ResidentialGridEnv
            return ResidentialGridEnv()

        rich_eval_cb = RichEvalCallback(
            eval_env_fn=_raw_eval_env,
            eval_freq=int(cfg["eval_freq"]),
            n_eval_episodes=int(cfg.get("eval_episodes", 30)),
            best_model_path=str(ckpt_dir / f"{run_name}_best"),
            verbose=1,
        )
        callbacks.append(rich_eval_cb)

    # ---- Train ------------------------------------------------------------
    print(f"[ppo] training for {remaining:,} more steps "
          f"(from {model.num_timesteps:,} to {total_steps:,})")
    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=False,   # continue the global step counter on resume
        tb_log_name=run_name,
        progress_bar=True,
    )

    # ---- Save final model -------------------------------------------------
    final_path = ckpt_dir / f"{run_name}_final"
    model.save(final_path)
    print(f"[ppo] saved final model -> {final_path}.zip")

    # ---- Per-seed metrics CSV (for rliable IQM + CIs later) ---------------
    if rich_eval_cb is not None and rich_eval_cb.last_agg:
        csv_path = RESULTS_CSV_DIR / f"{PHASE}_{cfg['config_id']}.csv"
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
        print(f"[ppo] wrote per-seed metrics row -> {csv_path}")

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 PPO trainer (config-driven, single seed per run)."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a PPO YAML config (e.g. configs/ppo_exp4.yaml).")
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
                        help="Clean run: append a timestamp to the W&B id (new run, "
                             "never resumes the existing one) and ignore checkpoints.")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override env_steps for a quick smoke test (debug only).")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())