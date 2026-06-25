"""
Training/ppo_training.py — Phase 1 PPO trainer (config-driven, multi-seed).
===========================================================================
Thin PPO front-end over Training/trainer_common, which owns the shared W&B /
checkpoint / evaluation / CSV machinery (so PPO and DQN cannot drift). This file
holds only the PPO-specific pieces: config validation, policy_kwargs assembly
(the inverted-asymmetry pi/vf dict), and how to build/load the SB3 PPO model.

    python -m Training.ppo_training --config configs/ppo_exp4.yaml --seed 0

Heavy deps (SB3 / Gymnasium / wandb / torch.nn) are imported lazily so --help
and config parsing work without them. The pure helpers are unit-tested in
Training/test_ppo_training.py.
"""

from __future__ import annotations

from typing import Any

# Re-export the shared helpers so existing imports/tests against this module's
# namespace keep working, and so the trainer reads as a self-contained unit.
from Training.trainer_common import (  # noqa: F401  (re-exported)
    SUPPORTED_ACTIVATIONS,
    build_arg_parser,
    find_latest_checkpoint,
    parse_checkpoint_steps,
    parse_config_file,
    plan_wandb,
    resolve_activation,
    resolve_wandb_run_id,
    run_training,
)

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


# ---------------------------------------------------------------------------
# PPO-specific config handling (pure, import-safe)
# ---------------------------------------------------------------------------

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
    if act is not None and act not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"activation_fn {act!r} not in supported {SUPPORTED_ACTIVATIONS}"
        )


def load_config(config_path) -> dict[str, Any]:
    """Load and validate a PPO YAML config."""
    cfg = parse_config_file(config_path)
    validate_ppo_config(cfg)
    return cfg


def extract_ppo_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the SB3 PPO constructor kwargs out of the config (no policy_kwargs)."""
    return {sb3_key: cfg[cfg_key]
            for cfg_key, sb3_key in _PPO_KWARG_MAP.items()
            if cfg_key in cfg}


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
# PPO model build / load (lazy SB3 import)
# ---------------------------------------------------------------------------

def build_ppo_model(cfg: dict[str, Any], env, seed: int, tensorboard_log: str):
    """Construct a fresh PPO model from the config."""
    from stable_baselines3 import PPO

    return PPO(
        policy=cfg.get("policy", "MlpPolicy"),
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=build_policy_kwargs(cfg),
        **extract_ppo_kwargs(cfg),
    )


def load_ppo_model(latest_path, env, tensorboard_log: str, seed: int, run_name: str):
    """Resume a PPO model from a checkpoint (PPO is on-policy: no replay buffer)."""
    from stable_baselines3 import PPO

    model = PPO.load(latest_path, env=env, tensorboard_log=tensorboard_log)
    model.set_random_seed(seed)
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(args) -> None:
    cfg = load_config(args.config)
    run_training(
        args,
        cfg=cfg,
        build_model_fn=build_ppo_model,
        load_model_fn=load_ppo_model,
        save_replay_buffer=False,
        tag="ppo",
    )


if __name__ == "__main__":
    parser = build_arg_parser(
        "Phase 1 PPO trainer (config-driven, single seed per run)."
    )
    train(parser.parse_args())
