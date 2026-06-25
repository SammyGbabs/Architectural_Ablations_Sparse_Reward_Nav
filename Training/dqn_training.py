"""
Training/dqn_training.py — Phase 1 DQN trainer (config-driven, multi-seed).
===========================================================================
Thin DQN front-end over Training/trainer_common (the SAME shared W&B /
checkpoint / evaluation / CSV machinery the PPO trainer uses — they cannot
drift). This file holds only the DQN-specific pieces.

    python -m Training.dqn_training --config configs/dqn_exp5.yaml --seed 0

Differences from PPO:
- SB3 ``DQN`` instead of ``PPO``.
- ``net_arch`` is a plain list (DQN has no separate actor/critic, so no pi/vf
  dict).
- DQN-specific config fields (buffer_size, exploration_*, target_update_interval,
  train_freq, gradient_steps, learning_starts).
- Off-policy resume restores the replay buffer (CheckpointCallback saves it with
  ``save_replay_buffer=True``).

The legacy values that used to live here are now authoritative in
configs/dqn_exp*.yaml; nothing is hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Training.trainer_common import (  # noqa: F401  (re-exported for symmetry/tests)
    SUPPORTED_ACTIVATIONS,
    build_arg_parser,
    find_latest_checkpoint,
    parse_checkpoint_steps,
    parse_config_file,
    plan_wandb,
    replay_buffer_path_for,
    resolve_activation,
    resolve_wandb_run_id,
    run_training,
)

# DQN constructor keys read from the config, mapped to SB3 DQN kwargs.
# (config_key -> sb3_kwarg). Anything not here stays at the SB3 default.
_DQN_KWARG_MAP: dict[str, str] = {
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


# ---------------------------------------------------------------------------
# DQN-specific config handling (pure, import-safe)
# ---------------------------------------------------------------------------

def validate_dqn_config(cfg: dict[str, Any]) -> None:
    """Raise if the config is not a usable DQN config (fail loud, not silent)."""
    if cfg.get("algo") != "dqn":
        raise ValueError(f"expected algo: dqn, got {cfg.get('algo')!r}")
    for key in ("config_id", "net_arch", "env_steps"):
        if key not in cfg:
            raise ValueError(f"config missing required key: {key!r}")
    net_arch = cfg["net_arch"]
    # DQN has no actor/critic split: net_arch is a flat list of layer widths.
    if not isinstance(net_arch, list) or not all(isinstance(n, int) for n in net_arch):
        raise ValueError(
            f"DQN net_arch must be a flat list of ints, got {net_arch!r}"
        )
    act = cfg.get("activation_fn")
    if act is not None and act not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"activation_fn {act!r} not in supported {SUPPORTED_ACTIVATIONS}"
        )


def load_config(config_path) -> dict[str, Any]:
    """Load and validate a DQN YAML config."""
    cfg = parse_config_file(config_path)
    validate_dqn_config(cfg)
    return cfg


def extract_dqn_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the SB3 DQN constructor kwargs out of the config (no policy_kwargs)."""
    return {sb3_key: cfg[cfg_key]
            for cfg_key, sb3_key in _DQN_KWARG_MAP.items()
            if cfg_key in cfg}


def build_policy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Assemble SB3 ``policy_kwargs`` for DQN (flat net_arch list; no pi/vf)."""
    policy_kwargs: dict[str, Any] = {"net_arch": list(cfg["net_arch"])}
    act = resolve_activation(cfg.get("activation_fn"))
    if act is not None:
        policy_kwargs["activation_fn"] = act
    return policy_kwargs


# ---------------------------------------------------------------------------
# DQN model build / load (lazy SB3 import)
# ---------------------------------------------------------------------------

def build_dqn_model(cfg: dict[str, Any], env, seed: int, tensorboard_log: str):
    """Construct a fresh DQN model from the config."""
    from stable_baselines3 import DQN

    return DQN(
        policy=cfg.get("policy", "MlpPolicy"),
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=build_policy_kwargs(cfg),
        **extract_dqn_kwargs(cfg),
    )


def load_dqn_model(latest_path, env, tensorboard_log: str, seed: int, run_name: str):
    """
    Resume a DQN model from a checkpoint, restoring the replay buffer too (DQN is
    off-policy: resuming without the buffer would discard all collected
    experience). The buffer .pkl is the sibling CheckpointCallback wrote.
    """
    from stable_baselines3 import DQN

    model = DQN.load(latest_path, env=env, tensorboard_log=tensorboard_log)
    model.set_random_seed(seed)

    buffer_path = replay_buffer_path_for(Path(latest_path), run_name)
    if buffer_path is not None and buffer_path.is_file():
        model.load_replay_buffer(buffer_path)
        print(f"[dqn] restored replay buffer: {buffer_path.name} "
              f"({model.replay_buffer.size():,} transitions)")
    else:
        print("[dqn] no replay buffer found alongside checkpoint; "
              "resuming with an empty buffer")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(args) -> None:
    cfg = load_config(args.config)
    run_training(
        args,
        cfg=cfg,
        build_model_fn=build_dqn_model,
        load_model_fn=load_dqn_model,
        save_replay_buffer=True,   # off-policy: persist the buffer for resume
        tag="dqn",
    )


if __name__ == "__main__":
    parser = build_arg_parser(
        "Phase 1 DQN trainer (config-driven, single seed per run)."
    )
    train(parser.parse_args())
