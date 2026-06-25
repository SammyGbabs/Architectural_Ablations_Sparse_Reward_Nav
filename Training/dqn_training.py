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
    DQN_KWARG_MAP,
    SUPPORTED_ACTIVATIONS,
    build_value_based_policy_kwargs as build_policy_kwargs,
    build_arg_parser,
    find_latest_checkpoint,
    parse_checkpoint_steps,
    parse_config_file,
    plan_wandb,
    replay_buffer_path_for,
    resolve_activation,
    resolve_wandb_run_id,
    run_training,
    select_kwargs,
    validate_value_based_config,
)


# ---------------------------------------------------------------------------
# DQN-specific config handling (delegates to the shared value-based helpers)
# ---------------------------------------------------------------------------

def validate_dqn_config(cfg: dict[str, Any]) -> None:
    """Raise if the config is not a usable DQN config (fail loud, not silent)."""
    validate_value_based_config(cfg, "dqn")


def load_config(config_path) -> dict[str, Any]:
    """Load and validate a DQN YAML config."""
    cfg = parse_config_file(config_path)
    validate_dqn_config(cfg)
    return cfg


def extract_dqn_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the SB3 DQN constructor kwargs out of the config (no policy_kwargs)."""
    return select_kwargs(cfg, DQN_KWARG_MAP)


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
