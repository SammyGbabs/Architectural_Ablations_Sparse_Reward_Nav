"""
Training/baselines/a2c.py — Phase 1 A2C baseline trainer.
=========================================================
A2C is in core SB3 and is an on-policy actor-critic, so this trainer is the
direct analogue of ppo_training.py: same trainer_common machinery, the symmetric
[256,256] architecture matched to PPO Exp 1, and A2C-native hyperparameters from
configs/a2c.yaml. See that config / CLAUDE.md: A2C is a "standard alternative"
baseline (architecture-matched, algorithm-appropriate), not a controlled ablation.

    python -m Training.baselines.a2c --config configs/a2c.yaml --seed 0
"""

from __future__ import annotations

from typing import Any

from Training.trainer_common import (
    build_actor_critic_policy_kwargs,
    build_arg_parser,
    parse_config_file,
    run_training,
    select_kwargs,
    validate_actor_critic_config,
)

# A2C config keys -> SB3 A2C kwargs (only those A2C actually takes).
_A2C_KWARG_MAP: dict[str, str] = {
    "lr": "learning_rate",
    "gamma": "gamma",
    "n_steps": "n_steps",
    "gae_lambda": "gae_lambda",
    "ent_coef": "ent_coef",
    "vf_coef": "vf_coef",
}


def validate_a2c_config(cfg: dict[str, Any]) -> None:
    validate_actor_critic_config(cfg, "a2c")


def load_config(config_path) -> dict[str, Any]:
    cfg = parse_config_file(config_path)
    validate_a2c_config(cfg)
    return cfg


def extract_a2c_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return select_kwargs(cfg, _A2C_KWARG_MAP)


def build_a2c_model(cfg: dict[str, Any], env, seed: int, tensorboard_log: str):
    from stable_baselines3 import A2C

    return A2C(
        policy=cfg.get("policy", "MlpPolicy"),
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=build_actor_critic_policy_kwargs(cfg),
        **extract_a2c_kwargs(cfg),
    )


def load_a2c_model(latest_path, env, tensorboard_log: str, seed: int, run_name: str):
    from stable_baselines3 import A2C

    model = A2C.load(latest_path, env=env, tensorboard_log=tensorboard_log)
    model.set_random_seed(seed)
    return model


def train(args) -> None:
    cfg = load_config(args.config)
    run_training(
        args,
        cfg=cfg,
        build_model_fn=build_a2c_model,
        load_model_fn=load_a2c_model,
        save_replay_buffer=False,   # A2C is on-policy: no replay buffer
        tag="a2c",
    )


if __name__ == "__main__":
    parser = build_arg_parser(
        "Phase 1 A2C baseline trainer (config-driven, single seed per run)."
    )
    train(parser.parse_args())
