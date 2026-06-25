"""
Training/baselines/double_dqn.py — Phase 1 Double DQN baseline trainer.
=======================================================================
SB3's core ``DQN`` is VANILLA: its TD target is ``r + gamma * max_a Q_target(s', a)``
(a single-network max), NOT Double-Q. So plain DQN (configs/dqn_exp5.yaml) and
this baseline are genuinely different algorithms, not duplicates.

``DoubleDQN`` overrides only the target computation to decouple action SELECTION
(online net) from action EVALUATION (target net):

    a*      = argmax_a Q_online(s', a)
    target  = r + gamma * Q_target(s', a*)

Everything else (architecture, replay, exploration) is inherited from SB3 DQN and
matched to DQN Exp 5 via configs/double_dqn.yaml.

    python -m Training.baselines.double_dqn --config configs/double_dqn.yaml --seed 0

NOTE: this subclass is coupled to SB3's DQN.train() internals (SB3 >= 2.0). It
cannot be exercised without SB3 installed, so it MUST pass a Colab smoke run
before the sweep (Stage 5 pre-launch checklist).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Training.trainer_common import (
    DQN_KWARG_MAP,
    build_arg_parser,
    build_value_based_policy_kwargs,
    parse_config_file,
    replay_buffer_path_for,
    run_training,
    select_kwargs,
    validate_value_based_config,
)

# The genuine Double-DQN class is defined at MODULE level (so SB3 save/load can
# reference it by qualified name) but guarded so this module still imports
# without SB3 — config-helper tests and --help do not need it.
try:
    import numpy as np
    import torch as th
    from torch.nn import functional as F
    from stable_baselines3 import DQN

    class DoubleDQN(DQN):
        """SB3 DQN with the Double-Q target (selection by online net)."""

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            # Mirrors SB3 DQN.train() exactly except the target-Q block.
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            losses = []
            for _ in range(gradient_steps):
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )
                with th.no_grad():
                    # --- Double-Q target (the only change vs vanilla DQN) ---
                    next_actions = self.q_net(replay_data.next_observations).argmax(
                        dim=1, keepdim=True
                    )
                    next_q_values = self.q_net_target(
                        replay_data.next_observations
                    ).gather(1, next_actions)
                    target_q_values = replay_data.rewards + (
                        1 - replay_data.dones
                    ) * self.gamma * next_q_values

                current_q_values = self.q_net(replay_data.observations)
                current_q_values = th.gather(
                    current_q_values, dim=1, index=replay_data.actions.long()
                )
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                losses.append(loss.item())

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates,
                               exclude="tensorboard")
            self.logger.record("train/loss", float(np.mean(losses)))

except ImportError:  # pragma: no cover - import-safe without SB3
    DoubleDQN = None  # type: ignore[assignment, misc]


def validate_double_dqn_config(cfg: dict[str, Any]) -> None:
    validate_value_based_config(cfg, "double_dqn")


def load_config(config_path) -> dict[str, Any]:
    cfg = parse_config_file(config_path)
    validate_double_dqn_config(cfg)
    return cfg


def extract_double_dqn_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return select_kwargs(cfg, DQN_KWARG_MAP)


def build_double_dqn_model(cfg: dict[str, Any], env, seed: int, tensorboard_log: str):
    if DoubleDQN is None:  # pragma: no cover
        raise ImportError("stable_baselines3 is required to build DoubleDQN")
    return DoubleDQN(
        policy=cfg.get("policy", "MlpPolicy"),
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=build_value_based_policy_kwargs(cfg),
        **extract_double_dqn_kwargs(cfg),
    )


def load_double_dqn_model(latest_path, env, tensorboard_log: str, seed: int,
                          run_name: str):
    if DoubleDQN is None:  # pragma: no cover
        raise ImportError("stable_baselines3 is required to load DoubleDQN")
    model = DoubleDQN.load(latest_path, env=env, tensorboard_log=tensorboard_log)
    model.set_random_seed(seed)
    buffer_path = replay_buffer_path_for(Path(latest_path), run_name)
    if buffer_path is not None and buffer_path.is_file():
        model.load_replay_buffer(buffer_path)
        print(f"[double_dqn] restored replay buffer: {buffer_path.name}")
    else:
        print("[double_dqn] no replay buffer found; resuming with an empty buffer")
    return model


def train(args) -> None:
    cfg = load_config(args.config)
    run_training(
        args,
        cfg=cfg,
        build_model_fn=build_double_dqn_model,
        load_model_fn=load_double_dqn_model,
        save_replay_buffer=True,    # off-policy: persist buffer for resume
        tag="double_dqn",
    )


if __name__ == "__main__":
    parser = build_arg_parser(
        "Phase 1 Double DQN baseline trainer (config-driven, single seed per run)."
    )
    train(parser.parse_args())
