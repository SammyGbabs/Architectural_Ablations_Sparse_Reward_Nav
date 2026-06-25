"""
Training/baselines/dueling_dqn.py — Phase 1 Dueling DQN baseline trainer.
=========================================================================
Dueling DQN is NOT available in core SB3 (not via policy_kwargs, not a flag): it
needs a custom Q-network with separate value V(s) and advantage A(s,a) streams,
recombined as

    Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

so it is a genuine architectural variant, not a duplicate of plain DQN. The
shared trunk uses the matched [512,256] arch (configs/dueling_dqn.yaml, matched
to DQN Exp 5); the otherwise-vanilla DQN training loop is inherited from SB3.

    python -m Training.baselines.dueling_dqn --config configs/dueling_dqn.yaml --seed 0

NOTE: the custom network/policy are coupled to SB3's QNetwork/DQNPolicy internals
(SB3 >= 2.0). They cannot be exercised without SB3 installed, so this MUST pass a
Colab smoke run before the sweep (Stage 5 pre-launch checklist).
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

# Custom dueling network + policy, defined at MODULE level (so SB3 save/load can
# reference DuelingDQNPolicy by qualified name) but guarded so the module imports
# without SB3 for config-helper tests and --help.
try:
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import create_mlp
    from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

    class DuelingQNetwork(QNetwork):
        """QNetwork with separate value and advantage streams."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # net_arch[:-1] forms the shared trunk; net_arch[-1] is the width
            # feeding the two heads.
            arch = list(self.net_arch) if self.net_arch else [64, 64]
            trunk = create_mlp(self.features_dim, arch[-1], arch[:-1],
                               self.activation_fn)
            self.shared = nn.Sequential(*trunk, self.activation_fn())
            self.value_head = nn.Linear(arch[-1], 1)
            self.advantage_head = nn.Linear(arch[-1], int(self.action_space.n))
            del self.q_net  # drop the inherited single-head net (unused)

        def forward(self, obs: th.Tensor) -> th.Tensor:
            features = self.extract_features(obs, self.features_extractor)
            z = self.shared(features)
            value = self.value_head(z)
            advantage = self.advantage_head(z)
            return value + advantage - advantage.mean(dim=1, keepdim=True)

    class DuelingDQNPolicy(DQNPolicy):
        """DQNPolicy whose q_net/q_net_target are DuelingQNetworks."""

        def make_q_net(self) -> "DuelingQNetwork":
            net_args = self._update_features_extractor(self.net_args,
                                                       features_extractor=None)
            return DuelingQNetwork(**net_args).to(self.device)

except ImportError:  # pragma: no cover - import-safe without SB3
    DuelingDQNPolicy = None  # type: ignore[assignment, misc]


def validate_dueling_dqn_config(cfg: dict[str, Any]) -> None:
    validate_value_based_config(cfg, "dueling_dqn")


def load_config(config_path) -> dict[str, Any]:
    cfg = parse_config_file(config_path)
    validate_dueling_dqn_config(cfg)
    return cfg


def extract_dueling_dqn_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return select_kwargs(cfg, DQN_KWARG_MAP)


def build_dueling_dqn_model(cfg: dict[str, Any], env, seed: int, tensorboard_log: str):
    if DuelingDQNPolicy is None:  # pragma: no cover
        raise ImportError("stable_baselines3 is required to build Dueling DQN")
    from stable_baselines3 import DQN

    return DQN(
        policy=DuelingDQNPolicy,           # custom dueling policy (not "MlpPolicy")
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=build_value_based_policy_kwargs(cfg),
        **extract_dueling_dqn_kwargs(cfg),
    )


def load_dueling_dqn_model(latest_path, env, tensorboard_log: str, seed: int,
                           run_name: str):
    from stable_baselines3 import DQN

    # The saved model records DuelingDQNPolicy as its policy_class; DQN.load
    # reconstructs it (the class is importable at module scope).
    model = DQN.load(latest_path, env=env, tensorboard_log=tensorboard_log)
    model.set_random_seed(seed)
    buffer_path = replay_buffer_path_for(Path(latest_path), run_name)
    if buffer_path is not None and buffer_path.is_file():
        model.load_replay_buffer(buffer_path)
        print(f"[dueling_dqn] restored replay buffer: {buffer_path.name}")
    else:
        print("[dueling_dqn] no replay buffer found; resuming with an empty buffer")
    return model


def train(args) -> None:
    cfg = load_config(args.config)
    run_training(
        args,
        cfg=cfg,
        build_model_fn=build_dueling_dqn_model,
        load_model_fn=load_dueling_dqn_model,
        save_replay_buffer=True,
        tag="dueling_dqn",
    )


if __name__ == "__main__":
    parser = build_arg_parser(
        "Phase 1 Dueling DQN baseline trainer (config-driven, single seed per run)."
    )
    train(parser.parse_args())
