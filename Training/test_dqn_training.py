"""
Tests for the import-safe helpers in Training/dqn_training.py (no SB3/Gymnasium).
The training loop is exercised by the step-3 GATE smoke run on Colab.

Run with:  pytest Training/test_dqn_training.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Training.dqn_training import (
    build_policy_kwargs,
    extract_dqn_kwargs,
    load_config,
    validate_dqn_config,
)
from Training.trainer_common import replay_buffer_path_for

REPO = Path(__file__).resolve().parents[1]
DQN_CFGS = sorted((REPO / "configs").glob("dqn_exp*.yaml"))


# ---------------------------------------------------------------------------
# Config loading & validation against the real dqn_exp*.yaml
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_path", DQN_CFGS, ids=lambda p: p.stem)
def test_every_dqn_config_loads(cfg_path):
    cfg = load_config(cfg_path)
    assert cfg["algo"] == "dqn"
    assert isinstance(cfg["net_arch"], list)        # flat list, not pi/vf dict


def test_loads_exp5_values():
    cfg = load_config(REPO / "configs" / "dqn_exp5.yaml")
    assert cfg["net_arch"] == [512, 256]
    assert cfg["gamma"] == 0.990
    assert cfg["buffer_size"] == 100000


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(REPO / "configs" / "nope.yaml")


def test_rejects_non_dqn_algo():
    with pytest.raises(ValueError, match="algo: dqn"):
        validate_dqn_config({"algo": "ppo", "config_id": "x",
                             "net_arch": [256, 256], "env_steps": 1})


def test_rejects_pi_vf_dict_net_arch():
    # the PPO-style dict must be rejected for DQN
    with pytest.raises(ValueError, match="flat list"):
        validate_dqn_config({"algo": "dqn", "config_id": "x",
                             "net_arch": {"pi": [256], "vf": [256]}, "env_steps": 1})


@pytest.mark.parametrize("missing", ["config_id", "net_arch", "env_steps"])
def test_rejects_missing_required_key(missing):
    cfg = {"algo": "dqn", "config_id": "x", "net_arch": [256, 256], "env_steps": 1}
    cfg.pop(missing)
    with pytest.raises(ValueError, match="required key"):
        validate_dqn_config(cfg)


# ---------------------------------------------------------------------------
# DQN kwarg extraction & policy_kwargs
# ---------------------------------------------------------------------------

def test_extract_dqn_kwargs_maps_and_filters():
    cfg = load_config(REPO / "configs" / "dqn_exp5.yaml")
    kw = extract_dqn_kwargs(cfg)
    assert kw["learning_rate"] == 0.0003           # lr -> learning_rate
    assert kw["buffer_size"] == 100000
    assert kw["exploration_fraction"] == 0.15
    assert kw["exploration_final_eps"] == 0.02
    assert kw["target_update_interval"] == 1000
    assert kw["train_freq"] == 4
    assert kw["gradient_steps"] == 1
    # config-only keys must not leak into the DQN constructor kwargs
    for leaked in ("net_arch", "env_steps", "n_seeds", "eval_freq",
                   "eval_episodes", "policy", "config_id", "algo"):
        assert leaked not in kw


def test_build_policy_kwargs_flat_list_no_activation():
    cfg = load_config(REPO / "configs" / "dqn_exp5.yaml")
    pk = build_policy_kwargs(cfg)
    assert pk["net_arch"] == [512, 256]
    # dqn_exp5 leaves activation unset -> SB3 default -> not in policy_kwargs
    assert "activation_fn" not in pk


def test_build_policy_kwargs_with_activation():
    pk = build_policy_kwargs({"net_arch": [256, 256], "activation_fn": "ReLU"})
    import torch.nn as nn
    assert pk["activation_fn"] is nn.ReLU


# ---------------------------------------------------------------------------
# Replay-buffer path derivation (off-policy resume)
# ---------------------------------------------------------------------------

def test_replay_buffer_path_derivation():
    rn = "dqn_dqn_exp5_seed0"
    ckpt = Path("runs/checkpoints") / f"{rn}_50000_steps.zip"
    buf = replay_buffer_path_for(ckpt, rn)
    assert buf is not None
    assert buf.name == f"{rn}_replay_buffer_50000_steps.pkl"
    assert buf.parent == ckpt.parent


def test_replay_buffer_path_none_for_nonmatching():
    assert replay_buffer_path_for(Path("dqn_dqn_exp5_seed0_final.zip"),
                                  "dqn_dqn_exp5_seed0") is None
