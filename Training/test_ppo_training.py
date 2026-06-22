"""
Tests for the import-safe helpers in Training/ppo_training.py.

These cover config loading/validation, checkpoint discovery, PPO-kwarg
extraction, and policy_kwargs assembly — i.e. everything that does NOT require
Stable-Baselines3 / Gymnasium / wandb. The training loop itself is exercised by
the step-4 smoke test on Colab, not here.

Run with:  pytest Training/test_ppo_training.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from Training.ppo_training import (
    build_policy_kwargs,
    extract_ppo_kwargs,
    find_latest_checkpoint,
    load_config,
    parse_checkpoint_steps,
    resolve_activation,
    validate_ppo_config,
)

REPO = Path(__file__).resolve().parents[1]
PPO_CFG = REPO / "configs" / "ppo_exp4.yaml"


# ---------------------------------------------------------------------------
# Config loading & validation against the real ppo_exp4.yaml
# ---------------------------------------------------------------------------

def test_loads_real_ppo_config():
    cfg = load_config(PPO_CFG)
    assert cfg["algo"] == "ppo"
    assert cfg["config_id"] == "ppo_exp4"
    assert cfg["net_arch"] == {"pi": [512, 256, 128], "vf": [256, 128]}


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(REPO / "configs" / "does_not_exist.yaml")


def test_rejects_non_ppo_algo():
    with pytest.raises(ValueError, match="algo: ppo"):
        validate_ppo_config({"algo": "dqn", "config_id": "x",
                             "net_arch": {"pi": [1], "vf": [1]}, "env_steps": 1})


def test_rejects_flat_net_arch():
    with pytest.raises(ValueError, match="net_arch must be a dict"):
        validate_ppo_config({"algo": "ppo", "config_id": "x",
                             "net_arch": [512, 256], "env_steps": 1})


def test_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation_fn"):
        validate_ppo_config({"algo": "ppo", "config_id": "x",
                             "net_arch": {"pi": [1], "vf": [1]},
                             "env_steps": 1, "activation_fn": "Swish"})


@pytest.mark.parametrize("missing", ["config_id", "net_arch", "env_steps"])
def test_rejects_missing_required_key(missing):
    cfg = {"algo": "ppo", "config_id": "x",
           "net_arch": {"pi": [1], "vf": [1]}, "env_steps": 1}
    cfg.pop(missing)
    with pytest.raises(ValueError, match="required key"):
        validate_ppo_config(cfg)


# ---------------------------------------------------------------------------
# PPO kwarg extraction
# ---------------------------------------------------------------------------

def test_extract_ppo_kwargs_maps_and_renames():
    cfg = load_config(PPO_CFG)
    kw = extract_ppo_kwargs(cfg)
    assert kw["learning_rate"] == 0.0003     # lr -> learning_rate
    assert kw["n_steps"] == 1024
    assert kw["gae_lambda"] == 0.92
    assert kw["ent_coef"] == 0.01
    # config-only keys must not leak into PPO constructor kwargs
    for leaked in ("net_arch", "activation_fn", "ortho_init", "env_steps",
                   "n_seeds", "eval_freq", "policy", "config_id", "algo"):
        assert leaked not in kw


# ---------------------------------------------------------------------------
# policy_kwargs assembly (uses torch, which is present)
# ---------------------------------------------------------------------------

def test_build_policy_kwargs_from_real_config():
    import torch.nn as nn
    cfg = load_config(PPO_CFG)
    pk = build_policy_kwargs(cfg)
    assert pk["net_arch"] == {"pi": [512, 256, 128], "vf": [256, 128]}
    assert pk["activation_fn"] is nn.LeakyReLU
    assert pk["ortho_init"] is True


def test_resolve_activation():
    import torch.nn as nn
    assert resolve_activation("ReLU") is nn.ReLU
    assert resolve_activation("LeakyReLU") is nn.LeakyReLU
    assert resolve_activation(None) is None
    with pytest.raises(ValueError):
        resolve_activation("Swish")


# ---------------------------------------------------------------------------
# Checkpoint discovery / resume
# ---------------------------------------------------------------------------

def test_parse_checkpoint_steps():
    rn = "ppo_ppo_exp4_seed0"
    assert parse_checkpoint_steps(f"{rn}_25000_steps.zip", rn) == 25000
    assert parse_checkpoint_steps(f"{rn}_200000_steps.zip", rn) == 200000
    assert parse_checkpoint_steps(f"{rn}_final.zip", rn) is None
    assert parse_checkpoint_steps("other_run_25000_steps.zip", rn) is None


def test_find_latest_checkpoint(tmp_path):
    rn = "ppo_ppo_exp4_seed0"
    for steps in (25000, 50000, 175000):
        (tmp_path / f"{rn}_{steps}_steps.zip").write_bytes(b"x")
    # decoys that must be ignored
    (tmp_path / f"{rn}_final.zip").write_bytes(b"x")
    (tmp_path / "ppo_ppo_exp4_seed1_300000_steps.zip").write_bytes(b"x")

    latest = find_latest_checkpoint(tmp_path, rn)
    assert latest is not None and latest.name == f"{rn}_175000_steps.zip"


def test_find_latest_checkpoint_none(tmp_path):
    assert find_latest_checkpoint(tmp_path, "ppo_ppo_exp4_seed0") is None
    assert find_latest_checkpoint(tmp_path / "nope", "ppo_ppo_exp4_seed0") is None
