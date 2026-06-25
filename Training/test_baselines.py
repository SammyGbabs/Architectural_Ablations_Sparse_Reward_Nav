"""
Tests for the import-safe helpers of the three baseline trainers (no SB3).
The SB3-coupled model classes (DoubleDQN, DuelingDQNPolicy) are validated by the
Stage-5 Colab smoke run, not here.

Run with:  pytest Training/test_baselines.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Training.baselines import a2c, double_dqn, dueling_dqn

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs"


# ---------------------------------------------------------------------------
# A2C (actor-critic baseline)
# ---------------------------------------------------------------------------

def test_a2c_loads_and_validates():
    cfg = a2c.load_config(CFG / "a2c.yaml")
    assert cfg["algo"] == "a2c"
    assert cfg["net_arch"] == {"pi": [256, 256], "vf": [256, 256]}


def test_a2c_kwargs_are_native_not_ppo():
    cfg = a2c.load_config(CFG / "a2c.yaml")
    kw = a2c.extract_a2c_kwargs(cfg)
    assert kw["learning_rate"] == 0.0007     # SB3 A2C default, not PPO's 2e-4
    assert kw["gae_lambda"] == 1.0
    assert kw["ent_coef"] == 0.0
    assert kw["n_steps"] == 5
    # PPO-only keys must never appear
    for ppo_only in ("n_epochs", "clip_range", "batch_size"):
        assert ppo_only not in kw


def test_a2c_rejects_dqn_config():
    with pytest.raises(ValueError, match="algo: a2c"):
        a2c.validate_a2c_config(a2c.parse_config_file(CFG / "dqn_exp5.yaml"))


# ---------------------------------------------------------------------------
# Double DQN (value-based variant)
# ---------------------------------------------------------------------------

def test_double_dqn_loads_matched_to_exp5():
    cfg = double_dqn.load_config(CFG / "double_dqn.yaml")
    assert cfg["algo"] == "double_dqn"
    assert cfg["net_arch"] == [512, 256]          # matched to DQN Exp 5
    assert cfg["matched_to"] == "dqn_exp5"


def test_double_dqn_kwargs():
    cfg = double_dqn.load_config(CFG / "double_dqn.yaml")
    kw = double_dqn.extract_double_dqn_kwargs(cfg)
    assert kw["buffer_size"] == 100000
    assert kw["target_update_interval"] == 1000


def test_double_dqn_rejects_pi_vf_dict():
    with pytest.raises(ValueError, match="flat list"):
        double_dqn.validate_double_dqn_config(
            {"algo": "double_dqn", "config_id": "x",
             "net_arch": {"pi": [1], "vf": [1]}, "env_steps": 1})


# ---------------------------------------------------------------------------
# Dueling DQN (value-based variant)
# ---------------------------------------------------------------------------

def test_dueling_dqn_loads_matched_to_exp5():
    cfg = dueling_dqn.load_config(CFG / "dueling_dqn.yaml")
    assert cfg["algo"] == "dueling_dqn"
    assert cfg["net_arch"] == [512, 256]
    assert cfg["matched_to"] == "dqn_exp5"


def test_dueling_dqn_rejects_wrong_algo():
    with pytest.raises(ValueError, match="algo: dueling_dqn"):
        dueling_dqn.validate_dueling_dqn_config(
            dueling_dqn.parse_config_file(CFG / "double_dqn.yaml"))


def test_modules_import_safely():
    # The modules must import whether or not SB3 is usable (proven by the
    # module-level import at the top of this file succeeding without SB3). The
    # guarded class symbol is always DEFINED — the real class when SB3 imports
    # cleanly (Colab), else None. Its runtime behaviour is validated by the
    # Stage-5 Colab smoke run.
    assert hasattr(double_dqn, "DoubleDQN")
    assert hasattr(dueling_dqn, "DuelingDQNPolicy")
