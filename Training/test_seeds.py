"""
Tests for Training/seeds.py — run identity and reproducible seeding.

Run with:  pytest Training/test_seeds.py -v
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from Training.seeds import (
    EXPLORATORY_SEEDS,
    MAIN_SEEDS,
    SANITY_SEEDS,
    SEED_TIERS,
    VALID_ALGOS,
    RunSpec,
    run_specs,
    seed_env,
    seed_everything,
)


# ---------------------------------------------------------------------------
# RunSpec identity & run_name
# ---------------------------------------------------------------------------

def test_run_name_format():
    spec = RunSpec("ppo", "ppo_exp4", 7)
    assert spec.run_name == "ppo_ppo_exp4_seed7"
    assert spec.checkpoint_stem() == "ppo_ppo_exp4_seed7"


def test_run_name_matches_claudemd_convention():
    # {algo}_{config_id}_seed{N}
    assert RunSpec("dqn", "dqn_exp5", 0).run_name == "dqn_dqn_exp5_seed0"


def test_runspec_is_frozen_and_hashable():
    spec = RunSpec("a2c", "a2c_base", 3)
    with pytest.raises((AttributeError, Exception)):
        spec.seed = 4  # type: ignore[misc]  frozen dataclass
    # usable as a set element / dict key (dedup a training queue)
    queue = {RunSpec("ppo", "ppo_exp4", 0), RunSpec("ppo", "ppo_exp4", 0)}
    assert len(queue) == 1


@pytest.mark.parametrize("algo", sorted(VALID_ALGOS))
def test_all_valid_algos_accepted(algo):
    assert RunSpec(algo, "cfg", 0).algo == algo


def test_invalid_algo_rejected():
    with pytest.raises(ValueError, match="not one of"):
        RunSpec("sac", "cfg", 0)


@pytest.mark.parametrize("bad_seed", [-1, -10])
def test_negative_seed_rejected(bad_seed):
    with pytest.raises(ValueError, match="non-negative"):
        RunSpec("ppo", "cfg", bad_seed)


def test_bool_seed_rejected():
    # bool is an int subclass; must not masquerade as a seed
    with pytest.raises(ValueError):
        RunSpec("ppo", "cfg", True)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_id", ["", "has space", "a/b", "a\\b"])
def test_bad_config_id_rejected(bad_id):
    with pytest.raises(ValueError):
        RunSpec("ppo", bad_id, 0)


# ---------------------------------------------------------------------------
# Seed tiers / run_specs
# ---------------------------------------------------------------------------

def test_seed_tier_sizes():
    assert MAIN_SEEDS == tuple(range(10))
    assert EXPLORATORY_SEEDS == tuple(range(5))
    assert SANITY_SEEDS == tuple(range(3))
    assert SEED_TIERS["main"] == MAIN_SEEDS


def test_run_specs_main_tier():
    specs = run_specs("dqn", "dqn_exp5", "main")
    assert len(specs) == 10
    assert [s.seed for s in specs] == list(range(10))
    assert all(s.algo == "dqn" and s.config_id == "dqn_exp5" for s in specs)
    assert specs[3].run_name == "dqn_dqn_exp5_seed3"


def test_run_specs_other_tiers():
    assert len(run_specs("ppo", "ppo_exp4", "exploratory")) == 5
    assert len(run_specs("ppo", "ppo_exp4", "sanity")) == 3


def test_run_specs_bad_tier():
    with pytest.raises(ValueError, match="tier"):
        run_specs("ppo", "ppo_exp4", "nonsense")


# ---------------------------------------------------------------------------
# seed_everything reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_reproduces_numpy_and_random():
    seed_everything(123)
    a_np, a_py = np.random.random(5), [random.random() for _ in range(5)]
    seed_everything(123)
    b_np, b_py = np.random.random(5), [random.random() for _ in range(5)]
    assert np.array_equal(a_np, b_np)
    assert a_py == b_py


def test_different_seeds_differ():
    seed_everything(1)
    a = np.random.random(5)
    seed_everything(2)
    b = np.random.random(5)
    assert not np.array_equal(a, b)


def test_seed_everything_returns_seed():
    assert seed_everything(42) == 42


def test_seed_everything_rejects_bad():
    with pytest.raises(ValueError):
        seed_everything(-1)
    with pytest.raises(ValueError):
        seed_everything(True)  # type: ignore[arg-type]


def test_seed_everything_seeds_torch_if_present():
    torch = pytest.importorskip("torch")
    seed_everything(7)
    a = torch.rand(4)
    seed_everything(7)
    b = torch.rand(4)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# seed_env (skips cleanly when gymnasium is absent)
# ---------------------------------------------------------------------------

def test_seed_env_makes_action_sampling_reproducible():
    gym = pytest.importorskip("gymnasium")
    from gymnasium import spaces

    class _Dummy:
        def __init__(self):
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.Box(0.0, 1.0, shape=(16,))

    e1, e2 = _Dummy(), _Dummy()
    seed_env(e1, 99)
    seed_env(e2, 99)
    assert e1.action_space.sample() == e2.action_space.sample()
    assert np.array_equal(e1.observation_space.sample(),
                          e2.observation_space.sample())
