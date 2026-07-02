"""
Tests for Environment/obs_variants.py — the Phase 2 POMDP observation wrappers.

Verifies the pre-registered rungs (docs/PHASE2_POMDP_PREREGISTRATION.md §1, the
"keep 14-D/13-D labels, pure removal, no direction vector" resolution):
  * output shapes are 14-D (A-MILD) and 13-D (A-STRICT),
  * removed dims are GENUINELY ABSENT (re-indexed out, not zeroed-in-place),
  * retained dims map to their original base values in order.

Run with:  pytest Environment/test_obs_variants.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from Environment.custom_env import ResidentialGridEnv
from Environment.obs_variants import (
    BASE_OBS_DIM,
    AMildObs,
    AStrictObs,
    OBS_VARIANTS,
)

# A sentinel base observation whose every dim's VALUE == its INDEX. This lets a
# test assert exactly which original dims survived: value v present <=> dim v kept.
SENTINEL = np.arange(BASE_OBS_DIM, dtype=np.float32)

# Expected kept-dim indices per rung (pure removal, order preserved).
MILD_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]        # 14 dims (no 9,10)
STRICT_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]          # 13 dims (no 9,10,11)


@pytest.fixture
def mild():
    return AMildObs(ResidentialGridEnv())


@pytest.fixture
def strict():
    return AStrictObs(ResidentialGridEnv())


# ---------------------------------------------------------------------------
# Observation-space shape
# ---------------------------------------------------------------------------

def test_amild_space_is_14d(mild):
    assert mild.observation_space.shape == (14,)
    assert mild.observation_space.dtype == np.float32


def test_astrict_space_is_13d(strict):
    assert strict.observation_space.shape == (13,)
    assert strict.observation_space.dtype == np.float32


def test_space_bounds_sliced_from_base(mild):
    # bounds are still the base [0,1] Box bounds, just re-indexed (not invented).
    assert np.all(mild.observation_space.low == 0.0)
    assert np.all(mild.observation_space.high == 1.0)


# ---------------------------------------------------------------------------
# Removed dims are genuinely absent (re-indexed), retained dims map correctly
# ---------------------------------------------------------------------------

def test_amild_reindexes_and_drops_position(mild):
    out = mild.observation(SENTINEL)
    # exact re-indexed vector: original values at kept indices, in order.
    assert np.array_equal(out, SENTINEL[MILD_KEEP])
    assert out.shape == (14,)
    # dims 9 and 10 GONE (their sentinel values 9.0/10.0 must not appear anywhere,
    # which they only could if the dim were kept — proves absence, not zeroing).
    assert 9.0 not in out and 10.0 not in out
    # distance-to-target (dim 11, value 11.0) is RETAINED in A-MILD.
    assert 11.0 in out
    # proximity block (dims 0-4) still at the front, unchanged.
    assert np.array_equal(out[:5], SENTINEL[:5])


def test_astrict_reindexes_and_drops_position_and_distance(strict):
    out = strict.observation(SENTINEL)
    assert np.array_equal(out, SENTINEL[STRICT_KEEP])
    assert out.shape == (13,)
    # dims 9, 10 AND 11 all gone.
    for absent in (9.0, 10.0, 11.0):
        assert absent not in out
    # remaining-time (dim 12) and region one-hot (dims 13-15) retained at the tail.
    assert np.array_equal(out[-4:], SENTINEL[[12, 13, 14, 15]])
    assert np.array_equal(out[:5], SENTINEL[:5])


def test_not_zeroed_in_place(mild, strict):
    # A zeroed-but-present impl would keep length 16 with 0s at 9,10(,11); a
    # genuine removal shortens the vector. Assert the length actually shrank.
    assert mild.observation(SENTINEL).size == BASE_OBS_DIM - 2
    assert strict.observation(SENTINEL).size == BASE_OBS_DIM - 3


# ---------------------------------------------------------------------------
# Integration: the wrapped env emits conforming observations on reset/step
# ---------------------------------------------------------------------------

def test_wrapped_env_reset_matches_base_with_dims_removed():
    # Same seed -> identical underlying observation; the wrapper output must be
    # exactly the base obs with the rung's dims deleted (retained dims map to the
    # SAME real values, not just the sentinel).
    base_obs, _ = ResidentialGridEnv().reset(seed=7)
    mild_obs, _ = AMildObs(ResidentialGridEnv()).reset(seed=7)
    strict_obs, _ = AStrictObs(ResidentialGridEnv()).reset(seed=7)
    assert np.allclose(mild_obs, np.delete(base_obs, [9, 10]))
    assert np.allclose(strict_obs, np.delete(base_obs, [9, 10, 11]))
    assert mild_obs.shape == (14,) and strict_obs.shape == (13,)


@pytest.mark.parametrize("wrapper,dim", [(AMildObs, 14), (AStrictObs, 13)])
def test_step_obs_in_space(wrapper, dim):
    env = wrapper(ResidentialGridEnv())
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs.shape == (dim,)
    assert env.observation_space.contains(obs)


def test_registry_maps_rung_keys():
    assert OBS_VARIANTS["mild"] is AMildObs
    assert OBS_VARIANTS["strict"] is AStrictObs
