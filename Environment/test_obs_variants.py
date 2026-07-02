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
    RUNG3_DIM,
    AMildObs,
    AStrictObs,
    AAliasObs,
    FlickerObs,
    FrameStackObs,
    OBS_VARIANTS,
    wrap_rung,
)

WAIT = 4  # action 4 = Wait (no move, never collides) -> safe for long step loops

# A sentinel base observation whose every dim's VALUE == its INDEX. This lets a
# test assert exactly which original dims survived: value v present <=> dim v kept.
SENTINEL = np.arange(BASE_OBS_DIM, dtype=np.float32)

# Expected kept-dim indices per rung (pure removal, order preserved).
MILD_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]        # 14 dims (no 9,10)
STRICT_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]          # 13 dims (no 9,10,11)
ALIAS_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12]                       # 10 dims (no 9,10,11,13,14,15)


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
    assert OBS_VARIANTS["alias"] is AAliasObs


# ---------------------------------------------------------------------------
# Rung 4 — aliasing (A-STRICT minus region one-hot) -> 10-D
# ---------------------------------------------------------------------------

def test_aalias_space_is_10d():
    env = AAliasObs(ResidentialGridEnv())
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32


def test_rung4_builder_is_10d():
    env = wrap_rung(ResidentialGridEnv(), "rung4")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape == (10,)
    assert obs.shape == (10,)


def test_aalias_reindexes_drops_position_distance_region():
    out = AAliasObs(ResidentialGridEnv()).observation(SENTINEL)
    assert np.array_equal(out, SENTINEL[ALIAS_KEEP])
    assert out.shape == (10,)
    # position (9,10), distance (11) AND region one-hot (13,14,15) all gone.
    for absent in (9.0, 10.0, 11.0, 13.0, 14.0, 15.0):
        assert absent not in out
    # proximity (0-4), target one-hot (5-8), remaining-time (12) retained, in order.
    assert np.array_equal(out[:9], SENTINEL[:9])         # proximity + target
    assert out[-1] == SENTINEL[12]                        # remaining-time last


def test_aalias_reset_matches_base_with_dims_removed():
    base_obs, _ = ResidentialGridEnv().reset(seed=7)
    alias_obs, _ = AAliasObs(ResidentialGridEnv()).reset(seed=7)
    assert np.allclose(alias_obs, np.delete(base_obs, [9, 10, 11, 13, 14, 15]))
    assert alias_obs.shape == (10,)


# ---------------------------------------------------------------------------
# Rung 3 — frame-stack (Rung 3a) and flicker+frame-stack (Rung 3b)
# ---------------------------------------------------------------------------

def _roll(env, n, seed=0):
    """Reset (seeded) and take n Wait steps; return list of (obs, info)."""
    out = []
    env.reset(seed=seed)
    for _ in range(n):
        obs, _, term, trunc, info = env.step(WAIT)
        out.append((obs, info))
        if term or trunc:
            break
    return out


def test_framestack_shape_is_52d():
    env = FrameStackObs(AStrictObs(ResidentialGridEnv()), k=4)
    assert env.observation_space.shape == (RUNG3_DIM,) == (52,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (52,)
    assert env.observation_space.contains(obs)


def test_rung3a_and_3b_builder_are_52d():
    for rung in ("rung3a", "rung3b"):
        env = wrap_rung(ResidentialGridEnv(), rung)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (52,)
        assert obs.shape == (52,)


def test_framestack_reset_pads_with_first_frame():
    # initial stacked obs = 4 copies of the A-STRICT reset frame.
    base_astrict, _ = AStrictObs(ResidentialGridEnv()).reset(seed=3)
    stacked, _ = FrameStackObs(AStrictObs(ResidentialGridEnv()), k=4).reset(seed=3)
    assert np.allclose(stacked, np.tile(base_astrict, 4))


def test_flicker_obs_space_unchanged():
    env = FlickerObs(AStrictObs(ResidentialGridEnv()), p=0.5)
    assert env.observation_space.shape == (13,)


def test_flicker_reset_frame_is_not_masked():
    # the first (reset) frame must be the true A-STRICT observation, never zeroed.
    base_astrict, _ = AStrictObs(ResidentialGridEnv()).reset(seed=1)
    flick_obs, _ = FlickerObs(AStrictObs(ResidentialGridEnv()), p=1.0).reset(seed=1)
    assert np.allclose(flick_obs, base_astrict)
    assert not np.all(flick_obs == 0.0)   # a real A-STRICT frame is not all-zero


def test_flicker_rate_is_approx_p_and_seed_deterministic():
    p, n = 0.5, 140
    env = FlickerObs(AStrictObs(ResidentialGridEnv()), p=p)
    rolled = _roll(env, n, seed=0)
    masks = [info["flicker_masked"] for _, info in rolled]
    # masked steps emit an all-zero 13-D frame; unmasked do not.
    for obs, info in rolled:
        assert np.all(obs == 0.0) == info["flicker_masked"]
    rate = sum(masks) / len(masks)
    assert 0.35 < rate < 0.65, f"flicker rate {rate:.2f} far from p={p}"
    # same seed -> identical mask sequence (reproducible).
    env2 = FlickerObs(AStrictObs(ResidentialGridEnv()), p=p)
    masks2 = [info["flicker_masked"] for _, info in _roll(env2, n, seed=0)]
    assert masks == masks2


def test_masked_frames_enter_stack_as_zeros():
    # p=1.0 masks every step; after k steps the whole stack is masked -> all zeros.
    env = FrameStackObs(FlickerObs(AStrictObs(ResidentialGridEnv()), p=1.0), k=4)
    env.reset(seed=0)
    obs = None
    for _ in range(4):
        obs, _, _, _, _ = env.step(WAIT)
    assert np.all(obs == 0.0)


def test_nonflickered_composition_equals_plain_stack():
    # p=0.0 flicker must be a no-op: stacked stream identical to the plain stack.
    a = FrameStackObs(FlickerObs(AStrictObs(ResidentialGridEnv()), p=0.0), k=4)
    b = FrameStackObs(AStrictObs(ResidentialGridEnv()), k=4)
    oa, _ = a.reset(seed=5)
    ob, _ = b.reset(seed=5)
    assert np.allclose(oa, ob)
    for _ in range(20):
        oa, _, ta, _, _ = a.step(WAIT)
        ob, _, tb, _, _ = b.step(WAIT)
        assert np.allclose(oa, ob)
        assert ta == tb
