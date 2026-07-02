"""
Environment/obs_variants.py — Phase 2 POMDP observability-ladder wrappers.
=========================================================================
Two Gymnasium observation wrappers that implement the pre-registered rungs of
the Phase 2 observability ladder (docs/PHASE2_POMDP_PREREGISTRATION.md §1). Both
are **pure removals**: they drop dimensions from the frozen 16-D observation and
re-index the vector so the removed dims are *genuinely absent*, never
zeroed-in-place (a zeroed-but-present dim would still leak positional structure
as a constant input — see pre-reg §5.1).

This file NEVER imports or mutates the frozen ``custom_env.py``: it wraps the env
from the outside and only rewrites the emitted observation vector. The env's
internal state, reward, action space, spawn, and dynamics are untouched.

Rungs (pre-reg §1, resolved to the "keep labels, drop direction" reading —
the 2-D relative-direction feature is intentionally NOT added, so each rung is a
strict subset of the base observation):

    Rung 0  CONTROL   16-D   (base env; no wrapper — reuse Phase 1)
    Rung 1  A-MILD    14-D   remove dims 9-10 (normalised global position)
    Rung 2  A-STRICT  13-D   also remove dim 11 (distance-to-target)

Usage::

    from Environment.custom_env import ResidentialGridEnv
    from Environment.obs_variants import AMildObs, AStrictObs
    env = AStrictObs(ResidentialGridEnv())        # emits a 13-D observation
"""

from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Dimensionality of the frozen Phase 1 observation (custom_env.py / pre-reg §1).
BASE_OBS_DIM = 16

# Semantic index blocks of the base 16-D observation. Declared here purely to
# document *which* dims each rung removes; this module does not read the env's
# internals, it only slices the emitted vector by these indices.
IDX_PROXIMITY = (0, 1, 2, 3, 4)      # binary obstacle/off-grid sensors
IDX_TARGET_ONEHOT = (5, 6, 7, 8)     # target-room one-hot
IDX_POSITION = (9, 10)               # normalised global (x, y) — self-localisation
IDX_DISTANCE = (11,)                 # distance-to-target centroid — "the crutch"
IDX_TIME = (12,)                     # remaining-time fraction
IDX_REGION_ONEHOT = (13, 14, 15)     # in_room / in_hallway / in_doorway

# Rung -> dims removed (pre-registered, pure removal).
AMILD_REMOVE = IDX_POSITION                    # 16 - 2 -> 14-D
ASTRICT_REMOVE = IDX_POSITION + IDX_DISTANCE   # 16 - 3 -> 13-D
# Rung 4 (Amendment 3): A-STRICT and ALSO drop the region one-hot, so
# in-room / in-hallway / in-doorway cells emit identical observations (perceptual
# aliasing) and cannot be de-aliased from a single frame.
ALIAS_REMOVE = ASTRICT_REMOVE + IDX_REGION_ONEHOT   # 16 - 6 -> 10-D


class _RemoveDimsObs(gym.ObservationWrapper):
    """
    Base wrapper: emit the observation with a fixed set of dims removed and the
    remaining dims re-indexed (contiguous, order-preserving). Subclasses set
    ``remove_dims``. The ``observation_space`` is rebuilt by slicing the base
    Box's per-dim bounds, so bounds/dtype stay faithful to the source.
    """

    remove_dims: tuple[int, ...] = ()

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base = env.observation_space
        if not isinstance(base, spaces.Box) or base.shape != (BASE_OBS_DIM,):
            raise ValueError(
                f"{type(self).__name__} expects a Box shape ({BASE_OBS_DIM},) base "
                f"observation, got {base!r}"
            )
        remove = set(self.remove_dims)
        if not remove <= set(range(BASE_OBS_DIM)):
            raise ValueError(f"remove_dims {self.remove_dims} out of range "
                             f"[0,{BASE_OBS_DIM})")
        # Indices to KEEP, in ascending order -> genuine removal + re-indexing.
        self._keep = np.array([i for i in range(BASE_OBS_DIM) if i not in remove],
                              dtype=np.intp)
        self.observation_space = spaces.Box(
            low=np.asarray(base.low)[self._keep],
            high=np.asarray(base.high)[self._keep],
            shape=(int(self._keep.size),),
            dtype=base.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation)
        return obs[self._keep].astype(self.observation_space.dtype, copy=False)


class AMildObs(_RemoveDimsObs):
    """Rung 1 — A-MILD (14-D): remove normalised global position (dims 9-10).

    The agent loses global self-localisation but keeps distance-to-target
    (dim 11), so it still has a coarse "how far" signal."""

    remove_dims = AMILD_REMOVE


class AStrictObs(_RemoveDimsObs):
    """Rung 2 — A-STRICT (13-D): remove global position AND distance-to-target
    (dims 9-11).

    The agent knows which room is the target (one-hot), its immediate
    surroundings (proximity), region, and remaining time — but neither where it
    is globally nor how far the target is."""

    remove_dims = ASTRICT_REMOVE


class AAliasObs(_RemoveDimsObs):
    """Rung 4 — ALIASING (10-D): A-STRICT and also drop the region one-hot
    (dims 13-15), so in-room / in-hallway / in-doorway states are
    indistinguishable from a single frame (Amendment 3, Track-1-directed).

    Result: 10-D = proximity (5) + target one-hot (4) + remaining-time (1). The
    optimal action must use trajectory history to disambiguate confusable states
    — genuine policy-function complexity, not mere input deprivation."""

    remove_dims = ALIAS_REMOVE


# Convenience registry (rung key -> wrapper class) for the Phase 2 trainer/sweep.
OBS_VARIANTS: dict[str, type[_RemoveDimsObs]] = {
    "mild": AMildObs,
    "strict": AStrictObs,
    "alias": AAliasObs,
}


# ---------------------------------------------------------------------------
# Rung 3 (Amendment 2): temporal difficulty — flicker + frame-stack
# ---------------------------------------------------------------------------
# Rung 2 (A-STRICT) proved learnable but too easy: removing *static* features
# still leaves a reactively-solvable policy (proximity + region + target one-hot
# suffice per step). Amendment 2 adds temporal hidden state — the source of
# genuine policy-hardness in the POMDP literature — via:
#   * FLICKER (Hausknecht & Stone 2015): each step, with prob p, the frame is
#     fully zero-masked, forcing integration across time.
#   * FRAME-STACK k=4 (Mnih 2015): stack the last k emitted frames, turning
#     temporal integration into a static-but-complex function a feedforward MLP
#     can represent — so we test actor DEPTH (H1's knob), not recurrence.
# Rung 3a = A-STRICT -> frame-stack (52-D, no flicker).
# Rung 3b = A-STRICT -> flicker(p) -> frame-stack (52-D). Mask is applied BEFORE
# stacking, so masked frames enter the stack as zeros (pre-reg §Amendment 2).

FLICKER_P = 0.7          # per-step full-obscure probability (Amendment 2). Calibrated
                         # p=0.5 (p^k=6%, ceilinged) -> 0.8 (41%, degenerate near/far
                         # split) -> 0.7 (p^k~24%, between). See docs/results_log.md.
FRAME_STACK_K = 4        # frames stacked (Amendment 2 / Mnih 2015)
ASTRICT_DIM = BASE_OBS_DIM - len(ASTRICT_REMOVE)   # 13
RUNG3_DIM = ASTRICT_DIM * FRAME_STACK_K            # 52

# Rung 5 (Amendment 4): flip each binary proximity bit with probability q. The
# proximity block is the first 5 dims of BOTH the base and the A-STRICT vector.
PROX_NOISE_Q = 0.3       # per-bit flip probability (Amendment 4)
PROX_IDX = (0, 1, 2, 3, 4)


class FlickerObs(gym.Wrapper):
    """
    Flickering-POMDP wrapper (Hausknecht & Stone 2015). On each ``step`` the
    emitted observation is, with probability ``p``, replaced by an all-zeros
    vector of the same shape (full obscure); otherwise passed through unchanged.
    The mask decision uses a **dedicated Generator seeded from the episode's
    reset seed**, so the flicker sequence is reproducible per seed *without*
    consuming (and thus perturbing) the env's own RNG stream (which drives target
    -room sampling). The reset frame is never masked — the agent always sees a
    true first observation. ``info['flicker_masked']`` flags each masked step.

    Applied to A-STRICT and placed BEFORE frame-stacking, so masked frames enter
    the stack as zeros (pre-reg Amendment 2).
    """

    def __init__(self, env: gym.Env, p: float = FLICKER_P) -> None:
        super().__init__(env)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"flicker probability p must be in [0,1], got {p}")
        self.p = float(p)
        self._rng: np.random.Generator | None = None
        # observation_space is unchanged (same shape; zeros are within [0,1]).

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Seed the flicker stream deterministically from the run seed; if reset is
        # called without a seed (SB3 auto-reset between episodes), keep advancing
        # the existing stream so the whole run stays reproducible from its seed.
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()
        return obs, info                       # first frame is never flickered

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        masked = bool(self._rng.random() < self.p)
        if masked:
            obs = np.zeros_like(obs)
        info = {**info, "flicker_masked": masked}
        return obs, reward, terminated, truncated, info


class FrameStackObs(gym.Wrapper):
    """
    Stack the last ``k`` emitted frames into a single flat observation
    (``k * base_dim``), the standard memoryless approach to POMDPs (Mnih 2015).
    A manual deque-based wrapper (rather than SB3 ``VecFrameStack``) so it acts at
    the single-env level, composes with the flicker wrapper, is directly unit-
    testable, and yields a flat (k*d,) Box. At reset the stack is padded with
    ``k`` copies of the first observation (chosen over zeros so an episode does
    not open with spurious all-zero history). Concatenation order is oldest ->
    newest.
    """

    def __init__(self, env: gym.Env, k: int = FRAME_STACK_K) -> None:
        super().__init__(env)
        base = env.observation_space
        if not isinstance(base, spaces.Box) or len(base.shape) != 1:
            raise ValueError(f"FrameStackObs expects a 1-D Box, got {base!r}")
        self.k = int(k)
        self._d = int(base.shape[0])
        self.observation_space = spaces.Box(
            low=np.tile(np.asarray(base.low), self.k),
            high=np.tile(np.asarray(base.high), self.k),
            shape=(self._d * self.k,),
            dtype=base.dtype,
        )
        self._frames: deque = deque(maxlen=self.k)

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames)).astype(
            self.observation_space.dtype, copy=False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(np.asarray(obs))   # pad with the first frame
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.asarray(obs))
        return self._stacked(), reward, terminated, truncated, info


class AProxNoiseObs(gym.Wrapper):
    """
    Rung 5 — PROXIMITY NOISE (Amendment 4). Over a 13-D A-STRICT observation,
    flip each of the 5 (binary) proximity bits independently with probability q,
    every emitted observation (reset and step). Dimensionality is unchanged.

    Degrades STATE IDENTIFICATION (the agent can no longer cleanly resolve
    cell-type) while leaving enough signal to mostly avoid walls — noise, not
    masking, so collision-floor ("can't see walls to avoid dying") is separable
    from clean policy-hardness (report collision_rate). Reproducible via a
    dedicated per-episode Generator seeded from the reset seed (as with
    FlickerObs), so it does not perturb the env's own RNG stream.
    """

    def __init__(self, env: gym.Env, q: float = PROX_NOISE_Q,
                 prox_idx: tuple[int, ...] = PROX_IDX) -> None:
        super().__init__(env)
        base = env.observation_space
        if not isinstance(base, spaces.Box) or len(base.shape) != 1:
            raise ValueError(f"AProxNoiseObs expects a 1-D Box, got {base!r}")
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"proximity noise q must be in [0,1], got {q}")
        if max(prox_idx) >= base.shape[0]:
            raise ValueError(f"prox_idx {prox_idx} out of range for {base!r}")
        self.q = float(q)
        self._prox = np.asarray(prox_idx, dtype=np.intp)
        self._rng: np.random.Generator | None = None
        # observation_space unchanged: proximity is binary in [0,1], a flip
        # (1 - v) stays in [0,1].

    def _corrupt(self, obs: np.ndarray) -> np.ndarray:
        out = np.array(obs, dtype=self.observation_space.dtype, copy=True)
        flips = self._rng.random(self._prox.size) < self.q
        idx = self._prox[flips]
        out[idx] = 1.0 - out[idx]          # binary bit-flip on selected proximity dims
        return out

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()
        return self._corrupt(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._corrupt(obs), reward, terminated, truncated, info


def wrap_rung(env: gym.Env, rung: str, *, flicker_p: float = FLICKER_P,
              stack_k: int = FRAME_STACK_K, prox_q: float = PROX_NOISE_Q) -> gym.Env:
    """
    Compose the observation-wrapper chain for a ladder rung onto ``env`` (a raw
    ``ResidentialGridEnv``). Central place so the gate and the sweep build the
    exact same observation.

        mild    -> A-MILD (14-D)
        strict  -> A-STRICT (13-D)
        rung3a  -> A-STRICT -> frame-stack (52-D, no flicker)
        rung3b  -> A-STRICT -> flicker(p) -> frame-stack (52-D)
        rung4   -> A-STRICT - region one-hot = aliasing (10-D)
        rung5   -> A-STRICT + proximity-noise(q) (13-D)
    """
    if rung == "mild":
        return AMildObs(env)
    if rung == "strict":
        return AStrictObs(env)
    if rung == "rung3a":
        return FrameStackObs(AStrictObs(env), k=stack_k)
    if rung == "rung3b":
        return FrameStackObs(FlickerObs(AStrictObs(env), p=flicker_p), k=stack_k)
    if rung == "rung4":
        return AAliasObs(env)
    if rung == "rung5":
        return AProxNoiseObs(AStrictObs(env), q=prox_q)
    raise ValueError(f"unknown rung {rung!r}; expected one of "
                     "mild/strict/rung3a/rung3b/rung4/rung5")


if __name__ == "__main__":  # pragma: no cover - smoke test
    from Environment.custom_env import ResidentialGridEnv

    base_env = ResidentialGridEnv()
    print(f"base observation_space: {base_env.observation_space}")
    for name, wrapper in OBS_VARIANTS.items():
        env = wrapper(ResidentialGridEnv())
        obs, _ = env.reset(seed=0)
        print(f"  {name:6s} -> {env.observation_space.shape}  "
              f"reset obs shape {obs.shape}  removed dims {wrapper.remove_dims}")
    for rung in ("rung3a", "rung3b"):
        env = wrap_rung(ResidentialGridEnv(), rung)
        obs, _ = env.reset(seed=0)
        print(f"  {rung:6s} -> {env.observation_space.shape}  reset obs shape "
              f"{obs.shape}")
    print("[OK] obs_variants smoke test complete.")
