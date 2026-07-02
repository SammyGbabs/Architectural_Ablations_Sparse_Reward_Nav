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


# Convenience registry (rung key -> wrapper class) for the Phase 2 trainer/sweep.
OBS_VARIANTS: dict[str, type[_RemoveDimsObs]] = {
    "mild": AMildObs,
    "strict": AStrictObs,
}


if __name__ == "__main__":  # pragma: no cover - smoke test
    from Environment.custom_env import ResidentialGridEnv

    base_env = ResidentialGridEnv()
    print(f"base observation_space: {base_env.observation_space}")
    for name, wrapper in OBS_VARIANTS.items():
        env = wrapper(ResidentialGridEnv())
        obs, _ = env.reset(seed=0)
        print(f"  {name:6s} -> {env.observation_space.shape}  "
              f"reset obs shape {obs.shape}  removed dims {wrapper.remove_dims}")
    print("[OK] obs_variants smoke test complete.")
