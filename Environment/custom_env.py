"""
Environment/custom_env.py
=========================
Custom Gymnasium environment: 20x20 residential grid-world for assistive
indoor navigation. This file is aligned with the paper's specification:

    * Observation : Flat 16-d Box vector
        [proximity_sensors (5)] + [target_room_onehot (4)] + [nav_state (7)]
    * Action      : Discrete(5) — 0:Up, 1:Down, 2:Left, 3:Right, 4:Wait
    * Task        : Reach a uniformly sampled TARGET ROOM (not a cell)
    * Dynamics    : Attempting to step into an obstacle terminates the
                    episode with a collision penalty
    * Rewards     : -0.1 per step
                    -5.0 collision (terminates)
                    +1.0 first time a doorway cell is entered (per episode)
                    +15 + K * t_rem  on target-room entry (K = 0.1)
                    -3.0 timeout if T_max reached without success
    * Horizon     : T_max = 150 steps

Usage
-----
    >>> from Environment.custom_env import ResidentialGridEnv
    >>> env = ResidentialGridEnv()
    >>> obs, info = env.reset(seed=0)
    >>> obs, r, terminated, truncated, info = env.step(action=0)

Or via the Gymnasium registry:
    >>> import gymnasium as gym
    >>> from Environment import custom_env  # registers the env id
    >>> env = gym.make("ResidentialGrid-v0")

The map (rooms, doorways, obstacles, agent start) lives in
`Environment/rendering.py` so visualisation and RL dynamics cannot drift
apart — one source of truth.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Reuse the map data + render helpers from the rendering module. Relative
# import when used as a package; absolute fallback when run as a script.
try:
    from .rendering import (
        GRID_SIZE,
        AGENT_START,
        LIVING_ROOM, KITCHEN, BEDROOM, BATHROOM, HALLWAY,
        build_room_grid,
        get_doorways,
        get_obstacles,
        render_environment,
        render_rgb_array,
    )
except ImportError:
    from rendering import (  # type: ignore[no-redef]
        GRID_SIZE,
        AGENT_START,
        LIVING_ROOM, KITCHEN, BEDROOM, BATHROOM, HALLWAY,
        build_room_grid,
        get_doorways,
        get_obstacles,
        render_environment,
        render_rgb_array,
    )


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------
# Deltas are (drow, dcol). Row grows downward (north-up display convention).
# Action 4 (Wait) consumes a timestep without moving — included to match the
# paper's 5-action space and to allow future dynamic-obstacle extensions.
# ---------------------------------------------------------------------------
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1,  0),   # Up
    1: ( 1,  0),   # Down
    2: ( 0, -1),   # Left
    3: ( 0,  1),   # Right
    4: ( 0,  0),   # Wait
}
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Wait"]


# ---------------------------------------------------------------------------
# Target-room encoding
# ---------------------------------------------------------------------------
# The observation's 4-dim one-hot uses the fixed index order below; this is
# also the canonical order used when sampling a target room uniformly.
TARGET_ROOMS: tuple[int, ...] = (LIVING_ROOM, KITCHEN, BEDROOM, BATHROOM)
TARGET_ROOM_NAMES: dict[int, str] = {
    LIVING_ROOM: "living_room",
    KITCHEN:     "kitchen",
    BEDROOM:     "bedroom",
    BATHROOM:    "bathroom",
}


class ResidentialGridEnv(gym.Env):
    """
    20x20 residential grid-world with a 16-dimensional flat observation
    vector, 5 discrete actions, and a 5-component shaped reward.

    Observation layout (indices into the 16-d vector):

        0 .. 4   proximity sensors  (binary; 1 = obstacle OR boundary)
                 order: Up, Down, Left, Right, Current
        5 .. 8   target-room one-hot (order: living, kitchen, bedroom, bath)
        9 ..10   normalized (x, y) in [0, 1]
        11       Euclidean distance to target-room centroid, normalized
        12       remaining time fraction t_rem / T_max, in [0, 1]
        13..15   current-region one-hot: (in_a_room, in_hallway, in_doorway)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # ---- Reward components (exposed as class attributes for easy tuning) ---
    STEP_PENALTY      = -0.1
    COLLISION_PENALTY = -5.0
    DOORWAY_BONUS     =  1.0
    TARGET_BASE_BONUS = 15.0
    TARGET_K          =  0.1   # bonus per remaining step
    TIMEOUT_PENALTY   = -3.0

    DEFAULT_MAX_STEPS = 150    # T_max per paper §3.5

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        # --- Static map -------------------------------------------------
        self._room_grid = build_room_grid()
        self._doorways  = frozenset(get_doorways())
        self._obstacles = frozenset(o["pos"] for o in get_obstacles())

        # walkable[r, c] is True iff the cell is NOT an obstacle.
        self._walkable = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
        for (r, c) in self._obstacles:
            self._walkable[r, c] = False

        # Pre-compute each target room's cells + centroid (for the obs's
        # normalized-distance field and for rendering).
        self._room_cells: dict[int, np.ndarray] = {}
        self._room_centroids: dict[int, tuple[float, float]] = {}
        for rid in TARGET_ROOMS:
            cells = np.argwhere(self._room_grid == rid)
            self._room_cells[rid] = cells
            self._room_centroids[rid] = (
                float(cells[:, 0].mean()),
                float(cells[:, 1].mean()),
            )

        # Max possible distance (corner-to-corner) — normalizes obs[11].
        self._max_dist = float(np.hypot(GRID_SIZE - 1, GRID_SIZE - 1))

        # --- Spaces -----------------------------------------------------
        self.action_space = spaces.Discrete(5)
        # A uniform Box(0, 1, 16) is an honest spec: every dim is either
        # binary, one-hot, or already normalized into [0, 1].
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(16,), dtype=np.float32,
        )

        # --- Dynamic state (populated on reset) ------------------------
        self._agent_pos: tuple[int, int] = AGENT_START
        self._target_room: int = LIVING_ROOM
        self._steps: int = 0
        self._visited_doorways: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Start a new episode. The target room is sampled uniformly from
        {living, kitchen, bedroom, bathroom} each reset.
        """
        super().reset(seed=seed)
        self._agent_pos = AGENT_START
        self._steps = 0
        self._visited_doorways = set()
        self._target_room = int(TARGET_ROOMS[
            int(self.np_random.integers(0, len(TARGET_ROOMS)))
        ])
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply `action` and return the standard Gymnasium 5-tuple."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._steps += 1
        reward = self.STEP_PENALTY       # dense step cost (unconditional)
        terminated = False
        truncated = False
        collided = False
        passed_doorway = False
        reached_target = False

        dr, dc = ACTION_DELTAS[int(action)]
        r, c = self._agent_pos
        nr, nc = r + dr, c + dc

        # --- Resolve the move ------------------------------------------
        # Action 4 (Wait) has (dr, dc) == (0, 0) and therefore neither
        # collides nor moves; it only pays the step penalty above.
        if (dr, dc) != (0, 0):
            out_of_bounds = not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE)
            hits_obstacle = (not out_of_bounds) and (not self._walkable[nr, nc])

            if out_of_bounds:
                # Grid borders are walls, not collisions. Movement is
                # silently cancelled (agent stays put and pays step cost).
                pass
            elif hits_obstacle:
                # Strict paper semantics: obstacle contact = catastrophic
                # collision. Episode terminates immediately.
                reward += self.COLLISION_PENALTY
                terminated = True
                collided = True
                # Agent does NOT enter the obstacle cell (position unchanged).
            else:
                # Free move.
                self._agent_pos = (nr, nc)

                # Doorway shaping bonus — paid once per unique doorway cell
                # per episode (prevents reward farming by oscillation).
                if (self._agent_pos in self._doorways
                        and self._agent_pos not in self._visited_doorways):
                    reward += self.DOORWAY_BONUS
                    self._visited_doorways.add(self._agent_pos)
                    passed_doorway = True

                # Target completion: did the agent just enter the target room?
                if self._room_grid[self._agent_pos] == self._target_room:
                    t_rem = max(0, self.max_steps - self._steps)
                    reward += self.TARGET_BASE_BONUS + self.TARGET_K * t_rem
                    terminated = True
                    reached_target = True

        # --- Timeout ---------------------------------------------------
        if (not terminated) and (self._steps >= self.max_steps):
            reward += self.TIMEOUT_PENALTY
            truncated = True

        info = self._get_info()
        info.update({
            "collision": collided,
            "passed_doorway": passed_doorway,
            "reached_target": reached_target,
            "doorways_passed": len(self._visited_doorways),
        })
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        """Matplotlib PNG (human) or compact RGB array (rgb_array)."""
        if self.render_mode is None:
            return None
        # Use the target-room centroid as a visual stand-in for the goal.
        goal_cell = self._target_centroid_cell()
        if self.render_mode == "rgb_array":
            return render_rgb_array(self._agent_pos, goal_cell)
        if self.render_mode == "human":
            return render_environment(
                agent_pos=self._agent_pos,
                goal_pos=goal_cell,
                save_path="custom_env_render.png",
                dpi=300,
            )
        raise ValueError(f"Unknown render_mode: {self.render_mode}")

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation builder (the 16-d vector)
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Construct the flat 16-d observation vector."""
        obs = np.zeros(16, dtype=np.float32)
        r, c = self._agent_pos

        # --- Proximity sensors (5 dims, indices 0..4) ------------------
        # 1.0 if the neighbour is an obstacle OR outside the grid boundary
        # (both untraversable); otherwise 0.0. The "current" sensor is 1.0
        # only if the agent is standing on an obstacle — impossible in
        # normal play, kept for shape consistency with the paper.
        for i, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                obs[i] = 1.0           # out-of-bounds treated as blocking
            elif not self._walkable[nr, nc]:
                obs[i] = 1.0           # obstacle
            else:
                obs[i] = 0.0

        # --- Target room one-hot (4 dims, indices 5..8) ----------------
        obs[5 + TARGET_ROOMS.index(self._target_room)] = 1.0

        # --- Navigation state (7 dims, indices 9..15) ------------------
        # Normalized (x, y); using (col, row) so x is horizontal.
        obs[9]  = c / (GRID_SIZE - 1)
        obs[10] = r / (GRID_SIZE - 1)

        # Normalized Euclidean distance to the target-room centroid
        tr_r, tr_c = self._room_centroids[self._target_room]
        obs[11] = min(1.0, float(np.hypot(r - tr_r, c - tr_c)) / self._max_dist)

        # Remaining time fraction
        obs[12] = max(0, self.max_steps - self._steps) / self.max_steps

        # Current-region one-hot (3 dims):
        #   13: in one of the four rooms
        #   14: in the hallway
        #   15: in a doorway cell
        on_doorway = (r, c) in self._doorways
        if on_doorway:
            obs[15] = 1.0
        elif int(self._room_grid[r, c]) == HALLWAY:
            obs[14] = 1.0
        else:
            obs[13] = 1.0

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Diagnostic info dict (not used by the policy)."""
        r, c = self._agent_pos
        tr_r, tr_c = self._room_centroids[self._target_room]
        return {
            "steps": self._steps,
            "agent_pos": self._agent_pos,
            "target_room": TARGET_ROOM_NAMES[self._target_room],
            "target_centroid": (tr_r, tr_c),
            "distance_to_target": float(np.hypot(r - tr_r, c - tr_c)),
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def in_target_room(self) -> bool:
        """True iff the agent is currently standing in the target room."""
        return bool(self._room_grid[self._agent_pos] == self._target_room)

    def _target_centroid_cell(self) -> tuple[int, int]:
        """Round the target centroid to the nearest actual grid cell."""
        tr_r, tr_c = self._room_centroids[self._target_room]
        return (int(round(tr_r)), int(round(tr_c)))


# ---------------------------------------------------------------------------
# Register as a Gymnasium id for gym.make() usage.
# ---------------------------------------------------------------------------
try:
    gym.register(
        id="ResidentialGrid-v0",
        entry_point=f"{__name__}:ResidentialGridEnv",
        max_episode_steps=ResidentialGridEnv.DEFAULT_MAX_STEPS,
    )
except gym.error.Error:
    pass  # already registered


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("[1/4] Gymnasium env_checker ...")
    env = ResidentialGridEnv()
    check_env(env, skip_render_check=True)
    print("      ✓ passed")

    print("[2/4] Random-policy rollout (3 episodes) ...")
    rng = np.random.default_rng(0)
    n_actions = env.action_space.n
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        G, done = 0.0, False
        while not done:
            a = int(rng.integers(0, n_actions))
            obs, r, term, trunc, info = env.step(a)
            G += r
            done = term or trunc
        print(f"      ep={ep}  target={info['target_room']:<12}  "
              f"steps={info['steps']:3d}  "
              f"collision={info.get('collision', False)!s:5}  "
              f"reached={info.get('reached_target', False)!s:5}  "
              f"return={G:+.2f}")

    print("[3/4] Verify Wait action behaves correctly ...")
    obs, info = env.reset(seed=0)
    pos0 = info["agent_pos"]
    obs, r, term, trunc, info = env.step(4)  # Wait
    assert info["agent_pos"] == pos0, "Wait should not move the agent"
    assert abs(r - env.STEP_PENALTY) < 1e-9, "Wait should pay exactly the step penalty"
    assert not term and not trunc, "Wait should not end the episode on step 1"
    print(f"      ✓ Wait: pos unchanged, reward = {r:+.3f}")

    print("[4/4] Tabular Q-learning sanity check (fixed target = kitchen) ...")
    # Pin the target room so the tabular learner has a stationary task.
    train_env = ResidentialGridEnv()
    original_reset = train_env.reset

    def fixed_reset(*, seed=None, options=None):
        obs, info = original_reset(seed=seed, options=options)
        train_env._target_room = KITCHEN
        return train_env._get_obs(), train_env._get_info()

    train_env.reset = fixed_reset  # type: ignore[method-assign]

    Q = np.zeros((GRID_SIZE, GRID_SIZE, train_env.action_space.n),
                 dtype=np.float64)
    alpha, gamma, eps = 0.2, 0.95, 0.2
    returns: list[float] = []
    for ep in range(400):
        obs, _ = train_env.reset(seed=ep)
        rc = train_env._agent_pos  # (row, col)
        done, G = False, 0.0
        while not done:
            a = (int(rng.integers(0, train_env.action_space.n))
                 if rng.random() < eps
                 else int(np.argmax(Q[rc[0], rc[1]])))
            obs, r, term, trunc, _ = train_env.step(a)
            nxt = train_env._agent_pos
            td = r + (0.0 if term else gamma * np.max(Q[nxt[0], nxt[1]]))
            Q[rc[0], rc[1], a] += alpha * (td - Q[rc[0], rc[1], a])
            rc = nxt
            G += r
            done = term or trunc
        returns.append(G)

    early, late = float(np.mean(returns[:50])), float(np.mean(returns[-50:]))
    print(f"      avg return first 50 eps:  {early:+.3f}")
    print(f"      avg return last  50 eps:  {late:+.3f}")
    assert late > early, "Q-learning did not improve — env may be broken"
    print("      ✓ agent's return improved — env is learnable")

    print("\nAll checks passed. From your training scripts, use either:")
    print('    env = ResidentialGridEnv()')
    print('    env = gymnasium.make("ResidentialGrid-v0")')
