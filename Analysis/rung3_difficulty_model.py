# %% [markdown]
# # Rung 3 difficulty model — geometry ground-truth vs flicker survival
#
# Phase 2 calibration evidence for the flicker+frame-stack rung (pre-reg
# Amendment 2). Two questions:
#  1. What are the TRUE optimal path lengths H from spawn (0,0) to each target
#     room, on the actual frozen grid (BFS over the real walkable map)?
#  2. Under flicker(p) + frame-stack(k), what is P(the agent goes fully blind —
#     a run of >= k consecutive masked frames — at least once in an episode)?
#     Does this "blackout" model explain the observed per-room success pattern,
#     or is difficulty geometry-dependent rather than distance(H)-dependent?
#
# Saves: figures/p2_rung3_blackout_vs_p.png, figures/p2_rung3_observed_vs_predicted.png

# %%
from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Environment.custom_env import (
    ResidentialGridEnv, GRID_SIZE, AGENT_START,
    KITCHEN, BEDROOM, BATHROOM, TARGET_ROOM_NAMES,
)

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

FIG_DIR = Path(os.environ.get("ARCH_ABLATIONS_FIG_DIR", "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

K = 4                                   # frame-stack size (Amendment 2)
TARGET_ROOMS3 = [KITCHEN, BEDROOM, BATHROOM]
ROOM_NAME = {rid: TARGET_ROOM_NAMES[rid] for rid in TARGET_ROOMS3}

# 4-connected moves matching the env's Up/Down/Left/Right (Wait doesn't move).
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# %% [markdown]
# ## 1. Room BFS — true optimal path length H on the real grid

# %%
_env = ResidentialGridEnv()
WALKABLE = np.asarray(_env._walkable)          # True = in-bounds & not an obstacle
ROOM_GRID = np.asarray(_env._room_grid)


def bfs_optimal_H(goal_room_id: int) -> int:
    """Shortest #moves from AGENT_START to the NEAREST cell of goal_room_id,
    over the env's real 4-connected walkable map (obstacles block; the env's
    'reached target' fires when the agent's cell is in the target room)."""
    start = tuple(AGENT_START)
    goal_cells = {tuple(c) for c in np.argwhere(ROOM_GRID == goal_room_id)}
    if start in goal_cells:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (r, c), d = q.popleft()
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                continue
            if (nr, nc) in seen or not WALKABLE[nr, nc]:
                continue
            if (nr, nc) in goal_cells:
                return d + 1
            seen.add((nr, nc))
            q.append(((nr, nc), d + 1))
    return -1        # unreachable (should not happen)


H = {rid: bfs_optimal_H(rid) for rid in TARGET_ROOMS3}
print(f"spawn AGENT_START = {tuple(AGENT_START)}  (room {int(ROOM_GRID[tuple(AGENT_START)])})")
print("True optimal path length H (BFS over real walkable map):")
for rid in TARGET_ROOMS3:
    print(f"  {ROOM_NAME[rid]:9s} (id {rid}): H = {H[rid]:2d}")
print(f"\n  assumption was kitchen 11 / bedroom 11 / bathroom 22 -> "
      f"{'CONFIRMED' if [H[KITCHEN],H[BEDROOM],H[BATHROOM]]==[11,11,22] else 'CORRECTED (see above)'}")

# %% [markdown]
# ## 2. Consecutive-mask survival model
# Masks ~ iid Bernoulli(p). With frame-stack k, the stack has NO real frame iff
# the last k frames were all masked, i.e. a run of >= k consecutive masks. We want
# P(at least one such disabling blackout during an episode of length H).

# %%
def blackout_prob_approx(p: float, k: int, H: int) -> float:
    """Union-bound-style approximation: P ~ 1 - (1 - p^k)^(H-k+1)."""
    if H < k:
        return 0.0
    return 1.0 - (1.0 - p ** k) ** (H - k + 1)


def blackout_prob_exact(p: float, k: int, H: int) -> float:
    """EXACT P(>=1 run of >=k masks in H iid Bernoulli(p) trials), via a DP over
    the current trailing-mask run-length (absorbing once a k-run occurs)."""
    if H < k:
        return 0.0
    dp = np.zeros(k)          # dp[c] = P(in state 'c trailing masks', not yet absorbed)
    dp[0] = 1.0
    for _ in range(H):
        ndp = np.zeros(k)
        ndp[0] += dp.sum() * (1.0 - p)          # any state, unmask -> run resets to 0
        for c in range(k - 1):                  # mask extends the run; c=k-1 -> absorbed
            ndp[c + 1] += dp[c] * p
        dp = ndp
    return float(1.0 - dp.sum())


# sanity: exact vs approx agreement
for rid in TARGET_ROOMS3:
    e = blackout_prob_exact(0.7, K, H[rid])
    a = blackout_prob_approx(0.7, K, H[rid])
    print(f"  p=0.7 {ROOM_NAME[rid]:9s} H={H[rid]:2d}: exact {e:.3f}  approx {a:.3f}  "
          f"(diff {abs(e-a):.4f})")

# %% [markdown]
# ## 3. Blackout probability vs p (k=4), one curve per room

# %%
ps = np.linspace(0.5, 0.9, 41)
palette = {KITCHEN: sns.color_palette("tab10")[1],
           BEDROOM: sns.color_palette("tab10")[0],
           BATHROOM: sns.color_palette("tab10")[3]}

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhspan(0.40, 0.75, color="green", alpha=0.10,
           label="hard-but-learnable band (40-75% blackout, proxy)")
for rid in TARGET_ROOMS3:
    ys = [blackout_prob_exact(p, K, H[rid]) for p in ps]
    ax.plot(ps, ys, color=palette[rid], lw=2,
            label=f"{ROOM_NAME[rid]} (H={H[rid]})")
for pg in (0.5, 0.7, 0.8):
    ax.axvline(pg, color="0.6", ls=":", lw=1)
    ax.text(pg, 1.01, f"p={pg}", ha="center", va="bottom", fontsize=8, color="0.4")
ax.set_xlabel("flicker probability p  (frame-stack k=4)")
ax.set_ylabel("P(disabling blackout ≥1 per episode)")
ax.set_title("Rung 3 difficulty model — blackout probability vs p, per room")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
p1 = FIG_DIR / "p2_rung3_blackout_vs_p.png"
fig.savefig(p1, dpi=300, bbox_inches="tight")
print(f"[fig] {p1}")

# %% [markdown]
# ## 3b (decisive). Observed per-room success vs model-predicted difficulty

# %%
# Observed single-seed gate per-room success (docs/results_log.md Rung 3 trail).
OBSERVED = {
    0.5: {"kitchen": 1.00, "bedroom": 1.00, "bathroom": 1.00},
    0.7: {"kitchen": 0.75, "bedroom": 1.00, "bathroom": 1.00},
    0.8: {"kitchen": 0.00, "bedroom": 1.00, "bathroom": 0.00},
}
name_to_id = {v: k for k, v in ROOM_NAME.items()}

print("\nObserved success vs model blackout (higher blackout => model says harder):")
print(f"  {'p':>4} {'room':9s} {'H':>3} {'pred_blackout':>13} {'obs_success':>11} "
      f"{'obs_difficulty':>14}")
for p in (0.5, 0.7, 0.8):
    for name in ("kitchen", "bedroom", "bathroom"):
        rid = name_to_id[name]
        bo = blackout_prob_exact(p, K, H[rid])
        succ = OBSERVED[p][name]
        print(f"  {p:>4} {name:9s} {H[rid]:>3} {bo:>13.3f} {succ:>11.2f} "
              f"{1-succ:>14.2f}")

# Ordering check at p=0.7 (the decisive one).
p = 0.7
model_rank = sorted(("kitchen", "bedroom", "bathroom"),
                    key=lambda n: -blackout_prob_exact(p, K, H[name_to_id[n]]))
obs_rank = sorted(("kitchen", "bedroom", "bathroom"),
                  key=lambda n: OBSERVED[p][n])       # lowest success = hardest first
print(f"\n  p=0.7 model difficulty order (hardest->easiest by blackout): {model_rank}")
print(f"  p=0.7 observed difficulty order (hardest->easiest by 1-success): {obs_rank}")
match = model_rank[0] == obs_rank[0]
print(f"  hardest-room match? {'YES' if match else 'NO -> H is NOT the mechanism'}")

# The same-H control: kitchen and bedroom have (near-)equal H but very different
# observed difficulty -> distance cannot be the explanation.
same_H = H[KITCHEN] == H[BEDROOM]
print(f"\n  same-H control: kitchen H={H[KITCHEN]} vs bedroom H={H[BEDROOM]} "
      f"({'EQUAL' if same_H else 'differ'}); observed p=0.8 success "
      f"kitchen {OBSERVED[0.8]['kitchen']:.2f} vs bedroom {OBSERVED[0.8]['bedroom']:.2f}")

# Diagnostic figure: predicted blackout (bars) vs observed difficulty (markers).
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, p in zip(axes, (0.5, 0.7, 0.8)):
    names = ["kitchen", "bedroom", "bathroom"]
    preds = [blackout_prob_exact(p, K, H[name_to_id[n]]) for n in names]
    obsd = [1 - OBSERVED[p][n] for n in names]
    x = np.arange(3)
    ax.bar(x, preds, color="0.7", label="model: P(blackout)")
    ax.plot(x, obsd, "o-", color="crimson", label="observed: 1 - success")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\nH={H[name_to_id[n]]}" for n in names])
    ax.set_title(f"p={p}")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("difficulty (blackout prob / 1-success)")
axes[-1].legend(loc="upper right", fontsize=8)
fig.suptitle("Model-predicted difficulty (H-based) vs observed per-room difficulty")
fig.tight_layout()
p2 = FIG_DIR / "p2_rung3_observed_vs_predicted.png"
fig.savefig(p2, dpi=300, bbox_inches="tight")
print(f"[fig] {p2}")

# %% [markdown]
# ## 4. Feasibility verdict — is there a single p with all 3 rooms in-band?

# %%
BAND = (0.40, 0.75)
feasible_ps = [p for p in ps
               if all(BAND[0] <= blackout_prob_exact(p, K, H[rid]) <= BAND[1]
                      for rid in TARGET_ROOMS3)]
print(f"\nFEASIBILITY (H-model): p in [0.5,0.9] with ALL three rooms' blackout in "
      f"[{BAND[0]},{BAND[1]}]:")
if feasible_ps:
    print(f"  feasible window: p in [{min(feasible_ps):.3f}, {max(feasible_ps):.3f}]")
else:
    print("  NONE — no single p puts all three rooms in-band simultaneously.")
    # show why: at the p where the far room enters the band, where are the near rooms?
    for rid in TARGET_ROOMS3:
        in_band = [p for p in ps if BAND[0] <= blackout_prob_exact(p, K, H[rid]) <= BAND[1]]
        if in_band:
            print(f"    {ROOM_NAME[rid]:9s} (H={H[rid]}) in-band for p in "
                  f"[{min(in_band):.3f}, {max(in_band):.3f}]")
        else:
            print(f"    {ROOM_NAME[rid]:9s} (H={H[rid]}) never in-band on this grid")

print("\n[done] see printed diagnostic above and the two figures.")
