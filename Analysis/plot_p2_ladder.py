# %% [markdown]
# # Phase 2 §5 figures — observability-ladder IQM ladder + p=0.8 fracture per-seed
#
# Reads the committed claim-grade CSVs (`results/csv/p2_*_sym.csv`, 10 seeds each)
# and produces the two §5 figures:
#   figures/p2_ladder_iqm.png            — per-cell IQM + 95% CI (rliable), with the
#                                          ~27.8 ceiling reference; ceilings vs the
#                                          p=0.8 fracture at a glance.
#   figures/p2_flicker08_perseed.png     — p=0.8 fracture companion: per-seed
#                                          success/collision/timeout split + a
#                                          per-room (kitchen/bedroom/bathroom) SR
#                                          heatmap showing kitchen abandoned 10/10.
#
# The p=0.7 cell (p2_flicker07_sym) is included automatically once its CSV exists.

# %%
from __future__ import annotations
import os, csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rliable import library as rly, metrics

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

CSV_DIR = Path(os.environ.get("ARCH_ABLATIONS_CSV_DIR", "results/csv"))
FIG_DIR = Path(os.environ.get("ARCH_ABLATIONS_FIG_DIR", "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)
CEILING = 27.8            # A-STRICT/control reward ceiling (reference line)
# Paper-standard bootstrap settings, matching Analysis/rliable_analysis.py: 50k reps and a
# FIXED seed, so every CI drawn here regenerates to the digit from the committed CSVs.
REPS = int(os.environ.get("ARCH_ABLATIONS_REPS", "50000"))
BOOTSTRAP_SEED = int(os.environ.get("ARCH_ABLATIONS_BOOTSTRAP_SEED", "20260817"))

# (config_id, label, behavioural regime) — display order = ceilings, then the p=0.8 cell
# at both budgets. flicker07 is optional (appears once its claim-grade CSV lands).
# p=0.8 appears TWICE, at 200k and at matched 500k: the pair is the §5 finding, so the
# ladder figure has to show both or it contradicts Table 2.
CELLS = [
    ("p2_strict_sym",        "A-STRICT (13-D)",            "ceiling"),
    ("p2_alias_sym",         "Aliasing (10-D)",            "ceiling"),
    ("p2_proxnoise_sym",     "Prox-noise q=0.3 (13-D)",    "ceiling"),
    ("p2_flicker07_sym",     "Flicker p=0.7 (52-D, 500k)", "ceiling"),   # optional
    ("p2_flicker08_sym",     "Flicker p=0.8 (52-D, 200k)", "fracture"),
    ("p2_flicker08_500k_sym","Flicker p=0.8 (52-D, 500k)", "fragmentation"),
]
COLOR = {"ceiling": sns.color_palette("tab10")[0],          # blue
         "fracture": sns.color_palette("tab10")[3],         # red
         "fragmentation": sns.color_palette("tab10")[1]}    # orange


def load(cid: str):
    path = CSV_DIR / f"{cid}.csv"
    if not path.is_file():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["seed"]))
    return rows


def iqm_ci(vals):
    sd = {"x": np.asarray(vals, float).reshape(-1, 1)}
    fn = lambda x: np.array([metrics.aggregate_iqm(x)])
    np.random.seed(BOOTSTRAP_SEED)      # reproducible CI bounds (see BOOTSTRAP_SEED)
    p, c = rly.get_interval_estimates(sd, fn, reps=REPS)
    return float(p["x"][0]), float(c["x"][0, 0]), float(c["x"][1, 0])


# %%
# ---------------------------------------------------------------------------
# FIGURE 1 — per-cell IQM + 95% CI ladder
# ---------------------------------------------------------------------------
present = [(cid, lab, kind) for cid, lab, kind in CELLS if load(cid)]
fig, ax = plt.subplots(figsize=(8, 4.6))
yticks, ylabels = [], []
for y, (cid, lab, kind) in enumerate(present):
    ret = [float(r["eval_return_mean"]) for r in load(cid)]
    iqm, lo, hi = iqm_ci(ret)
    ax.errorbar(iqm, y, xerr=[[iqm - lo], [hi - iqm]], fmt="o", color=COLOR[kind],
                capsize=4, markersize=9, lw=2, elinewidth=2)
    ax.annotate(f"{iqm:.2f}", (iqm, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color=COLOR[kind])
    yticks.append(y); ylabels.append(lab)
ax.axvline(CEILING, color="0.5", ls="--", lw=1.2)
ax.text(CEILING, len(present) - 0.4, f"ceiling {CEILING}", rotation=90,
        va="top", ha="right", fontsize=8, color="0.4")
ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
ax.invert_yaxis(); ax.margins(y=0.12)
ax.set_xlabel("IQM eval return  (95% stratified-bootstrap CI, 10 seeds)")
ax.set_title("Phase 2 — observability-ladder cells (symmetric PPO):\n"
             "ceilings vs the p=0.8 fracture (200k) and its matched-budget fragmentation (500k)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR[k],
                          markersize=9, label=k)
                   for k in ("ceiling", "fracture", "fragmentation")],
          loc="lower right", frameon=True)
fig.tight_layout()
f1 = FIG_DIR / "p2_ladder_iqm.png"
fig.savefig(f1, dpi=300, bbox_inches="tight")
print(f"[fig] {f1}  ({len(present)} cells)")

# %%
# ---------------------------------------------------------------------------
# FIGURE 2 — p=0.8 fracture, per-seed companion
# ---------------------------------------------------------------------------
rows = load("p2_flicker08_sym")
if rows is None:
    print("[skip] p2_flicker08_sym.csv not found")
else:
    seeds = [int(r["seed"]) for r in rows]
    succ = np.array([float(r["success_rate"]) for r in rows])
    coll = np.array([float(r["collision_rate"]) for r in rows])
    tmo = np.clip(1 - succ - coll, 0, 1)
    room = np.array([[float(r["sr_kitchen"]), float(r["sr_bedroom"]),
                      float(r["sr_bathroom"])] for r in rows])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # Left: stacked outcome split per seed.
    y = np.arange(len(seeds))
    axL.barh(y, succ, color=sns.color_palette("tab10")[2], label="success")
    axL.barh(y, tmo, left=succ, color="0.7", label="timeout")
    axL.barh(y, coll, left=succ + tmo, color=sns.color_palette("tab10")[3], label="collision")
    axL.set_yticks(y); axL.set_yticklabels([f"seed {s}" for s in seeds])
    axL.invert_yaxis(); axL.set_xlim(0, 1)
    axL.set_xlabel("fraction of eval episodes")
    axL.set_title("Outcome split per seed — failures are TIMEOUTS, not collisions")
    axL.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Right: per-room SR heatmap (seeds x rooms). Kitchen column = 0 in 10/10.
    im = axR.imshow(room, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axR.set_xticks(range(3)); axR.set_xticklabels(["kitchen\n(H=11)", "bedroom\n(H=11)", "bathroom\n(H=22)"])
    axR.set_yticks(y); axR.set_yticklabels([f"seed {s}" for s in seeds])
    for i in range(len(seeds)):
        for j in range(3):
            axR.text(j, i, f"{room[i, j]:.2f}", ha="center", va="center", fontsize=8,
                     color="black")
    axR.set_title("Per-room success — kitchen abandoned 10/10, bedroom solved 10/10")
    fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="per-room success rate")
    fig.suptitle("Flicker p=0.8 at the 200k base budget: the fracture is systematic "
                 "across all 10 seeds\n(at matched 500k budget this resolves into six "
                 "distinct per-seed regimes — see Appendix A)", y=1.04)
    fig.tight_layout()
    f2 = FIG_DIR / "p2_flicker08_perseed.png"
    fig.savefig(f2, dpi=300, bbox_inches="tight")
    print(f"[fig] {f2}")

print("[done]")
