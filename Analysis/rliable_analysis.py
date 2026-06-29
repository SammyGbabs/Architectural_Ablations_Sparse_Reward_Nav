# %% [markdown]
# # Phase 1 — rliable statistical analysis
#
# Full Phase 1 analysis of the 12 per-seed CSVs (9 main configs @ 10 seeds +
# 3 baselines @ 5 seeds) produced by `Training/metrics.RichEvalCallback`.
# Everything aggregates with **IQM + 95% stratified-bootstrap CIs** (Agarwal
# et al., 2021, *rliable*).
#
# Run this in Colab with Drive mounted; it reads
# `/content/drive/MyDrive/arch-ablations/csv/p1_*.csv` directly. Override the
# input dir with the env var `ARCH_ABLATIONS_CSV_DIR` if your path differs.
#
# Analyses (each prints a clean text block you can pull poster numbers from):
#   A. H1 — PPO Exp 1 (symmetric) vs PPO Exp 4 (inverted): IQM+CIs, P(Exp4>Exp1),
#      performance profiles, explicit CI-overlap verdict.
#   B. PPO architecture sweep (Exp 1-4): do any separate, or all tie ~27.8?
#   C. DQN architecture sweep (Exp 1-5): does Exp 5 actually lead?
#   D. PPO vs DQN: final IQM+CIs AND sample efficiency (steps-to-90%). Checks the
#      original "~12.5x more sample-efficient" claim under 10 seeds.
#   E. Baselines (5-seed): do Double/Dueling beat vanilla DQN? A2C collapse.
#   F. Robustness: per-seed converged-vs-collapsed counts for ALL configs,
#      reported SEPARATELY from IQM (IQM trims outliers; the collapse count is
#      itself a finding).
#
# Figures saved to `figures/` (300 DPI):
#   p1_iqm_main_configs.png                      — poster centerpiece
#   p1_perf_profile_ppo_inverted_vs_symmetric.png — H1 performance profile
#   p1_sample_efficiency_ppo_vs_dqn.png           — sample-efficiency comparison

# %%
# ---------------------------------------------------------------------------
# Imports & configuration
# ---------------------------------------------------------------------------
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "Arial"

# Input dir: the mounted-Drive csv/ folder where the sweep writes per-seed CSVs.
DATA_DIR = Path(os.environ.get(
    "ARCH_ABLATIONS_CSV_DIR", "/content/drive/MyDrive/arch-ablations/csv"))
FIG_DIR = Path(os.environ.get("ARCH_ABLATIONS_FIG_DIR", "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

PHASE = "p1"
# Bootstrap reps for the CIs. 50_000 is the rliable default for final numbers;
# drop to e.g. 2_000 for a fast pass while iterating.
REPS = int(os.environ.get("ARCH_ABLATIONS_REPS", "50000"))

# Per-seed scalar used as a run's "score" for cross-seed IQM. Eval is
# deterministic within a seed (fixed spawn; only the target room varies), so
# the per-seed mean and per-seed IQM are near-identical; mean is the run score.
SCORE_COL = "eval_return_mean"
SAMPLE_EFF_COL = "sample_eff_steps_90"

# A seed is "collapsed" if its final success rate is below this (it fell into
# the collision/suicide local optimum instead of solving the task). Reported in
# analysis F, SEPARATELY from IQM.
COLLAPSE_SUCCESS_THRESHOLD = 0.5
CLEAN_SUCCESS_THRESHOLD = 0.9
CLEAN_COLLISION_THRESHOLD = 0.1

# Config groups (config_id -> human label for plots/tables).
PPO_CONFIGS = {
    "ppo_exp1": "PPO Exp 1 (sym)",
    "ppo_exp2": "PPO Exp 2",
    "ppo_exp3": "PPO Exp 3",
    "ppo_exp4": "PPO Exp 4 (inv)",
}
DQN_CONFIGS = {
    "dqn_exp1": "DQN Exp 1",
    "dqn_exp2": "DQN Exp 2",
    "dqn_exp3": "DQN Exp 3",
    "dqn_exp4": "DQN Exp 4",
    "dqn_exp5": "DQN Exp 5",
}
BASELINE_CONFIGS = {
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "a2c": "A2C",
}
MAIN_CONFIGS = {**PPO_CONFIGS, **DQN_CONFIGS}              # 9 main configs
ALL_CONFIGS = {**MAIN_CONFIGS, **BASELINE_CONFIGS}         # 12 total
LABELS = ALL_CONFIGS

# Collected one-line summary blocks, printed verbatim at the end.
SUMMARY: list[str] = []


def _log(line: str = "") -> None:
    """Print and also retain for the consolidated end-of-run summary."""
    print(line)
    SUMMARY.append(line)


# %%
# ---------------------------------------------------------------------------
# Data loading: one CSV per config -> per-seed arrays
# ---------------------------------------------------------------------------
def load_config_frame(config_id: str) -> dict[str, np.ndarray] | None:
    """Read p1_{config_id}.csv (one row per seed) into a {column: np.ndarray}
    dict, sorted by seed. Numeric columns are float arrays (blank -> NaN);
    non-numeric columns stay object arrays. None if the file is absent/empty.
    stdlib csv only — no pandas dependency."""
    path = DATA_DIR / f"{PHASE}_{config_id}.csv"
    if not path.is_file():
        print(f"[warn] missing CSV (skipping): {path}")
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    rows.sort(key=lambda r: int(float(r["seed"])))          # sort by seed
    cols: dict[str, np.ndarray] = {}
    for key in rows[0]:
        raw = [r[key] for r in rows]
        try:
            cols[key] = np.array(
                [float(x) if x not in ("", None) else np.nan for x in raw], dtype=float)
        except ValueError:
            cols[key] = np.array(raw, dtype=object)          # e.g. wandb_run_name
    return cols


def n_seeds(config_id: str) -> int:
    return int(FRAMES[config_id]["seed"].size)


FRAMES: dict[str, dict[str, np.ndarray]] = {}
for cid in ALL_CONFIGS:
    cols = load_config_frame(cid)
    if cols is not None and cols["seed"].size:
        FRAMES[cid] = cols

present = list(FRAMES)
print(f"Loaded {len(present)}/{len(ALL_CONFIGS)} configs from {DATA_DIR}")
for cid in present:
    print(f"  {LABELS[cid]:18s}  n_seeds={n_seeds(cid):2d}  "
          f"seeds={sorted(FRAMES[cid]['seed'].astype(int).tolist())}")
if not present:
    raise SystemExit(
        f"No CSVs found in {DATA_DIR}. Set ARCH_ABLATIONS_CSV_DIR to the folder "
        "holding the p1_*.csv files (the sweep's <output_dir>/csv/).")


def scores_of(config_id: str, col: str = SCORE_COL) -> np.ndarray:
    """Per-seed scores for a config as a (n_seeds, 1) matrix for rliable."""
    vals = np.asarray(FRAMES[config_id][col], dtype=float)
    return vals.reshape(-1, 1)


def score_dict(config_ids) -> dict[str, np.ndarray]:
    """{label: (n_seeds, 1) score matrix} for the given configs that are present."""
    return {LABELS[c]: scores_of(c) for c in config_ids if c in FRAMES}


# %%
# ---------------------------------------------------------------------------
# rliable helpers
# ---------------------------------------------------------------------------
def iqm_interval_estimates(sdict: dict[str, np.ndarray], reps: int = REPS):
    """IQM point estimate + 95% stratified-bootstrap CI per algorithm.

    Returns (points, cis) where points[label] -> shape (1,) and
    cis[label] -> shape (2, 1) (low/high)."""
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])
    return rly.get_interval_estimates(sdict, iqm_fn, reps=reps)


def iqm_ci_tuple(config_id: str, reps: int = REPS) -> tuple[float, float, float]:
    """(iqm, ci_low, ci_high) for a single config."""
    pts, cis = iqm_interval_estimates({LABELS[config_id]: scores_of(config_id)}, reps)
    lab = LABELS[config_id]
    return float(pts[lab][0]), float(cis[lab][0, 0]), float(cis[lab][1, 0])


def cis_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Do two (low, high) intervals overlap?"""
    (alo, ahi), (blo, bhi) = a, b
    return not (alo > bhi or blo > ahi)


def print_iqm_table(config_ids, reps: int = REPS) -> dict[str, tuple]:
    """Print an IQM + CI + mean/median/n table; return {config_id: (iqm,lo,hi)}."""
    sdict = score_dict(config_ids)
    pts, cis = iqm_interval_estimates(sdict, reps)
    _log(f"  {'config':18s} {'n':>3s} {'IQM':>7s} {'95% CI':>17s} "
         f"{'mean':>7s} {'median':>7s}")
    out: dict[str, tuple] = {}
    for cid in config_ids:
        if cid not in FRAMES:
            continue
        lab = LABELS[cid]
        iqm = float(pts[lab][0]); lo = float(cis[lab][0, 0]); hi = float(cis[lab][1, 0])
        raw = np.asarray(FRAMES[cid][SCORE_COL], dtype=float)
        _log(f"  {lab:18s} {len(raw):3d} {iqm:7.2f} "
             f"[{lo:7.2f},{hi:7.2f}] {raw.mean():7.2f} {np.median(raw):7.2f}")
        out[cid] = (iqm, lo, hi)
    return out


def probability_of_improvement(better: str, worse: str, reps: int = REPS):
    """P(score(better) > score(worse)) with 95% CI (rliable, stratified boot)."""
    pairs = {f"{LABELS[better]},{LABELS[worse]}":
             (scores_of(better), scores_of(worse))}
    probs, prob_cis = rly.get_interval_estimates(
        pairs, metrics.probability_of_improvement, reps=reps)
    key = list(pairs)[0]
    # probability_of_improvement returns a scalar point estimate; the CI is a
    # 2-vector. Flatten defensively so this works whether rliable hands back a
    # scalar or a length-1 array.
    pt = np.asarray(probs[key]).reshape(-1)
    ci = np.asarray(prob_cis[key]).reshape(-1)
    return float(pt[0]), float(ci[0]), float(ci[1])


def best_config_by_iqm(config_ids) -> str:
    """The config with the highest IQM among those present (the 'representative')."""
    candidates = [c for c in config_ids if c in FRAMES]
    return max(candidates, key=lambda c: metrics.aggregate_iqm(scores_of(c)))


# %%
# ---------------------------------------------------------------------------
# A. H1 — PPO Exp 1 (symmetric) vs PPO Exp 4 (inverted)
# ---------------------------------------------------------------------------
_log("=" * 74)
_log("A. H1 PRIMARY - PPO Exp 1 (symmetric)  vs  PPO Exp 4 (inverted)")
_log("=" * 74)

if "ppo_exp1" in FRAMES and "ppo_exp4" in FRAMES:
    a_table = print_iqm_table(["ppo_exp1", "ppo_exp4"])
    iqm1, lo1, hi1 = a_table["ppo_exp1"]
    iqm4, lo4, hi4 = a_table["ppo_exp4"]

    p_imp, p_lo, p_hi = probability_of_improvement("ppo_exp4", "ppo_exp1")
    overlap = cis_overlap((lo1, hi1), (lo4, hi4))

    _log("")
    _log(f"  IQM delta (Exp4 - Exp1)      : {iqm4 - iqm1:+.2f}")
    _log(f"  P(Exp4 > Exp1)               : {p_imp:.3f}  95% CI [{p_lo:.3f}, {p_hi:.3f}]")
    _log(f"  95% CIs overlap?             : {'YES' if overlap else 'NO'}")
    _log("")
    if not overlap and iqm4 > iqm1:
        verdict = ("VERDICT: Inverted (Exp4) IQM CI lies ABOVE symmetric (Exp1) "
                   "with no overlap -> H1 SUPPORTED on the primary comparison.")
    elif not overlap and iqm1 > iqm4:
        verdict = ("VERDICT: Symmetric (Exp1) IQM CI lies ABOVE inverted (Exp4) "
                   "with no overlap -> H1 REFUTED on the primary comparison.")
    else:
        verdict = ("VERDICT: 95% IQM CIs OVERLAP -> no separation between inverted "
                   "and symmetric on final return. H1 NOT supported by the primary "
                   "comparison at this budget; P(improvement) and the per-metric / "
                   "per-room results carry whatever signal exists.")
    _log("  " + verdict)
else:
    _log("  [skip] need both ppo_exp1 and ppo_exp4 CSVs.")

# %%
# ---------------------------------------------------------------------------
# B. PPO architecture sweep — all 4 PPO configs
# ---------------------------------------------------------------------------
_log("")
_log("=" * 74)
_log("B. PPO ARCHITECTURE SWEEP - Exp 1-4")
_log("=" * 74)
b_table = print_iqm_table(list(PPO_CONFIGS))
if b_table:
    iqms = {c: v[0] for c, v in b_table.items()}
    spread = max(iqms.values()) - min(iqms.values())
    # all-tie test: do all pairwise IQM CIs mutually overlap?
    cis = {c: (v[1], v[2]) for c, v in b_table.items()}
    all_overlap = all(
        cis_overlap(cis[a], cis[b]) for a in cis for b in cis if a < b)
    _log("")
    _log(f"  IQM spread (max-min)         : {spread:.2f}")
    _log(f"  all pairwise CIs overlap?    : {'YES' if all_overlap else 'NO'}")
    if all_overlap:
        _log("  VERDICT: all four PPO architectures TIE within CIs - no architecture "
             "separates on final return (consistent with the ~27.8 plateau).")
    else:
        sep = max(b_table, key=lambda c: b_table[c][0])
        _log(f"  VERDICT: PPO architectures do NOT all tie; {LABELS[sep]} has the "
             "highest IQM and at least one CI is disjoint - inspect the table.")

# %%
# ---------------------------------------------------------------------------
# C. DQN architecture sweep — all 5 DQN configs
# ---------------------------------------------------------------------------
_log("")
_log("=" * 74)
_log("C. DQN ARCHITECTURE SWEEP - Exp 1-5")
_log("=" * 74)
c_table = print_iqm_table(list(DQN_CONFIGS))
if c_table:
    cis = {c: (v[1], v[2]) for c, v in c_table.items()}
    all_overlap = all(
        cis_overlap(cis[a], cis[b]) for a in cis for b in cis if a < b)
    lead = max(c_table, key=lambda c: c_table[c][0])
    _log("")
    _log(f"  highest-IQM DQN config       : {LABELS[lead]} (IQM {c_table[lead][0]:.2f})")
    _log(f"  all pairwise CIs overlap?    : {'YES' if all_overlap else 'NO'}")
    if "dqn_exp5" in c_table:
        exp5_is_top = lead == "dqn_exp5"
        _log(f"  is Exp 5 the multi-seed leader: {'YES' if exp5_is_top else 'NO'}")
    if all_overlap:
        _log("  VERDICT: DQN configs TIE within CIs - the original's 'best' (Exp 5) "
             "does not separate from the others under 10 seeds.")
    else:
        _log(f"  VERDICT: {LABELS[lead]} leads with at least one disjoint CI - "
             "Exp 5's lead (or another's) survives multi-seed; see table.")

# %%
# ---------------------------------------------------------------------------
# D. PPO vs DQN — final performance AND sample efficiency
# ---------------------------------------------------------------------------
_log("")
_log("=" * 74)
_log("D. PPO vs DQN - algorithm comparison (final return + sample efficiency)")
_log("=" * 74)

rep_ppo = best_config_by_iqm(PPO_CONFIGS)
rep_dqn = best_config_by_iqm(DQN_CONFIGS)
_log(f"  representative PPO (max IQM)  : {LABELS[rep_ppo]}")
_log(f"  representative DQN (max IQM)  : {LABELS[rep_dqn]}")
_log("")
_log("  -- final return --")
d_table = print_iqm_table([rep_ppo, rep_dqn])
p_pd, p_pd_lo, p_pd_hi = probability_of_improvement(rep_ppo, rep_dqn)
ov_pd = cis_overlap((d_table[rep_ppo][1], d_table[rep_ppo][2]),
                    (d_table[rep_dqn][1], d_table[rep_dqn][2]))
_log(f"  P({LABELS[rep_ppo]} > {LABELS[rep_dqn]}) : {p_pd:.3f} "
     f"95% CI [{p_pd_lo:.3f}, {p_pd_hi:.3f}]")
_log(f"  final-return CIs overlap?     : {'YES' if ov_pd else 'NO'}")


def sample_eff_iqm(config_id: str, reps: int = REPS):
    """IQM + CI of steps-to-90% over NON-NaN seeds; (iqm, lo, hi, n_valid)."""
    raw = np.asarray(FRAMES[config_id][SAMPLE_EFF_COL], dtype=float)
    valid = raw[~np.isnan(raw)]
    if valid.size < 3:
        return (np.nan, np.nan, np.nan, int(valid.size))
    pts, cis = iqm_interval_estimates({LABELS[config_id]: valid.reshape(-1, 1)}, reps)
    lab = LABELS[config_id]
    return (float(pts[lab][0]), float(cis[lab][0, 0]), float(cis[lab][1, 0]),
            int(valid.size))


_log("")
_log("  -- sample efficiency (env-steps to 90% of asymptotic eval IQM; LOWER=better) --")
se_ppo = sample_eff_iqm(rep_ppo)
se_dqn = sample_eff_iqm(rep_dqn)
_log(f"  {LABELS[rep_ppo]:18s} IQM steps {se_ppo[0]:11,.0f}  "
     f"95% CI [{se_ppo[1]:,.0f}, {se_ppo[2]:,.0f}]  (n={se_ppo[3]})")
_log(f"  {LABELS[rep_dqn]:18s} IQM steps {se_dqn[0]:11,.0f}  "
     f"95% CI [{se_dqn[1]:,.0f}, {se_dqn[2]:,.0f}]  (n={se_dqn[3]})")
if np.isfinite(se_ppo[0]) and np.isfinite(se_dqn[0]) and se_ppo[0] > 0:
    ratio = se_dqn[0] / se_ppo[0]
    _log(f"  PPO sample-efficiency advantage: {ratio:.2f}x fewer steps than DQN "
         f"(original paper claimed ~12.5x)")
    if ratio >= 10:
        _log("  VERDICT: original ~12.5x sample-efficiency claim is in the right "
             "ballpark under 10 seeds (>=10x).")
    elif ratio >= 2:
        _log(f"  VERDICT: PPO is more sample-efficient ({ratio:.1f}x) but the "
             "original ~12.5x is NOT reproduced under 10 seeds.")
    else:
        _log(f"  VERDICT: the ~12.5x claim does NOT hold - measured advantage is "
             f"only {ratio:.1f}x.")
else:
    _log("  [warn] insufficient non-NaN sample-eff seeds for a ratio.")

# %%
# ---------------------------------------------------------------------------
# E. Baselines (5-seed exploratory): Double DQN, Dueling DQN, A2C vs vanilla DQN
# ---------------------------------------------------------------------------
_log("")
_log("=" * 74)
_log("E. BASELINES (5-seed exploratory) - Double/Dueling DQN, A2C")
_log("=" * 74)
# Compare the baselines against vanilla DQN Exp 5 (what double/dueling are matched to).
e_table = print_iqm_table(["dqn_exp5", "double_dqn", "dueling_dqn", "a2c"])
if e_table and "dqn_exp5" in e_table:
    base_ci = (e_table["dqn_exp5"][1], e_table["dqn_exp5"][2])
    for cid in ("double_dqn", "dueling_dqn"):
        if cid not in e_table:
            continue
        ov = cis_overlap(base_ci, (e_table[cid][1], e_table[cid][2]))
        d = e_table[cid][0] - e_table["dqn_exp5"][0]
        verb = ("TIES vanilla DQN (CIs overlap)" if ov
                else ("BEATS" if d > 0 else "TRAILS") + " vanilla DQN (disjoint CI)")
        _log(f"  {LABELS[cid]:12s}: IQM delta vs DQN Exp 5 = {d:+.2f} -> {verb}")
if "a2c" in FRAMES:
    succ = np.asarray(FRAMES["a2c"]["success_rate"], dtype=float)
    coll = np.asarray(FRAMES["a2c"]["collision_rate"], dtype=float)
    ret = np.asarray(FRAMES["a2c"][SCORE_COL], dtype=float)
    n_floor = int(np.sum(succ < COLLAPSE_SUCCESS_THRESHOLD))
    _log("")
    _log(f"  A2C COLLAPSE: {n_floor}/{len(succ)} seeds at the collision floor "
         f"(mean success {succ.mean():.2f}, mean collision {coll.mean():.2f}, "
         f"mean return {ret.mean():.2f}).")
    _log("  A2C is a deliberate exploratory contrast: native on-policy A2C fails "
         "this sparse-reward task (suicide local optimum), where off-policy DQN and "
         "PPO succeed. Not H1 evidence; reported as context.")

# %%
# ---------------------------------------------------------------------------
# F. Robustness — per-seed converged vs collapsed (reported SEPARATELY from IQM)
# ---------------------------------------------------------------------------
_log("")
_log("=" * 74)
_log("F. ROBUSTNESS - per-seed collapse counts (IQM trims outliers; counts do not)")
_log("=" * 74)
_log(f"  collapse rule: a seed is COLLAPSED if final success_rate < "
     f"{COLLAPSE_SUCCESS_THRESHOLD} (fell into the collision local optimum).")
_log(f"  {'config':18s} {'seeds':>5s} {'conv':>4s} {'coll':>4s}  collapsed-seed ids")


def collapse_report(config_id: str) -> dict:
    cols = FRAMES[config_id]
    succ = np.asarray(cols["success_rate"], dtype=float)
    coll = np.asarray(cols["collision_rate"], dtype=float)
    seeds = np.asarray(cols["seed"], dtype=int)
    collapsed_mask = succ < COLLAPSE_SUCCESS_THRESHOLD
    clean_mask = (succ >= CLEAN_SUCCESS_THRESHOLD) & (coll <= CLEAN_COLLISION_THRESHOLD)
    collapsed_seeds = seeds[collapsed_mask].tolist()
    return {
        "n": len(seeds),
        "n_converged": int((~collapsed_mask).sum()),
        "n_collapsed": int(collapsed_mask.sum()),
        "n_clean": int(clean_mask.sum()),
        "collapsed_seeds": collapsed_seeds,
    }


collapse = {}
for cid in ALL_CONFIGS:
    if cid not in FRAMES:
        continue
    r = collapse_report(cid)
    collapse[cid] = r
    ids = ",".join(map(str, r["collapsed_seeds"])) if r["collapsed_seeds"] else "-"
    _log(f"  {LABELS[cid]:18s} {r['n']:5d} {r['n_converged']:4d} "
         f"{r['n_collapsed']:4d}  {ids}")

_log("")
_log("  Notes:")
for cid, note_pair in (("ppo_exp4", "inverted"), ("ppo_exp1", "symmetric")):
    if cid in collapse:
        r = collapse[cid]
        _log(f"  - {LABELS[cid]} ({note_pair}): {r['n_collapsed']} collapsed "
             f"seed(s) {r['collapsed_seeds'] or '[none]'}; {r['n_clean']} cleanly "
             "converged. IQM drops the outlier(s); the count is reported here.")
if "a2c" in collapse:
    _log(f"  - {LABELS['a2c']}: {collapse['a2c']['n_collapsed']}/"
         f"{collapse['a2c']['n']} seeds collapsed (full collapse - see E).")

# %%
# ---------------------------------------------------------------------------
# FIGURE 1 (centerpiece): IQM + 95% CI for the 9 main configs
# ---------------------------------------------------------------------------
main_present = [c for c in MAIN_CONFIGS if c in FRAMES]
sdict_main = score_dict(main_present)
pts_main, cis_main = iqm_interval_estimates(sdict_main)

# rliable's plot_interval_estimates draws a horizontal point+CI plot.
palette = sns.color_palette("tab10", n_colors=len(main_present))
colors = {LABELS[c]: palette[i] for i, c in enumerate(main_present)}
algos = [LABELS[c] for c in main_present]

fig, ax = plt.subplots(figsize=(8, 6))
plot_utils.plot_interval_estimates(
    {a: pts_main[a] for a in algos},
    {a: cis_main[a] for a in algos},
    metric_names=["IQM eval return"],
    algorithms=algos,
    colors=colors,
    xlabel="IQM eval return (95% stratified-bootstrap CI)",
    ax=ax,
)
ax.set_title("Phase 1 — IQM eval return, main configs (PPO Exp 1-4, DQN Exp 1-5)")
fig.tight_layout()
fig1_path = FIG_DIR / "p1_iqm_main_configs.png"
fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
print(f"[fig] wrote {fig1_path}")
plt.show()

# %%
# ---------------------------------------------------------------------------
# FIGURE 2: H1 performance profile — PPO inverted vs symmetric
# ---------------------------------------------------------------------------
if "ppo_exp1" in FRAMES and "ppo_exp4" in FRAMES:
    h1_dict = score_dict(["ppo_exp1", "ppo_exp4"])
    all_scores = np.concatenate([v.ravel() for v in h1_dict.values()])
    lo = float(np.floor(all_scores.min())) - 1.0
    hi = float(np.ceil(all_scores.max())) + 1.0
    taus = np.linspace(lo, hi, 100)
    profiles, profile_cis = rly.create_performance_profile(h1_dict, taus, reps=REPS)

    fig, ax = plt.subplots(figsize=(7, 5))
    h1_colors = {"PPO Exp 1 (sym)": sns.color_palette("tab10")[0],
                 "PPO Exp 4 (inv)": sns.color_palette("tab10")[3]}
    plot_utils.plot_performance_profiles(
        profiles, taus,
        performance_profile_cis=profile_cis,
        colors=h1_colors,
        xlabel=r"Eval return threshold $\tau$",
        ax=ax,
    )
    ax.legend(loc="upper right")
    ax.set_title("H1 performance profile — PPO inverted vs symmetric")
    fig.tight_layout()
    fig2_path = FIG_DIR / "p1_perf_profile_ppo_inverted_vs_symmetric.png"
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    print(f"[fig] wrote {fig2_path}")
    plt.show()

# %%
# ---------------------------------------------------------------------------
# FIGURE 3: PPO vs DQN sample efficiency (IQM steps-to-90% + CIs, all configs)
# ---------------------------------------------------------------------------
def se_points(config_ids):
    rows = []
    for c in config_ids:
        if c not in FRAMES:
            continue
        iqm, lo, hi, n = sample_eff_iqm(c)
        if np.isfinite(iqm):
            rows.append((LABELS[c], iqm, lo, hi, n))
    return rows


ppo_se = se_points(list(PPO_CONFIGS))
dqn_se = se_points(list(DQN_CONFIGS))

fig, ax = plt.subplots(figsize=(8, 5))
y = 0
yticks, ylabels = [], []
for rows, color, grp in ((ppo_se, sns.color_palette("tab10")[0], "PPO"),
                         (dqn_se, sns.color_palette("tab10")[3], "DQN")):
    for lab, iqm, lo, hi, n in rows:
        ax.errorbar(iqm / 1000.0, y,
                    xerr=[[(iqm - lo) / 1000.0], [(hi - iqm) / 1000.0]],
                    fmt="o", color=color, capsize=4, markersize=7)
        yticks.append(y); ylabels.append(f"{lab} (n={n})")
        y += 1
    y += 0.5
ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
ax.set_xlabel("Sample efficiency — IQM env-steps to 90% asymptote (×10³, LOWER = better)")
ax.set_title("PPO vs DQN sample efficiency (95% CI)")
# annotate the representative ratio if available
if np.isfinite(se_ppo[0]) and np.isfinite(se_dqn[0]) and se_ppo[0] > 0:
    ax.annotate(f"{se_dqn[0] / se_ppo[0]:.1f}× PPO advantage\n(paper claimed ~12.5×)",
                xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=10, bbox=dict(boxstyle="round", fc="w", ec="0.6"))
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=sns.color_palette("tab10")[0], label="PPO"),
                   Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=sns.color_palette("tab10")[3], label="DQN")],
          loc="lower right" if False else "upper right")
fig.tight_layout()
fig3_path = FIG_DIR / "p1_sample_efficiency_ppo_vs_dqn.png"
fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
print(f"[fig] wrote {fig3_path}")
plt.show()

# %%
# ---------------------------------------------------------------------------
# Consolidated text summary (read straight off for the poster)
# ---------------------------------------------------------------------------
print("\n\n")
print("#" * 74)
print("# PHASE 1 - CONSOLIDATED SUMMARY (A-F)")
print("#" * 74)
for line in SUMMARY:
    print(line)
print("\nFigures written:")
for p in ("p1_iqm_main_configs.png",
          "p1_perf_profile_ppo_inverted_vs_symmetric.png",
          "p1_sample_efficiency_ppo_vs_dqn.png"):
    print(f"  figures/{p}")
