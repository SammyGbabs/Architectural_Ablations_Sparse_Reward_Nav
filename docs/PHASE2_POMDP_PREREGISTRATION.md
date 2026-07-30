# Phase 2 — POMDP Observability Ladder: Pre-Registration

**Status:** FROZEN before any Phase 2 POMDP run. Do not edit hypotheses,
rungs, or falsification criteria after the first run is launched. Design
changes require a dated amendment section at the bottom, not an in-place edit.

**Repo:** `Architectural_Ablations_Sparse_Reward_Nav`
**Depends on:** Phase 1 (committed, analysed). Env frozen except the obs-variant
wrappers defined here.
**Author:** Samuel Babalola. **Date registered:** 2026-07-02

---

## 0. Why this experiment exists (motivation from Phase 1)

Phase 1 refuted H1: inverted actor–critic asymmetry (`[512,256,128]/[256,128]`)
did **not** beat symmetric (`[256,256]/[256,256]`) — IQM 27.56 vs 27.66,
overlapping 95% CIs, P(inv>sym)=0.25. The diagnosis was **not** "the hypothesis
is false" but "the task could not test it": the 16-D observation hands the agent
normalised global position (dims 9–10) and distance-to-target (dim 11), making
the optimal policy close to *"move to reduce distance-to-target"* — a function so
simple a shallow actor represents it as well as a deep one. If the policy is
easy, actor depth has nothing to do, so asymmetry cannot help **by construction**.

The "policy-hard, value-easy" theory predicts asymmetry helps **only when the
policy is genuinely hard to represent while value stays smooth**. This experiment
tests that prediction directly by **progressively removing the information that
makes the policy easy** and measuring whether the inverted-vs-symmetric gap
*emerges* as the observation degrades.

This is the honest, pre-registered version of the "try other architectures"
instinct: we are **not** searching for an inverted config that wins. We fix two
architectures (the Phase 1 symmetric and inverted), fix a graded observability
ladder, commit to reporting **every cell**, and test for a **dose–response
trend**. Whatever the trend shows is the finding.

---

## 1. The observability ladder (independent variable)

Base 16-D layout (Phase 1, frozen):

| Dims  | Block                     | Content                                             |
|-------|---------------------------|-----------------------------------------------------|
| 0–4   | Proximity sensors (5)     | Binary obstacle/off-grid: Up, Down, Left, Right, Cur |
| 5–8   | Target-room one-hot (4)   | living, kitchen, bedroom, bath                       |
| 9–10  | Normalised position (2)   | (x,y)/(GRID−1) — **global self-localisation**        |
| 11    | Distance-to-target (1)    | Euclidean to target centroid, normalised — **the crutch** |
| 12    | Remaining-time frac (1)   | (T_max − steps)/T_max                                |
| 13–15 | Region one-hot (3)        | in_room, in_hallway, in_doorway                      |

**Three rungs**, each a deterministic wrapper over the frozen env (the env's
internal state is untouched; only the emitted observation vector changes):

### Rung 0 — CONTROL (16-D) — *already run in Phase 1*
The full Phase 1 observation. **No new runs needed** — reuse Exp 1 (symmetric)
and Exp 4 (inverted), both already at 10 seeds. This is the top of the ladder
(most information, easiest policy).

### Rung 1 — A-MILD (14-D) — pure removal (see Amendment 1)
- **Remove** dims 9–10 (normalised global position). Agent loses global
  self-localisation.
- **Retain** dim 11 (distance-to-target), dim 12 (remaining-time), dims 13–15
  (region one-hot), dims 0–4 (proximity), dims 5–8 (target one-hot).
- **Nothing added.** This is a strict subset of the control observation.
- Result: **14-D**. Policy is harder (no global frame) but the agent still knows
  *how far* the target is (distance-to-target retained) — just not *where it is*.

### Rung 2 — A-STRICT (13-D) — pure removal
- A-MILD, **and also remove** dim 11 (distance-to-target).
- Result: **13-D** = proximity (5) + target one-hot (4) + remaining-time (1) +
  region one-hot (3). Agent knows *which* room is the goal and its immediate
  surroundings, but not where it is globally nor how far the target is. It must
  **explore and build implicit layout sense** to navigate — the
  sharp-decision-boundary regime the theory says needs actor depth.

> **Design note (why pure removal, not §5.2's relative vector):** §5.2's
> Variant A adds a "relative target vector" as a standalone Reviewer-1 fix. For a
> *dose–response ladder* we deliberately do NOT add it: adding a target-direction
> feature partially re-introduces the very target-locating crutch the ladder
> exists to remove, confounding the manipulated variable. A strict-subset ladder
> (each rung = pure removal from the one above) is cleaner and more defensible:
> "we removed global position, then also distance, and measured whether asymmetry
> emerges as the target-locating signal disappears." See Amendment 1.

**Ladder ordering (easy → hard):** Rung 0 (16-D) → Rung 1 (14-D) → Rung 2 (13-D).
Monotone in "policy-relevant information removed."

> Amendment hook: if Rung 2 still ties (both architectures solve it easily), the
> pre-registered next step is **a Rung 3** that degrades further — e.g. perturbs
> or masks proximity sensors, or drops the region one-hot — added as a dated
> amendment, run separately. We do NOT retro-edit this rung.

---

## 2. Fixed factors (held constant across the ladder)

- **Algorithm:** PPO only (SB3, `MlpPolicy`, `policy_kwargs` net_arch dicts).
- **Architectures (2):**
  - `symmetric` = `pi=[256,256]`, `vf=[256,256]` (Phase 1 Exp 1)
  - `inverted`  = `pi=[512,256,128]`, `vf=[256,128]` (Phase 1 Exp 4)
- **Hyperparameters:** identical to Phase 1 Exp 1 / Exp 4 YAMLs, **verbatim**.
  Only `net_arch` and the obs-wrapper differ. No tuning per rung.
- **Budget:** 200k env-steps/run (Phase 1 showed PPO converges well inside this;
  re-verify on A-STRICT before the full sweep — see §5 gate).
- **Seeds:** 10 per cell (MAIN_SEEDS tier), same seeds as Phase 1.
- **Env internals:** spawn (0,0), 20×20 map, reward, Discrete(5), T_max=150 —
  all frozen. Wrappers touch **only** the emitted observation.
- **Eval / logging:** identical rich-metric logging to Phase 1 (per-seed CSV,
  eval IQM, success rate, collision rate, ep_len, per-room SR, sample-eff).

---

## 3. Design matrix & run count

New runs = **{symmetric, inverted} × {A-MILD, A-STRICT} × 10 seeds = 40 PPO runs.**

| Cell | Arch      | Rung      | Obs | Seeds | New? |
|------|-----------|-----------|-----|-------|------|
| C-S  | symmetric | 0 CONTROL | 16-D | 10   | reuse Phase 1 Exp 1 |
| C-I  | inverted  | 0 CONTROL | 16-D | 10   | reuse Phase 1 Exp 4 |
| M-S  | symmetric | 1 A-MILD  | 14-D | 10   | **yes** |
| M-I  | inverted  | 1 A-MILD  | 14-D | 10   | **yes** |
| T-S  | symmetric | 2 A-STRICT| 13-D | 10   | **yes** |
| T-I  | inverted  | 2 A-STRICT| 13-D | 10   | **yes** |

**Total new compute: 40 runs** (well inside the ~100-run single-Colab budget).
Control cells (20 runs) come free from Phase 1.

**Front-loading order** (so the decisive comparison lands first even if Colab
dies mid-sweep): **T-I, T-S** (A-STRICT pair — most likely to show the effect) →
**M-I, M-S** (A-MILD pair) → done. Read T-pair as soon as its 20 runs are in.

---

## 4. Hypotheses & falsification criteria (FROZEN)

Let `gap(rung) = IQM(inverted, rung) − IQM(symmetric, rung)`, with a 95%
stratified-bootstrap CI on the gap.

- **H-P2a (emergence):** `gap(A-STRICT) > 0` with a 95% CI excluding 0.
  *Falsified if* the A-STRICT gap CI includes 0 (asymmetry does not help even
  when the policy is hard).

- **H-P2b (dose–response, the strong claim):** `gap` increases monotonically
  across the ladder: `gap(16-D) ≤ gap(14-D) ≤ gap(13-D)`, with
  `gap(13-D) − gap(16-D)` having a 95% CI excluding 0.
  *Falsified if* the gap does not grow as observation degrades (no interaction
  between architecture and observability).

**Pre-committed interpretation table** (write the result before looking is not
possible, but write what each outcome *means* now, so it is not chosen post-hoc):

| Outcome                                    | Meaning |
|--------------------------------------------|---------|
| Gap grows monotonically, T-gap CI excludes 0 | **Theory validated** — asymmetry emerges exactly when policy is hard. Strong result. |
| Gap flat/zero at all rungs                  | **Theory refuted on this task** — asymmetry doesn't help even under partial obs. Clean, publishable null. |
| Gap non-zero at strict but not monotone     | Partial support — asymmetry helps under hard obs but no clean dose-response. Report honestly. |
| Both architectures collapse at A-STRICT     | **Floor effect** — obs too impoverished to learn at all. NOT evidence about asymmetry. Triggers §5 gate re-design, not a claim. |

**No metric-fishing rule:** the primary endpoint is eval-return IQM gap. Success
rate, sample-efficiency, per-room SR, and collapse-count are **secondary**,
reported for completeness, and cannot be substituted for the primary endpoint to
manufacture a positive result. If IQM ties but a secondary metric separates, that
is reported as *"tie on primary, suggestive on <metric>, worth confirming"* —
never as headline support.

---

## 5. Pre-sweep gate (do this BEFORE the 40-run commit)

Same discipline as the Phase 1 spawn fix — cheap checks that prevent a wasted sweep:

1. **Build the two wrappers** (A-MILD 14-D, A-STRICT 13-D). Unit-test that the
   emitted vector has the right dims and that removed dims are truly absent (not
   zeroed-but-present, which would still leak positional structure via constant
   inputs).
2. **Strict-subset sanity:** confirm each rung's observation is exactly the
   control minus the removed dims — no feature added, no residual encoding of
   the removed dims (e.g. distance must not be reconstructable from any retained
   dim). A-MILD = control − {9,10}; A-STRICT = control − {9,10,11}.
3. **Symmetric smoke test on A-STRICT (1 seed):** does symmetric PPO still learn
   *anything* at 13-D within 200k? 
   - If it reaches a non-trivial success rate → task is learnable, proceed.
   - If it floors (collapses to collision/timeout) → **floor effect**; A-STRICT
     is too hard. Do NOT run the sweep. Re-design the strict rung milder (dated
     amendment) or extend the budget, and re-gate.
4. **Episode-budget check:** with less information, convergence may be slower.
   Confirm 200k still suffices on the smoke test; bump if the curve is still
   climbing at 200k (and apply the bump to ALL cells for comparability).

Only after gate passes: commit configs, launch the 40-run sweep front-loaded.

---

## 6. Analysis plan (pre-specified)

1. **Per-cell IQM + 95% CI** (rliable), all 6 cells (2 reused control + 4 new).
2. **Gap-of-gaps interaction:** bootstrap the gap at each rung and the
   difference `gap(13-D) − gap(16-D)`; report CI. This is the H-P2b test.
3. **Dose–response figure (the centerpiece):** x = observability rung
   (16-D → 14-D → 13-D, i.e. increasing degradation), y = IQM, two lines
   (symmetric, inverted) with CI bands. If the lines *diverge* as x increases,
   the theory holds and it's visible at a glance.
4. **Collapse-count table** per cell (Phase 1 showed a collision local optimum;
   degraded obs may worsen it — report per cell, don't let IQM hide it).
5. **Secondary metrics table** (success rate, sample-eff, per-room SR) per cell,
   clearly labelled secondary.

---

## 7. What this does NOT include (deferred, on purpose)

- **Variant C (LSTM / RecurrentPPO):** a real implementation change
  (sb3-contrib). Runs only **if** §4 shows an effect worth confirming with
  recurrence. Separate spec when we get there.
- **2A capacity-ratio sweep (125 runs):** deferred until we know asymmetry
  matters *somewhere*. If it does, the sweep runs **in the regime where it
  matters** (likely A-STRICT), not on the easy 16-D task where Phase 1 already
  showed everything ties.
- **2B procedural generalisation:** depends on the collaborator's `procedural.py`;
  orthogonal to the core theory test. Later.

---

## 8. Commit protocol

- Commit this file **before** any Phase 2 run: `Phase 2: pre-register POMDP
  observability ladder (H-P2a/b, frozen)`.
- Config YAMLs: `configs/ppo_pomdp_{mild,strict}_{sym,inv}.yaml`, hyperparameters
  verbatim from Phase 1 Exp1/Exp4, only net_arch + obs-wrapper differ.
- W&B run names: `ppo_{cell}_seed{N}` (e.g. `ppo_strict_inv_seed3`), project
  `arch-ablations-sparse-reward`.
- Wrappers live in `Environment/obs_variants.py` (new file; does not touch the
  frozen `custom_env.py`).

---

## Amendments

**Amendment 1 — 2026-07-02, before any Phase 2 run.**
The original draft described Rung 1 (A-MILD) as "drop global position (2-D) AND
add a relative-direction vector (2-D)" while labelling it 14-D. These conflict:
−2 +2 = net 16-D, not 14-D (and A-STRICT would be 15-D, not 13-D). Resolution:
**drop the added direction vector; make the ladder pure removal.** Authoritative
rungs are now **16-D → 14-D → 13-D** by strict subset:
- A-MILD (14-D) = control − {dims 9,10} (global position). Keeps distance-to-target.
- A-STRICT (13-D) = control − {dims 9,10,11} (global position + distance).

Rationale: a strict-subset ladder cleanly isolates "target-locating information
removed" as the single manipulated variable; an added direction feature would
re-introduce part of that signal and confound the dose–response. No runs had been
launched at amendment time, so this is a pre-run design fix, not a post-hoc change.

**Amendment 2 — 2026-07-02, after A-STRICT gate, before any sweep.**

*Trigger:* The §5 gate on Rung 2 (A-STRICT, 13-D) returned a clean learnability
PASS but revealed the manipulation was ineffective: symmetric PPO reached IQM
27.80, 100% success, all rooms solved, converged by ~40k — near-optimal and
essentially indistinguishable from the 16-D control (only ep_len rose, 14.8 vs
~13). Removing global position (dims 9,10) and distance-to-target (dim 11) did
**not** make the policy hard. With the symmetric actor already at the reward
ceiling, there is no headroom for the inverted actor to open a gap, so the
pre-registered 40-run sweep on rungs 0–2 would return a predictable, uninformative
gap≈0 ("false null"). This is exactly the §1 amendment-hook contingency ("Rung 2
still ties → add a Rung 3 that degrades further").

*Diagnosis (literature-grounded):* Removing *static* features leaves the policy
solvable **reactively** — proximity + region one-hot + target one-hot still
suffice to pick a good action each step. The POMDP literature is consistent that
policy-hardness (and the memory/capacity demand that goes with it) comes from
**temporal hidden state that must be integrated over time**, not from static
feature removal (Hausknecht & Stone 2015, flickering Pong; Flickering MuJoCo;
POPGym — low-dim feature POMDPs "not difficult enough for modern deep RL" unless
temporal structure is injected). Perceptual aliasing from dropping position was
insufficient because the target/region cues still de-alias most states.

*Confound guarded against:* Pure flickering makes the task **memory-hard**, which
the literature attributes to *recurrence* (LSTM), not feedforward actor depth.
Testing feedforward asymmetry (H1's knob) under pure flickering would test the
wrong mechanism. **Frame-stacking (k=4)** converts temporal integration into a
static-but-complex function over stacked history, representable by a feedforward
MLP — making the *policy* a genuinely harder function (the regime H1 needs) while
keeping value relatively smooth, and testing depth fairly. Frame-stack is the
standard memoryless approach to POMDPs (Mnih 2015, 4-frame stack).

*New rung (extends the ladder; strict-harder than Rung 2):*

- **Rung 3a — A-STRICT + frame-stack k=4, NO flicker.** Obs = last 4 A-STRICT
  (13-D) frames stacked = 52-D. Fully observable per-window; tests whether policy
  *complexity alone* (function over history width) separates architectures.
- **Rung 3b — A-STRICT + flicker(p=0.5) + frame-stack k=4.** Each step the 13-D
  frame is fully revealed or fully zero-masked with p=0.5, then the last 4
  (possibly-masked) frames are stacked = 52-D. True POMDP: the agent must
  integrate across masked steps. This is the memory-hard regime and the natural
  launch point for the Variant C (RecurrentPPO/LSTM) ablation later.

*Gate before sweep:* Rung 3b is re-gated with the same single-seed symmetric
check as A-STRICT. Target outcome is **hard-but-learnable** — symmetric PPO
clearly sub-ceiling (guideline: success ≈ 40–75%, or eval reward well short of
~27.8) but improving, indicating headroom for architecture to matter. If it
floors (both fail) → flicker p too high / k too small; soften (lower p, raise k)
and re-gate. If it ceilings (≈100% again) → still too easy; escalate (raise p, or
aliasing) and re-gate. Only build the sweep from a gate in the hard-but-learnable
band.

*Sweep, once gate passes:* the decisive factorial moves to the hard regime:
{symmetric, inverted} × {Rung 3a, Rung 3b} × 10 seeds = 40 runs, with Rung 2
(13-D) and Rung 0 (16-D) as the easy-end anchors of the dose–response (reuse
existing runs where available; Rung 2 symmetric gate seed can seed that cell).
Primary endpoint unchanged: inverted−symmetric IQM gap, and whether it grows
as the task enters the policy-hard regime.

*Frame-stack note:* use SB3 `VecFrameStack(k=4)` over the A-STRICT-wrapped env,
or an equivalent stacking wrapper; flicker wrapper applies the mask BEFORE
stacking so masked frames enter the stack as zeros. observation_space rebuilt to
the stacked shape. No change to env internals or reward.

**Amendment 3 — 2026-07-02, after the full flicker calibration, before any sweep.**

*Trigger:* The full flicker calibration (p ∈ {0.5, 0.7, 0.8}, k=4), each read **at
convergence** (500k budget-bump where needed), yields **no hard-but-learnable
regime**: p=0.5 ceilings fast (~60k), p=0.7 ceilings slow (converges ~240–340k to
100% success on all three rooms, IQM ~26.2 — a flicker path-tax below the 27.8
ceiling, not real difficulty), and p=0.8 fractures into a near/far give-up local
optimum (kitchen+bathroom abandoned via timeout, bedroom solved). So flicker either
delays the ceiling or induces pathology — it does not make the policy genuinely
hard. Track 1's difficulty model (`Analysis/rung3_difficulty_model.py`, BFS +
exact consecutive-mask survival) shows **per-room hardness is
geometry/perceptual-aliasing-driven, not distance- or flicker-driven**: same-H
rooms diverge (kitchen and bedroom both H=11, yet at p=0.8 kitchen 0.00 vs bedroom
1.00 success), and no single p lands all three rooms in-band. The mechanism to test
is therefore **aliasing**, not more flicker.

*New rung — Rung 4 (ALIASING):* Over the A-STRICT (13-D) base, **drop the region
one-hot (dims 13–15)** so that in-room / in-hallway / in-doorway states emit
identical observations and cannot be de-aliased from a single frame. **No flicker,
no frame-stack** — isolate the aliasing lever cleanly. Result: **10-D** = proximity
(5) + target one-hot (4) + remaining-time (1), i.e. control − {9,10,11,13,14,15}.

*Rationale:* this is the exact mechanism Track 1 identified. Removing the region
one-hot makes confusable states (a corridor cell vs a doorway cell vs an in-room
cell) emit the same observation, so the optimal action must depend on **trajectory
history** to disambiguate — genuine policy-function complexity (the "policy-hard,
value-easy" regime H1 needs), not mere input deprivation. Still a strict subset of
control (pure removal), so it stays on the observability ladder.

*Pre-committed interpretation (FIXED before the gate):*

| Rung 4 gate outcome (symmetric) | Meaning / next step |
|---|---|
| **Hard-but-learnable** — all three rooms partial (~40–75%), non-degenerate, converges within budget | Aliasing produced the target regime → **proceed to the 40-run sweep at this setting** ({sym,inv} × {Rung 4, and a paired control} × 10 seeds). |
| **Ceilings** (~100% success, ~27.8) | Even targeted aliasing doesn't make the policy hard → the **null is characterized across all mechanisms** (removal, flicker, aliasing) → write up the null. |
| **Fractures** (near/far degeneracy like p=0.8) **or floors** (collision/timeout collapse) | Aliasing induces pathology, not clean hardness → **write up** (not a clean asymmetry test). |

*Gate:* single-seed symmetric PPO (pi=[256,256]/vf=[256,256], Exp 1 verbatim,
seed 0), **500k** budget (Track 2 showed this map can need >200k; use the same
budget so a slow-learner is not misread as hard). Report the convergence curve
(100k/200k/300k/400k/500k) + per-room breakdown; **per-room shape is the gate
criterion**, not the aggregate.

*Stopping note:* Rung 4 tests the mechanism Track 1 flagged as most promising. The
cap decision (whether this is the final escalation) is **deferred until the gate
result is read**; **no further mechanism is added without an explicit new
decision** — this ladder is not open-ended.

**Amendment 4 — 2026-07-02, after the Rung 4 gate, by explicit new-mechanism decision.**

*Trigger:* Rung 4 (drop region one-hot) **ceilinged immediately** (IQM 27.80, all
three rooms 100% by 20k — faster than A-STRICT). Cause: the 5-D **proximity**
sensors already de-alias room / hallway / doorway cells, so the region one-hot was
redundant. Proximity is therefore the **last remaining de-aliaser** and the only
untested observation lever on this map. Per the Rung-4 stopping note this probe is
an **explicit, approved new-mechanism decision**, not an automatic escalation.

*New rung — Rung 5 (PROXIMITY NOISE):* over the A-STRICT (13-D) base, corrupt the
**5 proximity sensors only** with noise; everything else intact. Proximity is a
**binary** obstacle/off-grid pattern, so noise = **each proximity bit independently
flipped with probability q** (start **q=0.3**). Dimensionality unchanged (13-D).
Reproducible via a dedicated per-episode seeded RNG (as with the flicker wrapper).

*Rationale / why noise not masking:* corrupting (not deleting) proximity degrades
**state identification** — the agent can no longer cleanly resolve cell-type, so the
optimal action must integrate history to disambiguate (policy-hardness) — while
still leaving enough signal to mostly avoid walls. Full masking would conflate
"policy unrepresentable" with "agent literally can't see walls to avoid dying"; noise
separates those. Collision-rate is the tell (see interpretation).

*Pre-committed interpretation (FIXED before the gate):*

| Rung 5 gate outcome (symmetric) | Meaning / next step |
|---|---|
| **Hard-but-learnable** — all rooms partial (~40–75%), converges, non-degenerate | The regime exists (first time in the whole ladder) → **proceed to the sweep**. |
| **Ceilings** (~100%, ~27.8) | Proximity robust to noise; policy stays easy → **observation axis fully exhausted → writeup**. |
| **Collision-floor** — high collision_rate, short ep_len, low success | Proximity's role is wall-avoidance; degrading it **kills the agent** rather than making the policy hard → observation axis exhausted → **writeup**. Report collision_rate prominently to distinguish this from a clean hardness floor. |

*Cap note:* **Rung 5 is the FINAL observation-degradation rung on the 20×20 map.**
Any further test is a different task (Phase 3) under a separate pre-registration —
not an extension of this one. The writeup follows Rung 5 regardless of its outcome.
