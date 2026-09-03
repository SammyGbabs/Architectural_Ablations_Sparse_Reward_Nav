# Paper outline — "Knowing whether a task can test your architecture hypothesis"

**Framing:** method-contribution-first. The headline is a *methodological* result —
observation redundancy can render an architecture hypothesis **untestable**, and we
give a pre-registered protocol to *detect* that before spending a sweep. The H1
inverted-asymmetry null is the **evidence** that motivates and validates the protocol,
not the headline.

**Status:** outline only. Numbers below are placeholders keyed to their evidence
source; every result-grade number must trace to committed multi-seed data (see the
**Provenance** section — this is load-bearing: *no single-seed gate is ever cited as
a result*).

---

## Evidence provenance (READ FIRST — governs every cited number)

Three tiers of evidence. Only the first two may appear as **results**; the third is
**method/design narrative only**.

| Tier | Source | May be cited as a RESULT? | What it backs |
|---|---|---|---|
| **P1-MS** | Phase 1 — 10-seed, committed (`results/csv/p1_*.csv`, rliable) | **Yes** | §4: H1 null (IQM 27.56 vs 27.66, overlapping CIs, P(inv>sym)=0.25); PPO/DQN/baselines; the ~12.5× sample-efficiency audit |
| **P2-MS** | Phase 2 — 10-seed claim-grade cells (`configs/p2_*_sym.yaml`, `Training/run_phase2_sweep.py`) — **PENDING Colab** | **Yes, once run** | §5 cell table: A-STRICT / aliasing / flicker-0.8 fracture / flicker-0.7 slow-ceiling / prox-noise ceilings, each IQM+CI + per-room |
| **GATE** | Single-seed §5 gates + Track-1 difficulty model (`docs/results_log.md` Rung 2–5) | **NO — design decisions only** | Why each rung/parameter was chosen; the calibration trail; the mechanism hypothesis |

**Rule enforced throughout:** §5's *quantitative claims* (ceiling/fracture per cell)
cite **P2-MS**. The single-seed gates appear only in a "how we calibrated the ladder"
paragraph/appendix, explicitly labelled as design decisions. If a P2-MS cell is not
yet run, its claim is marked *pending* and not stated as established.

---

## 1. Abstract
- **Method contribution first:** when a task's observation lets a shallow policy
  already solve it, an actor-capacity hypothesis (e.g. inverted actor–critic
  asymmetry) cannot be tested *by construction* — the precondition ("policy hard to
  represent, value smooth") is unmet. We give a **pre-registered protocol** to detect
  this before committing a sweep: state the precondition, build an observability
  ladder targeting it, single-seed-gate for learnability, multi-seed the load-bearing
  cells, and conclude untestability if no rung meets the precondition.
- **Null as evidence:** applied to indoor grid-navigation, a prior single-seed
  inverted-asymmetry claim does not survive 10-seed evaluation [P1-MS], and **no**
  observation degradation (removal → flicker → aliasing → sensor noise) creates a
  testable regime [P2-MS] — the task is redundantly observed and reactively solvable.
- **Takeaway:** pre-registration + a cheap gating ladder turns "our architecture idea
  didn't win" into "this task couldn't have shown it, and here's how to check."

## 2. Introduction
- Single-seed RL fragility (motivate: same env, seed variance flips conclusions).
- The prior work: inverted actor–critic asymmetry reported to help on this nav task
  (single seed). The natural instinct — "try more architectures" — is p-hacking unless
  disciplined.
- **Two contributions:**
  1. A **replication audit**: multi-seed refutation of the H1 asymmetry claim and the
     ~12.5× sample-efficiency claim [P1-MS].
  2. An **untestability result + protocol**: a pre-registered observability ladder
     shows the task cannot test the hypothesis, and abstracts into a reusable recipe.
- Explicitly frame the null as informative, not a failure.

## 3. Related work
- **RL reproducibility / evaluation:** Henderson et al. (seed fragility); Agarwal et
  al. 2021 (rliable — IQM, stratified-bootstrap CIs, performance profiles). We adopt
  rliable throughout.
- **POMDP difficulty:** Hausknecht & Stone 2015 (flickering Atari); POPGym (low-dim
  feature POMDPs "not hard enough" without temporal structure); frame-stacking (Mnih
  2015) as the memoryless baseline. Motivates our flicker+frame-stack and the finding
  that temporal masking alone doesn't bite here.
- **Architecture asymmetry / actor-critic capacity:** the "policy-hard, value-easy"
  intuition; prior asymmetric actor-critic work. Position our precondition framing.

## 4. Phase 1 — multi-seed replication & refutation  [ALL P1-MS]
- Setup: 20×20 residential nav, `Discrete(5)`, 16-D obs, R(L)=30−0.2L; PPO Exp 1
  (symmetric) vs Exp 4 (inverted) at fixed budget; 10 seeds; rliable IQM+CI.
- **Result 1 — H1 refuted:** IQM 27.56 (inv) vs 27.66 (sym), overlapping 95% CIs,
  P(inv>sym)=0.25 → asymmetry does not help. [P1-MS]
- **Result 2 — sample-efficiency audit:** the original ~12.5× PPO-over-DQN claim
  re-measured under 10 seeds (report the multi-seed ratio + CI). [P1-MS]
- Figures: `p1_iqm_main_configs.png`, `p1_perf_profile_ppo_inverted_vs_symmetric.png`,
  `p1_sample_efficiency_ppo_vs_dqn.png`.
- **Diagnosis (bridge to §5):** the 16-D obs hands the agent global position + distance
  -to-target, so the optimal policy is ~"reduce distance" — shallow-representable.
  Actor depth has nothing to do ⇒ H1 untestable *on the control obs*. This motivates
  the ladder.

## 5. Phase 2 — the pre-registered observability ladder  [table = P2-MS; calibration = GATE]
- **The gating protocol (as applied):** pre-register hypotheses + rungs + falsification
  before any run (`docs/PHASE2_POMDP_PREREGISTRATION.md`, frozen; Amendments 1–4 are
  the dated design trail). Primary endpoint = inverted−symmetric IQM gap; ladder =
  monotone in "policy-relevant information removed."
- **Four mechanisms tried** (each a strict-subset or noise wrapper over the frozen env):
  1. **Pure removal** — A-MILD (14-D), A-STRICT (13-D).
  2. **Flicker + frame-stack** — p∈{0.5,0.7,0.8}, k=4 (52-D).
  3. **Targeted aliasing** — drop region one-hot (10-D).
  4. **Proximity noise** — q=0.3 (13-D).
- **Difficulty model (calibration, GATE):** BFS ground-truth H = {kitchen 11, bedroom
  11, bathroom 22}; exact consecutive-mask survival model; the decisive finding that
  per-room difficulty is **geometry/aliasing-driven, not distance-driven** (same-H
  rooms diverge). Figures `p2_rung3_blackout_vs_p.png`,
  `p2_rung3_observed_vs_predicted.png`. *Labelled as the calibration that directed the
  aliasing rung — not a result.*
- **Claim-grade results table [P2-MS — pending Colab]:** one row per cell, symmetric
  10-seed IQM + 95% CI + per-room SR (+ collision/timeout split for the fracture cell):

  | Cell | obs | IQM (±95% CI) | success | per-room (k/bed/bath) | classification |
  |---|---|---|---|---|---|
  | A-STRICT | 13-D | *pending* | *pending* | *pending* | ceiling |
  | Aliasing | 10-D | *pending* | *pending* | *pending* | ceiling |
  | Flicker p=0.8 | 52-D | *pending* | *pending* | *pending* | **fracture** (near/far give-up) |
  | Flicker p=0.7 | 52-D | *pending* | *pending* | *pending* | slow-ceiling |
  | Prox-noise q=0.3 | 13-D | *pending* | *pending* | *pending* | ceiling (not collision-floor) |

  Anchors: 16-D control (sym) and A-STRICT from Phase-1-adjacent data. **Every number
  here is P2-MS; the single-seed values in the results-log are NOT reproduced as
  results.**
- **Conclusion:** no rung yields hard-but-learnable ⇒ the precondition is never met on
  this task.

## 6. The protocol, generalized  *(the method-first payload — make this a BOXED figure, not prose)*
> **Protocol: is your task able to test your architecture hypothesis?**
> 1. **State the precondition** the hypothesis needs (here: policy hard-to-represent
>    while value stays smooth).
> 2. **Design a ladder** of task variants that monotonically target the precondition
>    (here: progressively remove/degrade the observation that makes the policy easy).
> 3. **Single-seed gate** each rung for *learnability* (cheap; reject floor effects
>    before spending seeds).
> 4. **Multi-seed** (rliable IQM+CI) only the **load-bearing** cells — the ones a claim
>    rests on.
> 5. **Decision:** if some rung is hard-but-learnable → the task can test it → run the
>    architecture sweep *there*. If **no** rung meets the precondition → the task
>    **cannot** test the hypothesis; report that (a null about testability), don't
>    keep tuning architectures on an easy task.

- Emphasise: cost-asymmetry (gates are cheap, sweeps are not); pre-registration
  prevents the ladder from becoming a fishing expedition; the same recipe transfers to
  any capacity/architecture hypothesis, not just actor-critic asymmetry.

## 7. Mechanism — why this task resists all degradation
- **Redundant observation:** proximity already de-aliases room/hallway/doorway, so the
  region one-hot is redundant (aliasing rung ceilinged) [GATE→confirmed P2-MS].
- **Reactive optimality:** a good action is a function of the *current* (recent) frame;
  frame-stacking k=4 lets the agent tolerate p≤0.7 flicker (P(all-k-masked) small).
- **Memorisable fixed map:** with a single fixed layout, the agent navigates from the
  target one-hot + implicit trajectory memory and doesn't need reliable proximity ⇒
  sensor noise is absorbed (prox-noise ceilinged, collisions→0).
- Together: information can be removed/degraded a lot before the *policy* becomes a
  genuinely hard function — so value stays as easy as policy, and asymmetry has no
  wedge.

## 8. What a testable task would need  *(→ Phase 3 hook)*
- Precise property spec for a task that *could* test H1:
  - **Genuine unroutable partial observability** — hidden state that cannot be
    recovered from any single frame *and* is not memorisable (procedural / changing
    layouts, so no fixed-map shortcut).
  - **Longer horizon / credit assignment** — so the policy must compose sub-decisions.
  - **States hard under full information** — the optimal policy is a sharp,
    high-frequency function of state even when fully observed (so depth matters).
- Map to concrete Phase 3 candidates (MiniGrid MultiRoom / DoorKey; procedural layouts;
  dynamic obstacles). Explicit: this is a *separate pre-registration*, not an extension
  of Phase 2 (per Amendment 4 cap).

## 9. Discussion
- **Transferable lesson:** "our architecture didn't help" is often "the benchmark
  couldn't have shown it." A cheap, pre-registered gating ladder distinguishes the two.
- **Pre-registration as methodological contribution:** the dated amendment trail
  (Amendments 1–4) is the artifact — it shows disciplined escalation with frozen
  interpretation tables, the antidote to post-hoc architecture search.
- **Limits:** single map/domain; one architecture pair; the protocol detects
  untestability, it doesn't repair the task (that's §8/Phase 3).
- **Reproducibility:** all configs, seeds, per-seed CSVs, pre-registration, and
  amendment history committed.

---

## Figure/asset manifest for the paper
- §4: `p1_iqm_main_configs.png`, `p1_perf_profile_ppo_inverted_vs_symmetric.png`,
  `p1_sample_efficiency_ppo_vs_dqn.png` [P1-MS].
- §5 calibration: `p2_rung3_blackout_vs_p.png`, `p2_rung3_observed_vs_predicted.png`
  [GATE — calibration, labelled as such].
- §5 results: **new** dose/ladder figure from P2-MS (IQM+CI per cell) — *to generate
  from the Phase 2 sweep CSVs via an `Analysis/` script* (pending P2-MS).
- §6: the boxed protocol (a figure/box, not prose).

## Open dependencies before the paper is result-complete
1. **P2-MS runs** (`Training/run_phase2_sweep.py`) — 40 runs now, +10 (p=0.7 500k) on
   confirm. Until then §5's table is *pending*.
2. **§5 results figure + `Analysis/` generator** — after P2-MS lands.
3. Phase 3 pre-registration (for §8) — separate document.
