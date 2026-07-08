# Paper draft (working)

Prose draft, section by section, against real committed numbers. Structure and
framing follow `docs/paper_outline.md`. **Provenance tags** ([P1-MS], [P2-MS],
[GATE]) mark every evidence source per the outline's provenance table; **`⟦…⟧`**
marks a number still to be filled from committed data (never invent one).

Status: §4 drafted (Phase 1). §5 held until cell 5 (p=0.7) lands — its figures and
the flicker-ceiling row complete first.

---

## 4. Phase 1 — multi-seed replication and the collapse of the single-seed claims

The starting point is a prior report that an *inverted* actor–critic asymmetry — an
actor network deeper than its critic (`π=[512,256,128]`, `v=[256,128]`) — improves
performance over a symmetric baseline (`π=v=[256,256]`) on a sparse-reward indoor
navigation task, at a matched parameter budget. That result was obtained from a
single training seed. Deep RL is notoriously seed-sensitive [Henderson et al.], so
before building on the claim we re-evaluate it under the evaluation protocol the
reproducibility literature now recommends: interquartile-mean (IQM) aggregation with
95 % stratified-bootstrap confidence intervals over many seeds [Agarwal et al.,
*rliable*].

**Setup.** The environment is a 20×20 residential grid (`Discrete(5)` actions, a
16-dimensional observation, reward `R(L)=30−0.2L` for an `L`-step success). We train
the two architectures — symmetric and inverted — with otherwise identical PPO
hyperparameters (verbatim from the original configurations) for a fixed 200k-step
budget, at **10 seeds each**, and aggregate with rliable. [P1-MS]

**Result 1 — the asymmetry advantage does not survive multiple seeds.** Under
10-seed evaluation the two architectures are statistically indistinguishable: IQM
eval return **27.56** (inverted) versus **27.66** (symmetric), with overlapping 95 %
confidence intervals and a probability of improvement **P(inverted > symmetric) =
0.25** — i.e. the inverted configuration is, if anything, *less* likely to beat the
symmetric one on a random seed. [P1-MS] The single-seed advantage was seed noise;
the pre-registered falsification criterion for the asymmetry hypothesis (H1) is met.

**Result 2 — the sample-efficiency claim.** The original report also claimed the
policy-gradient agent was ≈12.5× more sample-efficient than its value-based
counterpart. Re-measured across 10 seeds as env-steps-to-90 %-of-asymptotic-IQM, the
multi-seed ratio is ⟦ratio ± CI, from `results/csv/p1_*.csv`⟧, which ⟦confirms /
substantially shrinks⟧ the original figure. [P1-MS] *(Number pending the archival
Phase 1 aggregation; the claim is stated here only to be filled, not asserted.)*

Figures: `p1_iqm_main_configs.png` (per-config IQM + CI),
`p1_perf_profile_ppo_inverted_vs_symmetric.png` (H1 performance profile),
`p1_sample_efficiency_ppo_vs_dqn.png`.

**Why the advantage vanished — the diagnosis that motivates the rest of the paper.**
A null is not automatically informative: the asymmetry hypothesis could be false, or
the *task* could be unable to test it. Inspecting the observation makes the second
reading concrete. The 16-D observation hands the agent its normalised global position
and its distance-to-target; the reward is monotone in path length. The optimal policy
is therefore close to "take the action that most reduces distance-to-target" — a
smooth, low-complexity function that a shallow network represents as well as a deep
one. The theory under test ("policy-hard, value-easy": actor depth helps only when the
optimal *policy* is a genuinely hard function while the *value* stays smooth) predicts
**no** benefit from actor depth precisely when the policy is easy. On this observation
the precondition for the hypothesis is unmet *by construction*, so a null is exactly
what the theory predicts — and tells us nothing about the hypothesis itself. The
question this raises — *can this task test the hypothesis at all, under any
observation?* — is what §5 answers with a pre-registered observability ladder, and
what §6 abstracts into a general protocol.

---

## 5. Phase 2 — a pre-registered observability ladder shows the task cannot test H1

§4 leaves a specific question: is the null a fact about the *hypothesis* or about the
*task*? To answer it we ask whether **any** degradation of the observation can put the
task into the regime the hypothesis needs — a policy that is genuinely hard to
represent while the value function stays smooth — without simply making the task
unlearnable. We answer this with a **pre-registered** experiment (the design,
hypotheses, falsification criteria, and per-outcome interpretations were frozen
before any run; the dated amendment trail records every subsequent design decision).
The independent variable is a *ladder* of observation wrappers over the frozen
environment, monotone in "policy-relevant information removed"; the primary endpoint
is the inverted−symmetric IQM gap. Because a floor effect (task simply unlearnable)
would masquerade as the target regime, each rung is first **single-seed-gated for
learnability** (cheap) before any load-bearing cell is promoted to 10 seeds (rliable).
This section reports the claim-grade cells; the single-seed gates and the difficulty
model are design/calibration steps [GATE], never cited as results.

**Four degradation mechanisms.** Guided by the POMDP-difficulty literature (static
feature removal is often insufficient; temporal hidden state is what makes policies
hard [Hausknecht & Stone; POPGym]), we test, over the frozen env: (i) **pure removal**
— drop global position (A-MILD, 14-D) and additionally distance-to-target (A-STRICT,
13-D); (ii) **flicker + frame-stack** — each step the frame is fully masked with
probability `p`, then the last `k=4` frames are stacked (52-D); (iii) **targeted
aliasing** — drop the region one-hot so room/hallway/doorway cells are indistinguishable
from a single frame (10-D); (iv) **proximity noise** — flip each binary proximity bit
with probability `q=0.3` (13-D), degrading state-identification while leaving walls
mostly avoidable.

**A calibration aside (why flicker rate is not the difficulty knob).** [GATE] A
ground-truth BFS on the map gives optimal path lengths `H` = {kitchen 11, bedroom 11,
bathroom 22}, and an exact consecutive-mask survival model predicts blackout
probability rising with `H`. The observed per-room difficulty **contradicts** this:
under flicker the two equidistant rooms diverge sharply (at `p=0.8`, kitchen is
abandoned while bedroom is solved), so difficulty is **geometry/approach-dependent,
not distance-dependent** — no single flicker rate places all rooms in a partial band.
This directed the aliasing rung and is reported as calibration, not as a result
(figures `p2_rung3_blackout_vs_p.png`, `p2_rung3_observed_vs_predicted.png`).

**Claim-grade results.** [P2-MS] For a *standard* (symmetric) PPO agent — Exp 1
hyperparameters verbatim, 10 seeds, rliable IQM + 95% CI — every cell either ceilings
or fractures; none is hard-but-learnable (Table 2; `p2_ladder_iqm.png`):

| Cell | obs | steps | IQM eval return (95% CI) | per-room SR (k/bed/bath) | outcome |
|---|---|---|---|---|---|
| A-STRICT | 13-D | 200k | 27.24 [27.21, 27.27] | 1.00/1.00/1.00 | ceiling |
| Aliasing | 10-D | 200k | 27.17 [27.04, 27.37] | 1.00/1.00/1.00 | ceiling |
| Prox-noise q=0.3 | 13-D | 200k | 26.78 [26.41, 26.93] | 1.00/1.00/1.00 | ceiling |
| Flicker p=0.7 | 52-D | 500k | 26.14 [25.42, 26.32] | 1.00/1.00/1.00 | ceiling (slow) |
| Flicker p=0.8 | 52-D | 200k | 8.95 [6.29, 13.01] | 0.00/1.00/0.33 | **fracture** |

Three observations are decisive. **First, the ceilings are real ceilings, not slow
learners or wall-avoidance floors:** the four ceiling cells reach 100% success on all
three rooms with tight CIs, prox-noise does so at a collision rate of 0.00 (the noise
is absorbed, not lethal), and even the hardest *learnable* flicker (`p=0.7`) reaches
the ceiling given budget — 9/10 seeds at full success, the ~1.6 gap below 27.8 being a
path-length tax from occasional blackouts, not sub-ceiling difficulty. **Second, the
one cell that leaves the ceiling does not become hard-but-learnable — it fractures:**
at `p=0.8`, IQM collapses to 8.95 and the failure is a *systematic give-up*, robust
across all 10 seeds — **kitchen is abandoned in 10/10 seeds and bedroom solved in
10/10** (`p2_flicker08_perseed.png`), with failures being timeouts, not collisions.
That is a degenerate local optimum, not the graceful partial competence a fair
architecture test needs. **Third, this exhausts the axis:** across removal, flicker,
aliasing, and sensor noise — the pre-registered cap on single-map observation
degradation — no rung meets the precondition. On this task, the H1 asymmetry
hypothesis is **untestable**: the null of §4 is a fact about the task, not (on this
evidence) about the hypothesis.
