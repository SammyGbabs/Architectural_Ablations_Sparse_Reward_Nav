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

## 5. Phase 2 — the pre-registered observability ladder  *(HELD — drafting after cell 5)*

*Numbers and figures ready:* the four claim-grade cells (A-STRICT, aliasing,
prox-noise ceilings; p=0.8 fracture) with `p2_ladder_iqm.png` and
`p2_flicker08_perseed.png` [P2-MS]; the difficulty model / calibration trail [GATE].
*Waiting on:* the p=0.7 flicker-ceiling cell (cell 5) to complete the claim-grade
flicker dose-response before this section is written, so the "flicker ceilings (even
at its hardest) vs fractures" contrast is drawn against multi-seed data on both
sides. Prose to follow.
