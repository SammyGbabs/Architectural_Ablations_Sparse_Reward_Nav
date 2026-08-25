# Testing whether a task can test an architecture hypothesis

*(Working title. Draft prose; the TMLR LaTeX port lives in `paper/main.tex` and is
regenerated from this file.)*

**Notation left visible for the writing pass.** Bracketed provenance tags `[P1-MS]` /
`[P2-MS]` / `[GATE]` mark each number's evidence source: multi-seed Phase 1, multi-seed
Phase 2, or single-seed gate, the last never cited as a result. Figures are referenced by
number; the image files live in `figures/` (filename given in each caption).

---

## 1. Abstract

Empirical architecture comparisons carry an implicit assumption: that the benchmark can
actually *test* the hypothesis, that the task contains the structure the architecture is
meant to exploit. When it does not, a null is uninformative and a single-seed positive is
unfalsifiable noise. Yet both are routinely reported as evidence. We give a
**pre-registered protocol for deciding whether a task can test an architecture hypothesis
at all**, before trusting any comparison run on it: state the structural precondition the
hypothesis requires; build a graded manipulation ladder that targets it; gate each rung
for learnability with a single seed; promote only load-bearing rungs to multi-seed,
claim-grade evaluation; and attribute any induced hardness to the precondition rather than
to a confound before comparing architectures. We instantiate the protocol on a claim from
our own earlier, unpublished work: that an *inverted* actor–critic asymmetry, a deeper
actor than critic, improves sparse-reward indoor navigation. Under multi-seed evaluation
the single-seed advantage disappears (IQM 27.56 vs 27.66, overlapping 95% CIs,
P(inverted > symmetric) = 0.25). The ≈12.5× PPO sample-efficiency advantage we had earlier
reported likewise does not survive a defined, step-based, multi-seed measurement: four of
the five DQN configurations reach 90% of asymptotic return faster than any PPO
configuration, the slowest configuration in either family is a PPO one, and the two
families' step-to-90% distributions overlap. There is no resolvable PPO advantage, and the
point estimates if anything favour DQN. A pre-registered observability ladder, four
degradation mechanisms extended by dated amendment as each proved insufficient, then shows
*why* the asymmetry question cannot be settled here: no manipulation places the task in the
regime the hypothesis needs. Every rung either **ceilings** (the optimal policy stays easy
to represent), **fractures** into a give-up local optimum, or **fragments** across
seed-selected local optima, hard but not in the way the hypothesis specifies. We report
this as a worked *negative outcome of the protocol*, not as evidence against asymmetry:
this task class cannot arbitrate the hypothesis, which may still hold on a task meeting the
specification we derive. The contribution is the reusable check, and the finding that a
common style of benchmark silently fails it.

## 2. Introduction

The reproducibility literature has taught deep reinforcement learning to distrust a single
run. Seed variance can flip a conclusion, and much reported progress has not survived
multi-seed re-evaluation [Henderson et al.; Agarwal et al.]; the accepted remedy is to
report interquartile means with stratified-bootstrap confidence intervals rather than point
estimates. This closes one failure mode, mistaking noise for signal, but leaves a prior one
untouched. A comparison can be statistically impeccable and still meaningless: if the
benchmark does not contain the structure the architecture is meant to exploit, multi-seed
statistics measure, with great precision, an effect the task was never capable of showing.
And the two cases are observationally identical from inside the experiment. A null on a
task where the effect is genuinely absent and a null on a task that could never have
expressed it produce the same numbers. rliable can tell you a comparison is statistically
sound; it cannot tell you it is meaningful. Pre-registration can tell you that you did not
fish for the result; it cannot tell you the pond contains fish. That gap, between a sound
measurement and a meaningful one, is the subject of this paper.

This gap is not unnamed. In measurement theory it is a question of *construct validity*,
whether an instrument measures the phenomenon it purports to, and a growing literature
applies that lens critically to machine-learning benchmarks. That literature is largely
diagnostic, identifying validity failures in benchmarks after the fact; our contribution is
a constructive counterpart, an executable test a researcher can run *before* trusting a
comparison.

To see the shape of the gap concretely: an architecture hypothesis almost always asserts
that *architectural property `A` helps because it suits some structure `S` of the problem*.
Inverted actor–critic asymmetry suits a policy that is hard to represent, convolution suits
spatial locality, attention suits long-range dependency. Such a hypothesis is only testable
on a task that actually contains `S`. Where `S` is absent, `A` has nothing to act on: the
hypothesis predicts no effect *by construction*, so a null is uninformative and any
single-seed win is noise dressed as evidence. Nothing in a standard train-and-evaluate loop
flags this. The task fails to pose the question, and the experiment answers a different
one, both without any visible symptom.

We arrived at this through a concrete claim of our own. In earlier, unpublished experiments
of ours, an inverted actor–critic asymmetry, a policy network deeper than its value network
(`π=[512,256,128]` vs `v=[256,128]`), appeared to outperform a symmetric baseline at
matched budget on a sparse-reward indoor-navigation task, on a single seed, together with
an apparent ≈12.5× sample-efficiency advantage of PPO over DQN. Both claims come apart on
re-examination. But *how* they come apart is the more useful observation, and is what
pointed us at the gap above: one claim asked a question the task could not answer, the
other a question the measurement could not answer.

This paper makes two contributions. **First, a reusable protocol** (§6) for deciding
whether a task can test a given architecture hypothesis, before trusting any comparison run
on it: name the precondition `S`, build a manipulation ladder that targets it, gate rungs
for learnability cheaply, spend seeds only on load-bearing rungs, and, before comparing
architectures, confirm any induced hardness is attributable to `S` rather than a confound.
Pre-registration and the cheap-gate/expensive-seed split are integral to it, not
incidental. **Second, a worked negative outcome**: applied to the navigation task, the
protocol shows the reported asymmetry advantage does not survive multiple seeds, and that
*no* pre-registered degradation of the task induces the regime the hypothesis needs. The
task stays reactively, redundantly, memorably easy, or it breaks the wrong way, fracturing
or fragmenting across seed-selected local optima. Throughout, this is a statement about
*the task*, not the hypothesis: we show this task class cannot test inverted asymmetry, and
specify (§8) what a task that could would need, not that asymmetry fails in reinforcement
learning.

## 3. Related work

**Reproducibility and evaluation in deep RL.** A body of work has documented that deep
reinforcement learning results are highly sensitive to random seeds, implementation details,
and hyperparameters, and that single-run comparisons frequently fail to replicate
[Henderson et al., 2018]. The now-standard response is to report distributional statistics,
interquartile means with stratified-bootstrap confidence intervals, rather than point
estimates [Agarwal et al., 2021]. We adopt this machinery throughout. Our concern, however,
is orthogonal to it: these methods establish whether a measured difference is *statistically*
real, not whether the task was capable of producing the difference in the first place. A
comparison can satisfy every statistical standard and still be uninformative about the
hypothesis it purports to test.

**Construct validity and benchmark critique.** That benchmarks may not measure what they are
taken to measure is not a new observation. Under the banner of *construct validity*, a
concept originating in psychological measurement theory, which asks whether a test captures
the phenomenon it claims to [Cronbach & Meehl, 1955], and recently imported into
machine-learning evaluation critique [Raji et al., 2021], a growing literature examines the
epistemic foundations of ML evaluation. Recent work develops explicit conditions of construct
validity for predictive benchmarking [Freiesleben & Zezulka, 2025] and, through large-scale
systematic review, documents pervasive validity weaknesses across hundreds of benchmarks
[Bean et al., 2025]. This literature is, however, almost entirely *diagnostic*: it reviews
existing benchmarks, names the ways their validity fails, and issues design recommendations.
What it does not provide is a constructive, executable procedure that a researcher can run
*before* trusting a particular comparison, to establish whether a specific task can test a
specific hypothesis. Our contribution is exactly that instrument, applied to one
sharply-defined validity question: whether a task contains the structure `S` that an
architecture hypothesis requires. The outcome is pre-registered and pass/fail. We move the
construct-validity concern from retrospective critique to a prospective test.

**Asymmetric actor–critic architectures.** The assumption that actor and critic should share
topology and capacity has been questioned from more than one direction. A recent line studies
the *small-actor* regime, finding that shrinking the actor relative to the critic tends to
degrade performance through value underestimation and poor data collection, and that
critic-side optimism can recover it [Mastikhina et al., 2025]. A different sense of asymmetry
appears in image-based robot learning, where the critic is given privileged full state while
the actor sees only observations [Pinto et al., 2018]; that is an asymmetry of *information*
rather than of capacity, and it establishes that the field has explored several distinct
asymmetry axes. The hypothesis we examine concerns capacity, in the *opposite* direction to
the small-actor line: a policy network deeper than its value network ("policy-hard,
value-easy"), motivated by the intuition that actor depth should help when the optimal policy
is a harder function to represent than the optimal value. These directions of asymmetry are
not in tension. That capacity asymmetry can produce pronounced, mechanistically-explained
effects is established on tasks that contain the relevant structure: Mastikhina et al. report
clear effects on the control benchmarks they study. Their result is therefore a useful
counterpoint to ours, an instance of a task that *can* express a capacity-asymmetry effect,
and sharpens rather than undercuts our claim, which is not that asymmetry never matters but
that whether a *given* task can reveal it must itself be tested.

**Partial observability and task difficulty.** Our manipulation ladder degrades observability
in an attempt to induce a policy that is hard to represent. The canonical construction for
making a task memory-hard is temporal masking, the flickering setup that converts an MDP into
a POMDP by stochastically obscuring observations [Hausknecht & Stone, 2015], which we combine
with frame-stacking [Mnih et al., 2015] so that any resulting hardness is representable by a
feedforward policy rather than requiring recurrence. The stacking-versus-recurrence tradeoff
is itself well studied: a fixed stack supplies a bounded window of history at no
architectural cost, whereas recurrence is needed only when the relevant dependency exceeds
that window [Kapturowski et al., 2019]. Recent benchmark work emphasises that
low-dimensional feature-vector POMDPs are often insufficiently difficult for modern deep RL
unless temporal structure is deliberately injected [Morad et al., 2023], a caution our results
bear out directly.

**Pre-registration in machine learning.** Pre-registration, fixing hypotheses, protocol, and
interpretation criteria before observing results, is long established in medicine and
psychology, where registered reports are an accepted publication model [Chambers, 2013], but
has seen comparatively little uptake in machine learning. Dedicated efforts to introduce it
into machine learning include the NeurIPS Pre-registration Workshops [Bertinetto et al., 2021;
Albanie et al., 2022]. Their motivation is closely aligned with ours: an incentive structure
rewarding only positive results suppresses informative negative findings and discourages
rigorous experimental design. We adopt pre-registration not as a formality but as a
load-bearing component of the method: because the protocol involves searching over task
manipulations until one induces the required regime, freezing the ladder and a per-outcome
interpretation table in advance is what prevents that search from degenerating into selection
of whichever manipulation happens to favour a given architecture.

---

## 4. Phase 1 — the reported effects do not survive, and their disappearance is not yet an answer

We begin by holding the two claims from our earlier unpublished experiments to the evaluation
standard the reproducibility literature now recommends: interquartile-mean (IQM) aggregation
with 95 % stratified-bootstrap confidence intervals over many seeds [Agarwal et al.,
*rliable*], across ten seeds each. Both dissolve. The purpose of this section, though, is not
the dissolution itself but what it does and does not license. A dissolved effect on this task
turns out to say nothing about the hypotheses behind it, and that emptiness is the first
concrete symptom of the gap this paper is about.

**Setup.** The environment is a 20×20 residential grid (`Discrete(5)` actions, a
16-dimensional observation, reward `R(L)=30−0.2L` for an `L`-step success). Reward arrives
only on reaching the target room, making this a sparse-reward problem, a setting long
recognised as hard for value-based and policy-gradient methods alike because uninformative
early experience gives the learner almost no gradient to follow [Andrychowicz et al., 2017].
We train the two architectures, symmetric and inverted, with otherwise identical PPO
hyperparameters (verbatim from those earlier experiments' configurations) for a fixed
200k-step budget, at **10 seeds each**, and aggregate with rliable. [P1-MS] Two aggregation
conventions hold throughout the paper. A configuration's return is the IQM across seeds of
each seed's mean eval return, with a 95% stratified-bootstrap CI; the bootstrap is seeded, so
every interval reported here regenerates exactly from the released per-seed data.
**Per-room success-rate aggregates are computed as the mean across seeds of each seed's own
SR fraction**, a seed-mean rather than a pooled per-episode rate. Because eval target rooms
are sampled rather than balanced, seeds carry unequal per-room episode counts, and the
seed-mean keeps every aggregate arithmetically consistent with the per-seed tables: averaging
a per-seed column reproduces its aggregate.

**The nine configurations, and why DQN is among them.** The two architectures above are
the endpoints of a four-point PPO sweep over the actor:critic capacity ratio at fixed
budget; the other five configurations are DQN and carry no asymmetry signal at all. They
are present because Phase 1 holds *two* claims from the earlier work to account, and the
second (Result 2, below) is a PPO-vs-DQN sample-efficiency comparison, so the five DQN
runs are the value-based comparator family for that claim, not part of the asymmetry
test. The PPO configurations vary essentially only the actor:critic ratio; the DQN
configurations (a single Q-network, no actor–critic split) vary width, discount,
exploration schedule, and buffer, and include two deliberately destabilised settings
retained to replicate the original paper's DQN table faithfully. All nine share the
frozen environment, the 200k-step budget, and 10 seeds.

**Table 1.** The nine Phase-1 configurations. PPO spans the actor:critic ratio (H1); DQN
is the value-based comparator family for the sample-efficiency claim (Result 2).

| Family | Config | Network (actor π / critic v; DQN: Q-net) | Ratio / role |
|---|---|---|---|
| PPO (actor–critic; H1) | Exp 1 | π [256,256] / v [256,256] | symmetric 1:1, the control |
|  | Exp 2 | π [256,256] / v [512,256] | conventional (wider critic) |
|  | Exp 3 | π [128,128] / v [512,512,256] | conventional (deep critic) |
|  | Exp 4 | π [512,256,128] / v [256,128] | **inverted** (deeper actor), headline |
| DQN (value-based; comparator) | Exp 1 | Q [256,256] | slow ε-decay |
|  | Exp 2 | Q [512,256], γ=0.995 | discount variant |
|  | Exp 3 | Q [512,256] | aggressive ε-decay, high lr (destabilised) |
|  | Exp 4 | Q [1024,512] | low lr, high γ (unstable) |
|  | Exp 5 | Q [512,256], γ=0.99 | original best DQN |

**Result 1 — the asymmetry advantage does not survive multiple seeds.** Under
10-seed evaluation the two architectures are statistically indistinguishable: IQM
eval return **27.56** (inverted) versus **27.66** (symmetric), with overlapping 95 %
confidence intervals and a probability of improvement **P(inverted > symmetric) =
0.25**. The inverted configuration is, if anything, *less* likely to beat the
symmetric one on a random seed. [P1-MS] The single-seed advantage was seed noise;
the pre-registered falsification criterion for the asymmetry hypothesis (H1) is met.

**Result 2 — the sample-efficiency claim was the wrong kind of measurement.** Those same
earlier experiments also reported the policy-gradient agent (PPO) to be ≈12.5× more
sample-efficient than the value-based agent (DQN), on the basis that PPO reached stable high
performance in roughly 20 episodes against nearly 250 for DQN. That is, `12.5× = 250 / 20`,
a ratio of **episodes-to-visual-convergence** read off single-run learning curves. This
quantity is not a sound cross-algorithm sample-efficiency measure, for three reasons.
**(a) Episodes are not a common currency of experience.** Episode length on this task varies
five- to ten-fold with policy quality: a wandering early-training agent runs to the 150-step
timeout while a converged agent finishes in ~14 steps, so "one episode" purchases very
different amounts of environment interaction at different points in training. For two
algorithms whose episode-length trajectories differ, an episode-count ratio therefore
conflates sample efficiency with episode duration; the currency the agent actually spends,
and that the two algorithms share, is *environment steps*. **(b) "Stable high performance"
is eyeballed.** A convergence point identified by eye has no defined threshold, and, read off
a single curve, carries no uncertainty. **(c) It is a single run.** Our own H1 data shows how
misleading single runs are on this task: one *inverted*-configuration seed (Exp 4, seed 8)
collapses to a return of 3.6, a collision local optimum in which terminating the episode
early is easier to find than reaching a distant goal. Its nine siblings sit at ~27.6, and
the symmetric configuration has
no collapsed seed at all. This asymmetric collapse is the only asymmetry-shaped signal
anywhere in the H1 data.

We re-measured with a defined, step-based, multi-seed statistic: environment steps to reach
90% of the asymptotic eval-return IQM, over 10 seeds, aggregated with rliable [Agarwal et
al.]. No advantage remains. To keep the comparison from turning on a chosen exemplar we
report the full within-family distribution rather than a single pair. The four PPO
configurations reach the threshold at IQMs of 36.7k (Exp 1), 36.7k (Exp 2), 163.3k (Exp 3),
and 32.0k (Exp 4) steps; the five DQN configurations at 21.7k, 20.0k, 20.0k, 45.0k, and
26.7k, with overlapping confidence intervals throughout (e.g. PPO Exp 1 [26.7k, 40.0k], DQN
Exp 2 [14.0k, 74.0k]). [P1-MS] The two families occupy the same ≈20–45k band. The single
configuration that leaves it, PPO Exp 3 at 163.3k steps, is the *slowest* of either family,
so on an exemplar-independent reading PPO is no faster than DQN, and the one salient
difference runs opposite to the claimed 12.5× advantage.

These two results are two instances of a single pattern. The asymmetry claim put a question
to a *task* that could not answer it, a benchmark lacking the structure the hypothesis is
about. The sample-efficiency claim put a question to a *measurement* that could not answer
it, a metric that cannot separate sample efficiency from episode duration, nor one run's
eyeballed convergence point from noise. In both cases the instrument was incapable of
resolving the question asked of it; what differs is only which instrument failed, the *task*
or the *metric*. Neither number is yet an answer, and the rest of the paper is therefore
concerned less with re-scoring individual claims than with the prior question of whether the
task and the measurement can support the claim at all.

**Figure 1.** Per-configuration IQM eval return with 95% stratified-bootstrap CIs across the
nine main configs (PPO Exp 1–4, DQN Exp 1–5), 10 seeds each. (`figures/p1_iqm_main_configs.png`)

**Figure 2.** H1 performance profile: fraction of runs exceeding a return threshold, inverted
(Exp 4) vs symmetric (Exp 1); the two curves overlap across the range.
(`figures/p1_perf_profile_ppo_inverted_vs_symmetric.png`)

**Figure 3.** Sample efficiency: environment-steps to 90% of asymptotic IQM (IQM + 95% CI),
PPO vs DQN; the two families' distributions overlap, the one outlier being a *slow* PPO
config. (`figures/p1_sample_efficiency_ppo_vs_dqn.png`)

**Why the advantage vanished — the diagnosis that motivates the rest of the paper.**
A null is not automatically informative: the asymmetry hypothesis could be false, or
the *task* could be unable to test it. Inspecting the observation makes the second
reading concrete. The 16-D observation hands the agent its normalised global position
and its distance-to-target; the reward is monotone in path length. The optimal policy
is therefore close to "take the action that most reduces distance-to-target", a
smooth, low-complexity function that a shallow network represents as well as a deep
one. The theory under test ("policy-hard, value-easy": actor depth helps only when the
optimal *policy* is a genuinely hard function while the *value* stays smooth) predicts
**no** benefit from actor depth precisely when the policy is easy. On this observation
the precondition for the hypothesis is unmet *by construction*, so a null is exactly
what the theory predicts, and tells us nothing about the hypothesis itself. What it
raises instead is a sharper question: can this task test the hypothesis at all, under
any observation?

---

## 5. Phase 2 — a pre-registered observability ladder shows the task cannot test H1

Can **any** degradation of the observation put the task into the regime the hypothesis
needs, a policy that is genuinely hard to represent while the value function stays
smooth, without simply making the task unlearnable? We answer this with a
**pre-registered** experiment (the design, hypotheses, falsification criteria, and
per-outcome interpretations were frozen before any run; the dated amendment trail records
every subsequent design decision). The independent variable is a *ladder* of observation
wrappers over the frozen environment, monotone in "policy-relevant information removed";
the primary endpoint is the inverted−symmetric IQM gap. Because a floor effect (task
simply unlearnable) would masquerade as the target regime, each rung is first
**single-seed-gated for learnability** (cheap) before any load-bearing cell is promoted to
10 seeds (rliable). This section reports the claim-grade cells; the single-seed gates and
the difficulty model are design and calibration steps [GATE], never cited as results.

**Four degradation mechanisms.** Guided by the POMDP-difficulty literature (static
feature removal is often insufficient; temporal hidden state is what makes policies
hard [Hausknecht & Stone; POPGym]), we test, over the frozen env: (i) **pure removal**,
dropping global position (A-MILD, 14-D) and additionally distance-to-target (A-STRICT,
13-D); (ii) **flicker + frame-stack**, where each step the frame is fully masked with
probability `p`, then the last `k=4` frames are stacked (52-D); (iii) **targeted
aliasing**, dropping the region one-hot so room/hallway/doorway cells are indistinguishable
from a single frame (10-D); (iv) **proximity noise**, flipping each binary proximity bit
with probability `q=0.3` (13-D), degrading state-identification while leaving walls
mostly avoidable.

**A calibration aside (why flicker rate is not the difficulty knob).** [GATE] A
ground-truth BFS on the map gives optimal path lengths `H` = {kitchen 11, bedroom 11,
bathroom 22}, and an exact consecutive-mask survival model predicts blackout
probability rising with `H`. The observed per-room difficulty **contradicts** this:
under flicker the two equidistant rooms diverge sharply (at `p=0.8`, kitchen is
abandoned while bedroom is solved), so difficulty is **geometry/approach-dependent,
not distance-dependent**, and no single flicker rate places all rooms in a partial band.
This directed the aliasing rung and is reported as calibration, not as a result
(Figures 4–5).

**Figure 4.** P(disabling blackout ≥1 per episode) versus flicker probability `p` at `k=4`,
one curve per room using its real BFS path length `H`, with the "hard-but-learnable" band
shaded. (`figures/p2_rung3_blackout_vs_p.png`)

**Figure 5.** Model-predicted (distance-`H`-based) per-room difficulty vs observed per-room
difficulty at `p=0.5/0.7/0.8`; the observed ordering does not follow `H`, so difficulty is
geometry-, not distance-, driven. (`figures/p2_rung3_observed_vs_predicted.png`)

**Claim-grade results.** [P2-MS] For a *standard* (symmetric) PPO agent, using Exp 1
hyperparameters verbatim, 10 seeds, rliable IQM + 95% CI, every cell either ceilings,
fractures, or fragments across seed-selected local optima; none is hard-but-learnable
(Table 2; Figure 6):

**Table 2.** Phase 2 claim-grade ladder cells: symmetric PPO, **10 seeds per cell**,
rliable IQM + 95% CI; per-room SR is the seed-mean defined in the Setup above. All rows are
claim-grade [P2-MS]; the single-seed learnability gates that preceded them are
calibration [GATE] and appear nowhere in this table. "Patterns" counts the distinct
per-seed (kitchen, bedroom, bathroom) SR combinations observed across the ten seeds: 1
means every seed landed in the same behavioural regime, and a large count means the
seeds scattered.

| Cell | obs | steps | IQM eval return (95% CI) | per-room SR (k/bed/bath) | patterns | behavioural regime |
|---|---|---|---|---|---|---|
| A-STRICT | 13-D | 200k | 27.24 [27.21, 27.27] | 1.00/1.00/1.00 | 1/10 | ceiling |
| Aliasing | 10-D | 200k | 27.17 [27.04, 27.37] | 1.00/1.00/1.00 | 1/10 | ceiling |
| Prox-noise q=0.3 | 13-D | 200k | 26.78 [26.41, 26.93] | 1.00/1.00/1.00 | 1/10 | ceiling |
| Flicker p=0.7 | 52-D | 500k | 26.14 [25.41, 26.32] | 1.00/1.00/0.93 | 2/10 | ceiling (slow) |
| Flicker p=0.8 | 52-D | 200k | 8.95 [6.43, 13.01] | 0.00/1.00/0.33 | 3/10 | **fracture** |
| Flicker p=0.8 | 52-D | 500k | 13.24 [7.79, 18.91] | 0.50/0.75/0.83 | 6/10 | **fragmentation** |

**Figure 6.** The ladder cells of Table 2 plotted: per-cell IQM eval return + 95% CI, four
ceilings clustered near the 27.8 optimal-path ceiling, the `p=0.8` fracture at 8.95 (200k),
and its matched-budget fragmentation at 13.24 (500k), whose interval overlaps the
fracture's. Per-seed detail for the 500k cell is in Appendix A. (`figures/p2_ladder_iqm.png`)

Three observations are decisive. **First, the ceilings are real ceilings, not slow
learners or wall-avoidance floors.** The three 200k ceiling cells reach 100% success on
all three rooms with tight CIs, prox-noise doing so at a collision rate of 0.00 (the noise
is absorbed, not lethal), and even the hardest *learnable* flicker (`p=0.7`) reaches
the ceiling given budget: 9/10 seeds at full success, the tenth (seed 9) plateauing
partway with bathroom SR 0.33, which the IQM trims but which exists. The ~1.6 gap
below the 27.8 optimal-path ceiling (R(11)=27.8 is the optimal return for the two near
rooms, which the trimmed IQM tracks; the full-16-D symmetric configuration itself sits
just under it at IQM 27.66) is a path-length tax from occasional blackouts, not
sub-ceiling difficulty.

**Second, the one cell that leaves the ceiling does not become hard-but-learnable. It
fragments, and the two budgets together are the finding.** At the 200k base budget, `p=0.8`
looked like a clean systematic fracture: IQM collapsed to 8.95 [6.43, 13.01], kitchen was
abandoned in 10/10 seeds and bedroom solved in 10/10 (Figure 7), failures were timeouts
rather than collisions, and only three distinct per-room outcome patterns appeared across
the ten seeds. To remove the budget-parity asymmetry with the `p=0.7` cell, we re-ran
`p=0.8` at 10 seeds × 500k. [P2-MS] The aggregate barely moves: IQM 13.24 [7.79, 18.91], an
interval overlapping the 200k result substantially, so mean return is not statistically
distinguishable between the two budgets. But the *distribution* changes completely. Six
distinct kitchen/bedroom/bathroom patterns now appear across ten seeds, up from three. The
kitchen, abandoned by every seed at 200k, is solved by five of ten at 500k; conversely two
seeds (2 and 6) now abandon the *bedroom*, the one room every seed solved at 200k. Only
two seeds (4 and 9) solve all three rooms, at per-seed IQM 25.56 and 25.13; the other
eight specialise on different room subsets (per-seed detail in Appendix A).

We report both budgets in Table 2 rather than replacing the 200k result, because their
combination is what carries the argument. The 200k data alone would have supported the
description "systematic near/far give-up", a description the 500k data does not survive.
What the two together support is a stronger claim: `p=0.8` flicker places the task in a
regime of **landscape fragmentation**, in which multiple locally-optimal *reactive*
policies partition the reachable outcome space by room, and seed selection determines
which one a given run finds. The 200k pattern was one manifestation of that landscape,
the one visible while training is short enough that most seeds have not left their initial
basin. This is a stronger negative signal for the paper's central concern than the
original fracture story, not a weaker one: under a manipulation that looked like clean
structure at one budget, matched-budget multi-seed evaluation exposed seed-driven
behavioural diversity that a single budget, let alone a single run, would have missed.
Either way the rung is disqualified for the architecture comparison, and for the same
reason: a gap measured on it would reflect which architecture more often lands in which
basin, not whether `A` suits `S`. Read down the ladder, this is a graded trend rather than
an isolated result. The number of distinct per-seed behavioural patterns rises
monotonically with degradation severity: 1/10 at each of the three ceiling rungs, 2/10 at
`p=0.7`, 3/10 at `p=0.8` on the base budget, 6/10 at `p=0.8` on matched budget (Table 2).
The fragmentation of the hardest rung is thus the endpoint of a trend the whole ladder
exhibits, not a property peculiar to one cell.

**Third, this exhausts the axis.** Across removal, flicker, aliasing, and sensor noise, the
pre-registered set of single-map observation degradations (the pre-registration caps the
ladder at these four mechanisms on the fixed 20×20 map; any further degradation is deemed a
different task under a separate Phase-3 pre-registration), no rung meets the precondition.
On this task, the H1 asymmetry hypothesis is **untestable**: the Phase 1 null is a fact
about the task, not (on this evidence) about the hypothesis. This has a direct consequence
for the pre-registered primary endpoint. That endpoint, the inverted−symmetric IQM gap, is
defined only on a rung that passes the step-5 `S`-attribution guard, and no rung did: every
cell either ceilinged, fractured, or fragmented. The gap was therefore never computed on a
qualifying cell, and Table 2 is symmetric-only for exactly that reason. The non-computation
*is* the result. There was no cell on which measuring the architecture gap would have meant
anything.

**Figure 7.** The p=0.8 fracture **at the 200k budget**, per seed. Left: outcome split
(success / timeout / collision), showing failures are timeouts not collisions. Right:
per-room success-rate heatmap across the 10 seeds, kitchen abandoned 10/10 and bedroom
solved 10/10. The corresponding 500k per-seed outcomes are in Appendix A.
(`figures/p2_flicker08_perseed.png`)

---

## 6. The protocol: testing whether a task can test an architecture hypothesis

The navigation study is one instance of a problem that recurs whenever architectures
are compared empirically. An architecture hypothesis almost always has the form
*"architectural property `A` improves performance because it suits some structure `S`
of the problem"*: inverted actor–critic asymmetry suits a hard-to-represent policy;
convolution suits spatial locality; attention suits long-range dependency; extra depth
suits compositional structure. Such a hypothesis can only be tested on a task that
actually *contains* `S`. If the chosen benchmark lacks `S`, then `A` has nothing to act
on and the hypothesis predicts **no** effect *by construction*, so a null result is
uninformative and, just as dangerously, a positive result on one seed is unfalsifiable
noise. Before claiming a task confirms or refutes such a hypothesis, one should check
that the task is *able* to test it. We distil our study into a reusable procedure for
that check, stated below in task- and hypothesis-agnostic terms; the navigation study
above is the worked example that instantiates it.

> **Protocol — can this task test this architecture hypothesis?**
> *Given a hypothesis "architectural property `A` helps because it suits problem
> structure `S`":*
>
> 1. **State the precondition.** Write down `S` explicitly, the property the task must
>    have for `A` to matter, and the observable signature of `S` being present versus
>    absent. If you cannot name `S`, you cannot test the hypothesis.
> 2. **Design a manipulation ladder.** Construct a graded family of task variants that
>    monotonically vary the presence or degree of `S`, holding everything else fixed
>    (same reward, dynamics, budget, optimiser). Each rung moves the task toward
>    satisfying the precondition.
> 3. **Gate each rung for learnability — single seed, cheap.** Confirm a *standard*
>    (baseline) agent can still learn each rung before spending statistical compute.
>    Discard rungs where the baseline floors: an unlearnable rung looks "hard" but tests
>    nothing (a floor effect, not the target regime).
> 4. **Promote load-bearing rungs to multi-seed — claim-grade.** For the rungs a
>    conclusion rests on, run enough seeds for stratified-bootstrap confidence intervals
>    (report IQM + CI, not single-seed point estimates). Stage compute if needed:
>    decision-grade first, a handful of seeds, enough to make the go/no-go call; then
>    archival-grade, the full pre-registered seed count with confidence intervals,
>    before publication.
> 5. **Read the ladder — and attribute the hardness.** If some rung is
>    **hard-but-learnable**, baseline sub-ceiling but improving, do not stop at "it's
>    hard." Verify the hardness is attributable to `S` *specifically*, and not to a
>    confounding difficulty source (an optimisation pathology such as a give-up local
>    optimum, or raw sensory deprivation that starves the agent rather than complicating
>    the policy). A rung can be hard the *wrong* way: if it is, an architecture gap
>    measured there reflects "which architecture escapes the pathology more often," not
>    "`A` suits `S`," and the comparison silently tests the wrong thing. Only once the
>    hardness is `S`-attributable does the rung qualify; then run the architecture
>    comparison **on that rung**. If **no** rung reaches the precondition in a learnable,
>    `S`-attributable regime, the task *cannot* test the hypothesis. Report that, with the
>    mechanism that explains why the manipulation never induced `S`.
>
>    *(Our own p=0.8 flicker rung is the cautionary example: it is hard-but-not-ceiling,
>    yet it became hard the wrong way, a systematic give-up local optimum at the base
>    budget and, at matched budget, a fragmented landscape of seed-selected reactive
>    optima rather than a hard-to-represent policy. An architecture comparison there
>    would have measured which basin a run happens to land in, not `S`. It is disqualified
>    by this guard, not by its difficulty.)*

Two design choices in this procedure are themselves part of the methodological
contribution, not incidental lab habit. **Pre-registration** of the ladder, the
hypotheses, and, critically, a per-outcome *interpretation table* fixed before any
run, prevents the search from silently degenerating into architecture-fishing: with the
readings committed in advance, "we tried variants until one favoured our architecture"
is structurally impossible, and every escalation is a dated amendment with a frozen
meaning. **The gate/multi-seed split** is what makes exhaustive laddering affordable: a
single-seed learnability gate is one training run, so many candidate rungs can be
screened for the price of one claim-grade cell, and expensive seeds are spent only where
a conclusion actually rests. Together they turn "our architecture idea didn't win" into
a falsifiable, cheap-to-run diagnostic about the benchmark.

That the five steps never mention navigation, observations, or reinforcement learning is a
necessary condition for reusability, not a demonstration of it; domain-agnostic phrasing is
cheap. The demonstration is the positive control below, where the same five steps,
instantiated on image classification with `A` = convolution and `S` = spatial locality,
return a positive verdict where the navigation instantiation returned a negative one. A
reader studying width-versus-depth on a regression benchmark, or
attention-versus-convolution on a sequence task, instantiates `S` with their own structure,
input frequency content or dependency range, and follows the identical procedure. Our study
fixes `A` = inverted actor–critic asymmetry, `S` = a policy that is hard to represent while
the value stays smooth, the ladder = graded observation degradation, and finds no rung
satisfies the precondition: the concrete shape of one negative outcome the protocol can
return.

### 6.1 A positive control: the protocol returns *yes* where the precondition is present

A procedure that has so far returned only its negative branch invites a fair objection: is
it a detector, or merely a rejection stamp? To show it also returns **yes**, we run it as a
**positive control** on a task where the precondition can be inserted and removed at will,
recovering a *known-true* effect rather than asserting a new one: `A` = convolution,
`S` = spatial locality. The ladder is a single fixed permutation of the MNIST pixels applied
to a fraction `q ∈ {0, 0.25, 0.5, 0.75, 1.0}` of positions (nested subsets, a derangement
within each, identity elsewhere). Every rung is a **bijection** on pixel positions, a fixed
relabeling of the input units, so **information is provably preserved** (the classes remain
perfectly separable); what is progressively destroyed as `q → 1` is only spatial locality. We
compare a small CNN against a parameter-matched MLP (105.9k parameters each), 10 seeds,
everything else held fixed, at a sub-asymptotic budget where the convolutional advantage is
widest. Design, predictions, and interpretation table were pre-registered before any run.

| `q` | CNN test acc. (IQM, 95% CI) | MLP test acc. (IQM, 95% CI) | gap |
|---|---|---|---|
| 0.00 | 97.98 [97.81, 98.14] | 95.83 [95.72, 95.89] | **+2.15** |
| 0.25 | 97.31 [97.12, 97.45] | 95.66 [95.54, 95.82] | +1.65 |
| 0.50 | 96.27 [96.13, 96.47] | 95.69 [95.57, 95.86] | +0.59 |
| 0.75 | 93.59 [93.25, 93.89] | 95.77 [95.68, 95.85] | −2.18 |
| 1.00 | 93.18 [92.93, 93.45] | 95.70 [95.60, 95.82] | −2.52 |

**Table 3.** Positive-control dose-response: CNN vs parameter-matched MLP test accuracy
(IQM + 95% CI, 10 seeds) at each scramble fraction `q`.

**Figure 8.** The dose-response of Table 3 plotted: CNN and MLP test accuracy vs scramble
fraction `q`, with 95% CI bands and the shrinking gap; the CNN starts high, declines
monotonically, and crosses the flat MLP near `q≈0.55`. (`figures/pc_dose_response.png`)

Figure 8 shows the dose–response. The frozen predictions were: **P1**, the CNN degrades
monotonically as `q → 1`; **P2**, the MLP's achievable accuracy is identical, since the
learning problem is unchanged up to a relabeling of input units and the MLP hypothesis class
is closed under that relabeling; **P3**, the gap is large at `q=0` and closes to ≈0 at `q=1`;
**P4**, the protocol therefore reports the task *can* test `A`-suits-`S` at `q=0` and
*cannot* at `q=1`. P1 held (CNN IQM 97.98 → 93.18, monotone). P4 held: at `q=0` the CNN leads
by **+2.15 points with disjoint 95% CIs**, a resolvable effect, while at `q=1` there is no
positive gap to find.

**P3 was partially wrong, in an informative direction, and we report it as such.** The gap
does shrink monotonically, but at `q=1` it does not settle at zero. It crosses to **−2.52**,
the CNN landing *below* the MLP with disjoint CIs. We had frozen "≈0"; the data say the
convolutional prior is not merely uninformative on non-local inputs but mildly *harmful*,
because the CNN is constrained to seek local structure the scrambled task no longer contains,
while the position-agnostic MLP is unencumbered. This does not weaken the control; the effect
still appears and vanishes with `S`. But a prediction missed, and we let it stand rather than
round it to the number we pre-registered. The paper asks that discipline of the work it
examines, and owes it of itself.

The MLP curve is what licenses attributing the CNN's decline to `S` specifically. Because
each rung is a bijection, the MLP's **0.13-point** drift from `q=0` to `q=1` is optimization
noise, not information loss: the task did not get *harder*, only locality was removed, so the
CNN's fall can only be the loss of the structure convolution exploits. (Flatness is judged by
a pre-committed absolute threshold, `|Δ| < 1.0` point, which holds comfortably at 0.13 points.
The corroborating check asks whether the `q=1` MLP IQM (95.70) lies inside the `q=0` MLP's
95% CI of [95.72, 95.89]; it misses at the lower bound by 0.02 points. This is an artifact of
precision, not a real shift: with 10 seeds that interval is only ±0.1 point wide, so a change
far smaller than any practical significance still falls outside it. We pre-committed the
absolute threshold precisely because CI-overlap is not a sound flatness criterion at that
resolution, pre-registration working as intended.) This is the step-5 `S`-attribution guard
**succeeding**, the complement of the navigation `p=0.8` fracture, where the same guard
*disqualified* a rung that had become hard the wrong way.

None of this is about convolutions. What it establishes is that the protocol tracks the
effect **appearing and vanishing with `S`**: run one rung over, at `q=0`, and it recovers the
CNN advantage in full; run at `q=1`, and it correctly reports there is nothing to find. Where
`S` can be dialed in, the protocol says *yes*; on the navigation task, no manipulation could
dial `S` in at all, and it says *no*. **The instrument works; the benchmark doesn't.**

> A researcher who evaluated only the fully-scrambled task, who trained a CNN and an MLP on
> `q=1` MNIST and compared them, would not see the two merely tie. They would see the CNN
> lose by two and a half points and conclude that *convolution is actively harmful for image
> classification*. The conclusion is absurd, and it is absurd for exactly the reason the
> navigation null is: the task they measured has had its spatial locality permuted out of it,
> so it could never have shown the benefit of a locality prior, and a sound, multi-seed
> measurement of a meaningless comparison returns a confident wrong answer. That researcher
> stands in the identical structural position as one concluding "actor–critic asymmetry hurts
> in reinforcement learning" from the navigation result. The only difference is that here we
> can *see* the error, because the same protocol run one rung over recovers the effect in
> full, a luxury the navigation study, having no `q=0` to fall back on, does not afford.

This is a positive control for the protocol, not a contribution about convolutional
architectures: we claim nothing beyond the well-known fact that locality helps when it is
present, and use it only to show the instrument reads *yes* as readily as *no*.

## 7. Why the task resists every manipulation

Three properties of the environment, each visible in the results above, together explain why
no observation manipulation induced a hard-to-represent policy.

**Observation redundancy.** The 16-D observation carries several channels that each
independently suffice for competent navigation, so removing any one of them leaves the others
to carry the load. The cumulative case is the clearest: six of the sixteen dimensions can be
removed, global position (dims 9–10), distance-to-target (dim 11), and the region one-hot
(dims 13–15), and the task is still solved completely, with all three target rooms reached
in 100% of episodes (IQM 27.17 [27.04, 27.37] at 10-D, against 27.66 [27.56, 27.78] for the
full 16-D observation). The half-point difference is statistically resolvable but is a
path-length tax rather than a loss of capability: the agent still reaches every target,
taking marginally longer to do so. Nor is any single channel necessary in isolation. Removing
global position alone leaves a ceiling; additionally removing distance-to-target leaves a
ceiling (27.24 [27.21, 27.27]); additionally removing the region one-hot leaves a ceiling, at
no detectable cost in sample efficiency either (36.7k versus 40.0k environment steps to 90%
of asymptote, a difference within seed noise). The proximity sensors appear to be the
operative channel, since they alone distinguish room, hallway, and doorway cells, which is
what the region one-hot nominally encoded. Yet even proximity's reliability is dispensable:
corrupting it with 30% per-bit noise still ceilings (26.78 [26.41, 26.93]) at a collision
rate of 0.00. The observation is over-determined for this task.

**Reactive optimality.** The optimal policy on this task appears to be well approximated by a
function of the *current* observation rather than of the observation history. This is why
frame-stacking neutralises flickering so effectively: with a stack of four frames, the agent
is deprived of every recent frame only when four consecutive maskings occur, an event of
probability `p^k`, 6.25% at `p=0.5` and 24% at `p=0.7` (§5). Because a single recent frame
suffices to act well, the agent does not need to integrate across gaps; it needs only *some*
frame to be recent. Flickering therefore attacks the *availability* of the observation, not
the *complexity* of the function mapping observations to actions, and availability is not the
quantity the policy-hard hypothesis concerns. The distinction becomes sharpest at `p=0.8`
(41% fully-masked), where the agent's behaviour no longer cleanly converges. At 200k, returns
fall to IQM 8.95 [6.43, 13.01] with a single dominant failure mode: kitchen abandonment across
every seed, failures dominated by timeouts (a per-seed timeout fraction of 0.13–0.47 of
episodes) rather than collisions (≈0). At matched budget (500k, §5), the aggregate return is
not meaningfully higher (IQM 13.24 [7.79, 18.91]) but the per-seed structure fragments into
six distinct room-specialisation patterns across ten seeds, with two seeds finding a
near-ceiling policy. Neither budget shows the graceful degradation of a policy becoming
harder to represent; both show an optimisation landscape with multiple reactive local optima
whose selection is seed-driven. This is what the protocol's step-5 `S`-attribution guard
exists to disqualify: hardness arising from landscape structure over reactive policies, not
from any policy being genuinely hard to represent.

**A single memorisable layout.** The map is fixed across all episodes, with optimal path
lengths of 11, 11, and 22 steps to the three targets (§5). A policy can therefore encode
routes specific to this layout rather than computing them from the observation, which would
explain the robustness to proximity corruption, since a route-following policy needs little
reliable wall-sensing. We note this as an interpretation consistent with the data rather than
a directly measured property; we did not attempt to separate route memorisation from
observation-driven navigation, and doing so would require held-out layouts.

**Why this closes off the observation axis.** The three properties compound into a single
conclusion. The precondition the hypothesis requires, a policy that is hard to represent
while the value function remains smooth, is a property of the task's *decision structure*,
not of its observation vector. This task's decision structure is short-horizon, static, and
reactively solvable: the optimal action is a simple function of local information, and it
remains so however that information is delivered. Degrading the observation can make the
information *scarcer* (flickering), *narrower* (removal), or *noisier* (corruption), but on
this task none of these made the underlying decision function *more complex*. Its complexity
is set by the task's short-horizon, static, reactively-solvable structure, not by how the
observation is delivered. Hence the pattern across §5: manipulations either
leave the simple policy intact and reachable (ceiling), or starve the agent badly enough that
it stops attempting the task (fracture) or scatters across seed-selected reactive optima
(fragmentation). There is no intermediate regime in which the policy becomes hard but
learnable, because there is no hard policy to find.

## 8. What a task would need to test this hypothesis

The null is generative: the same analysis that shows this task cannot test the hypothesis
specifies what a task that *could* would look like. We state it as a property list a
future benchmark can be built against, the natural Phase 3 target, rather than as a
vague call for "harder tasks."

- **Genuine, unroutable partial observability.** Hidden state that (i) cannot be
  recovered from any single observation *and* (ii) cannot be memorised away, which means
  procedurally generated or changing layouts, so no fixed-map shortcut exists and the
  optimal policy must integrate history on every episode. (This directly negates the
  "reactive optimality" and "memorisable map" escapes identified above.) Procedurally
  generated benchmark suites already supply this property in a controlled form
  [Cobbe et al., 2020; Chevalier-Boisvert et al., 2023].
- **Non-redundant observation.** Each policy-relevant distinction is encoded once, so a
  targeted removal actually removes it. (Negates "observation redundancy".)
- **A longer horizon with real credit assignment.** Enough temporal depth that the
  optimal policy must compose sub-decisions, so representing it is a compositional
  problem rather than a one-step lookup.
- **A policy that is a sharp function even under full information.** Ideally the hardness
  is intrinsic to the optimal *mapping* (high-frequency, many decision boundaries), not
  merely induced by hiding information, so that policy capacity is taxed while value
  can remain comparatively smooth, which is exactly the asymmetry the hypothesis is
  about.

A task meeting this specification would place the precondition inside a learnable regime,
and the architecture comparison of §4 could be run where it is meaningful. Whether the
inverted-asymmetry advantage then appears is an open empirical question this paper does
not answer and does not prejudge.

## 9. Discussion

The transferable lesson is a discipline, not a result: **before claiming a task confirms
or refutes an architecture hypothesis, test whether the task is able to falsify it.** A
cheap, pre-registered manipulation ladder with single-seed learnability gates makes this
check affordable, and its outcome, which rung (if any) is hard-but-learnable, tells you
where or whether to spend a claim-grade comparison. Pre-registration is doing real work
here: it converts an open-ended "try architectures until one wins" into a bounded,
falsifiable procedure whose negative outcomes are as informative as its positive ones. In the
terms of §3, it operationalises the construct-validity question — does this benchmark
measure what we take it to? — as a prospective, pass/fail experiment run *before* a
comparison rather than a retrospective critique delivered after the fact.

The scope of our negative result is narrow by design. We have shown that **this task,
and by the mechanism of §7 static single-map gridworld navigation with a redundant,
low-dimensional observation, cannot test the inverted-asymmetry hypothesis**, because no
degradation of its observation places the precondition in a learnable regime. We have
**not** shown that inverted actor–critic asymmetry fails to help in reinforcement
learning generally; the hypothesis may well hold on a task of the kind specified above, and
nothing here bears on that. The contribution is precisely scoped on purpose: a rigorous
demonstration that a specific, widely-used *style* of task cannot test a specific
architectural hypothesis, together with a reusable protocol for detecting this failure
mode before it is mistaken for evidence either way. A single-seed positive result on such
a task, our own starting point here, is exactly what the protocol is designed to catch.

A protocol that only ever returned *no* would be a rejection stamp, not an instrument;
§6.1 answers that objection directly. On a vision task where the precondition can be
inserted and removed at will, the same procedure returns a clean *yes* where spatial
locality is present and *no* where it is not, tracking the effect as it appears and
vanishes: the positive branch, demonstrated. The remaining limitation is specific and we
state it plainly. That positive instantiation is on image classification, not on
reinforcement learning. We have exhibited the protocol's positive branch (on vision) and
its negative branch (on RL navigation), but **not a substantive positive instantiation on
an RL task**, one of the kind specified above, where the asymmetry precondition is present in
a learnable regime and the architecture comparison can actually run. Constructing that task
and testing inverted asymmetry where it *can* be tested is the natural next step; the
specification above is written to make it buildable.

---

## Appendix A. Per-seed outcomes, `p=0.8` flicker at matched budget (500k)

The fragmentation reported in §5 is a statement about the *distribution* across seeds, so
we give the per-seed data it rests on. [P2-MS] Ten seeds, symmetric PPO, 500k steps.
Per-room SR is each seed's own success fraction on episodes targeting that room; the
aggregate row is the seed-mean of each column, per the §4 definition. Six distinct
(kitchen, bedroom, bathroom) combinations occur across the ten seeds.

| Seed | kitchen SR | bedroom SR | bathroom SR | eval return IQM | regime |
|---|---|---|---|---|---|
| 0 | 0.00 | 1.00 | 0.33 | 15.70 | bedroom + partial bathroom |
| 1 | 0.00 | 1.00 | 0.67 | 17.96 | bedroom + partial bathroom |
| 2 | 1.00 | 0.00 | 1.00 | 0.22 | kitchen + bathroom, bedroom abandoned |
| 3 | 0.00 | 1.00 | 1.00 | 20.64 | bedroom + bathroom, kitchen abandoned |
| 4 | 1.00 | 1.00 | 1.00 | 25.56 | **all three rooms** |
| 5 | 0.00 | 1.00 | 1.00 | 21.73 | bedroom + bathroom, kitchen abandoned |
| 6 | 1.00 | 0.00 | 1.00 | 0.24 | kitchen + bathroom, bedroom abandoned |
| 7 | 1.00 | 0.50 | 0.67 | 18.51 | kitchen + partial bedroom/bathroom |
| 8 | 0.00 | 1.00 | 0.67 | 17.76 | bedroom + partial bathroom |
| 9 | 1.00 | 1.00 | 1.00 | 25.13 | **all three rooms** |
| **aggregate (seed-mean)** | **0.50** | **0.75** | **0.83** | **13.24** [7.79, 18.91] | fragmentation |

Two features stand out against the 200k result of Table 2. First, the kitchen,
abandoned by every seed at 200k, is solved by five of ten seeds here, while the bedroom,
solved by every seed at 200k, is abandoned outright by seeds 2 and 6; no room is
categorically out of reach, and no room is reliably reached. Second, seeds 2 and 6 carry
very low returns (IQM 0.22 and 0.24) despite solving two of three rooms, because the
sampled eval draw gave them a majority of episodes on the room they abandoned. This
illustrates why per-room SR and aggregate return are reported separately rather than
one being inferred from the other.
