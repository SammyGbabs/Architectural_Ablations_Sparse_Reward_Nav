# Testing whether a task can test an architecture hypothesis

*(Working title — draft prose. Not anonymized; not yet ported to the TMLR LaTeX template.)*

**Notation left visible for the writing pass.** Bracketed provenance tags `[P1-MS]` /
`[P2-MS]` / `[GATE]` mark each number's evidence source (multi-seed Phase 1 / multi-seed
Phase 2 / single-seed gate — the last never cited as a result). Bracketed author names
(`[Henderson et al.]`, `[Agarwal et al.]`, `[Hausknecht & Stone]`, `[POPGym]`,
`[Mnih et al.]`) are citation placeholders to be filled. No `⟦…⟧` data placeholders remain.
Figures are referenced by number; the image files live in `figures/` (filename given in
each caption).

---

## 1. Abstract

Empirical architecture comparisons carry an implicit assumption: that the benchmark can
actually *test* the hypothesis — that the task contains the structure the architecture
is meant to exploit. When it does not, a null is uninformative and a single-seed positive
is unfalsifiable noise, yet both are routinely reported as evidence. We give a
**pre-registered protocol for deciding whether a task can test an architecture hypothesis
at all**, before trusting any comparison run on it: state the structural precondition the
hypothesis requires; build a graded manipulation ladder that targets it; gate each rung
for learnability with a single seed; promote only load-bearing rungs to multi-seed,
claim-grade evaluation; and attribute any induced hardness to the precondition rather than
to a confound before comparing architectures. We instantiate the protocol on a claim from
our own earlier, unpublished work — that an *inverted* actor–critic asymmetry (a deeper actor
than critic) improves sparse-reward indoor navigation. Under multi-seed evaluation the
single-seed advantage disappears (IQM 27.56 vs 27.66, overlapping 95% CIs,
P(inverted > symmetric) = 0.25), and the 12.5× PPO sample-efficiency advantage we had earlier
reported does not survive a defined, step-based, multi-seed measurement: in environment-steps to
90% of asymptotic return the two algorithms' distributions overlap entirely — no PPO
configuration is faster than the DQN family, and the one outlier is a *slow* PPO
configuration. A pre-registered observability ladder
— four degradation mechanisms, each a dated amendment — then shows *why* the asymmetry
question cannot be settled here: no manipulation places the task in the regime the
hypothesis needs. Every rung either **ceilings** (the optimal policy stays easy to
represent) or **fractures** into a give-up local optimum (hard, but not in the way the
hypothesis specifies). We report this as a worked *negative outcome of the protocol*, not
as evidence against asymmetry: this task class cannot arbitrate the hypothesis, which may
still hold on a task meeting the specification we derive. The contribution is the reusable
check — and the finding that a widely-used style of benchmark silently fails it.

## 2. Introduction

The reproducibility literature has taught deep reinforcement learning to distrust a single
run. Seed variance can flip a conclusion, and much reported progress has not survived
multi-seed re-evaluation [Henderson et al.; Agarwal et al.]; the accepted remedy is to
report interquartile means with stratified-bootstrap confidence intervals rather than point
estimates. This closes one failure mode — mistaking noise for signal — but leaves a prior
one untouched. A comparison can be statistically impeccable and still meaningless: if the
benchmark does not contain the structure the architecture is meant to exploit, multi-seed
statistics measure, with great precision, an effect the task was never capable of showing.
And the two cases are observationally identical from inside the experiment — a null on a
task where the effect is genuinely absent and a null on a task that could never have
expressed it produce the same numbers. rliable can tell you a comparison is statistically
sound; it cannot tell you it is meaningful. Pre-registration can tell you that you did not
fish for the result; it cannot tell you the pond contains fish. That gap — between a sound
measurement and a meaningful one — is the subject of this paper.

The gap has a common structure. An architecture hypothesis almost always asserts that
*architectural property `A` helps because it suits some structure `S` of the problem* —
inverted actor–critic asymmetry suits a policy that is hard to represent, convolution suits
spatial locality, attention suits long-range dependency. Such a hypothesis is only testable
on a task that actually contains `S`. Where `S` is absent, `A` has nothing to act on: the
hypothesis predicts no effect *by construction*, so a null is uninformative and any
single-seed win is noise dressed as evidence. Nothing in a standard train-and-evaluate loop
flags this — the task quietly fails to pose the question, and the experiment quietly answers
a different one.

We arrived at this through a concrete claim of our own. In earlier, unpublished experiments
of ours, an inverted actor–critic asymmetry — a policy network deeper than its value network
(`π=[512,256,128]` vs `v=[256,128]`) — appeared to outperform a symmetric baseline at matched
budget on a sparse-reward indoor-navigation task, on a single seed, together with an apparent
≈12.5× sample-efficiency advantage of PPO over DQN. Both claims come apart on re-examination — but
*how* they come apart is the more useful observation, and is what pointed us at the gap
above: one claim asked a question the task could not answer, the other a question the
measurement could not answer.

This paper makes two contributions. **First, a reusable protocol** (§6) for deciding whether
a task can test a given architecture hypothesis, before trusting any comparison run on it:
name the precondition `S`, build a manipulation ladder that targets it, gate rungs for
learnability cheaply, spend seeds only on load-bearing rungs, and — before comparing
architectures — confirm any induced hardness is attributable to `S` rather than a confound.
Pre-registration and the cheap-gate/expensive-seed split are integral to it, not incidental.
**Second, a worked negative outcome**: applied to the navigation task, the protocol shows
the reported asymmetry advantage does not survive multiple seeds (§4) and that *no*
pre-registered degradation of the task (§5) induces the regime the hypothesis needs — the
task stays reactively, redundantly, memorably easy, or fractures the wrong way (§7).
Throughout we are careful that this is a statement about *the task*, not the hypothesis: we
show this task class cannot test inverted asymmetry, and specify (§8) what a task that could
would need — not that asymmetry fails in reinforcement learning.

## 3. Related work

**Reproducibility and evaluation in deep RL.** Henderson et al.'s *Deep Reinforcement
Learning that Matters* documented how seed variance, implementation details, and
under-powered comparisons produce non-reproducible conclusions; Agarwal et al. (*rliable*)
provide the statistical remedy we adopt — interquartile mean with stratified-bootstrap
confidence intervals and performance profiles, over many seeds. Our work builds on this but
targets a complementary failure mode: a comparison can be statistically impeccable and still
meaningless if the task cannot express the hypothesised effect. Good statistics answer
"is the measured difference real?"; our protocol answers the prior question, "*could* this
task have shown the difference at all?".

**Difficulty of partially-observed tasks.** The manipulation ladder is grounded in the
POMDP-difficulty literature. Flickering observations (Hausknecht & Stone) and benchmark
suites such as POPGym establish that *static* low-dimensional feature removal is often not
enough to make a task hard for modern deep RL, and that difficulty tends to require
temporal hidden state that must be integrated over time; frame-stacking (Mnih et al.) is the
standard memoryless response. We use exactly these levers — removal, flicker + frame-stack,
aliasing, sensor noise — and find, consistent with that literature, that they fail to make
*this* task's policy hard, for reasons we trace mechanistically.

**Architectural asymmetry and actor–critic capacity.** The hypothesis under test sits in a
line of work on how to allocate capacity between policy and value networks and on
architectural inductive biases more broadly. Our aim is not to adjudicate this hypothesis in
general — we take no position on whether inverted asymmetry helps on suitable tasks — but to
show that a specific, common style of task cannot arbitrate it, and to give the tools to
recognise such tasks.

**Pre-registration in machine learning.** Pre-registration — fixing hypotheses, design, and
analysis before seeing results — is standard in the experimental sciences but rare in ML,
where iterative tuning against a test signal is the norm. We treat it as a first-class part
of the methodological contribution rather than a compliance step: freezing the ladder and a
per-outcome interpretation table before any run is precisely what prevents an exhaustive
manipulation search from degenerating into architecture-fishing, and makes a negative
outcome as credible as a positive one. To our knowledge this discipline is under-used for
architecture-comparison studies specifically, which is where single-seed claims are most
tempting and least checked.

---

## 4. Phase 1 — the reported effects do not survive, and their absence is not yet an answer

We begin by holding the two claims from our earlier unpublished experiments to the evaluation
standard the reproducibility
literature now recommends — interquartile-mean (IQM) aggregation with 95 %
stratified-bootstrap confidence intervals over many seeds [Agarwal et al., *rliable*] —
across ten seeds each. Both dissolve. The purpose of this section, though, is not the
dissolution itself but what it does and does not license: a dissolved effect on this task
turns out to say nothing about the hypotheses behind it, and that emptiness is the first
concrete symptom of the gap §5 then diagnoses.

**Setup.** The environment is a 20×20 residential grid (`Discrete(5)` actions, a
16-dimensional observation, reward `R(L)=30−0.2L` for an `L`-step success). We train
the two architectures — symmetric and inverted — with otherwise identical PPO
hyperparameters (verbatim from those earlier experiments' configurations) for a fixed 200k-step
budget, at **10 seeds each**, and aggregate with rliable. [P1-MS]

**Result 1 — the asymmetry advantage does not survive multiple seeds.** Under
10-seed evaluation the two architectures are statistically indistinguishable: IQM
eval return **27.56** (inverted) versus **27.66** (symmetric), with overlapping 95 %
confidence intervals and a probability of improvement **P(inverted > symmetric) =
0.25** — i.e. the inverted configuration is, if anything, *less* likely to beat the
symmetric one on a random seed. [P1-MS] The single-seed advantage was seed noise;
the pre-registered falsification criterion for the asymmetry hypothesis (H1) is met.

**Result 2 — the sample-efficiency claim was the wrong kind of measurement.** Those same
earlier experiments also reported the policy-gradient agent (PPO) to be ≈12.5× more
sample-efficient than the value-based agent (DQN), on the basis that PPO reached stable high
performance in roughly 20 episodes against nearly 250 for DQN — that is, `12.5× = 250 / 20`,
a ratio of **episodes-to-visual-convergence** read off single-run learning curves. This quantity is not a sound cross-algorithm sample-efficiency
measure, for three reasons. **(a) Episodes are not a common currency of experience.**
Episode length on this task varies five- to ten-fold with policy quality — a wandering
early-training agent runs to the 150-step timeout while a converged agent finishes in
~14 steps — so "one episode" purchases very different amounts of environment interaction
at different points in training. For two algorithms whose episode-length trajectories
differ, an episode-count ratio therefore conflates sample efficiency with episode
duration; the currency the agent actually spends, and that the two algorithms share, is
*environment steps*. **(b) "Stable high performance" is eyeballed.** A convergence point
identified by eye has no defined threshold, and, read off a single curve, carries no
uncertainty. **(c) It is a single run** — and §4's own H1 data shows how misleading single
runs are on this task, where one PPO seed collapses to a return of 3.6 while its nine
siblings sit at ~27.6.

Re-measured with a defined, step-based, multi-seed statistic — environment steps to reach
90% of the asymptotic eval-return IQM, over 10 seeds, aggregated with rliable [Agarwal et
al.] — no advantage remains; and to keep the comparison from turning on a chosen exemplar we
report the full within-family distribution rather than a single pair. The four PPO
configurations reach the threshold at IQMs of 36.7k (Exp 1), 36.7k (Exp 2), 163.3k (Exp 3),
and 32.0k (Exp 4) steps; the five DQN configurations at 21.7k, 20.0k, 20.0k, 45.0k, and
26.7k — with overlapping confidence intervals throughout (e.g. PPO Exp 1 [26.7k, 40.0k], DQN
Exp 2 [14.0k, 74.0k]). [P1-MS] The two families occupy the same ≈20–45k band; the single
configuration that leaves it, PPO Exp 3 at 163.3k steps, is the *slowest* of either family —
so on an exemplar-independent reading PPO is no faster than DQN, and the one salient
difference runs opposite to the claimed 12.5× advantage.

These two results are one finding stated twice. The asymmetry claim put a question to a
*task* that could not answer it — a benchmark lacking the structure the hypothesis is about.
The sample-efficiency claim put a question to a *measurement* that could not answer it — a
metric that cannot separate sample efficiency from episode duration, nor one run's eyeballed
convergence point from noise. In both, the instrument was incapable of resolving the
question asked of it, and an incapable instrument returns a confident-looking number either
way. That is why neither number is yet an answer, and why the rest of the paper is concerned
less with re-scoring individual claims than with the prior question of whether the task and
the measurement can support the claim at all.

**Figure 1.** Per-configuration IQM eval return with 95% stratified-bootstrap CIs across the
nine main configs (PPO Exp 1–4, DQN Exp 1–5), 10 seeds each. (`figures/p1_iqm_main_configs.png`)

**Figure 2.** H1 performance profile — fraction of runs exceeding a return threshold, inverted
(Exp 4) vs symmetric (Exp 1); the two curves overlap across the range.
(`figures/p1_perf_profile_ppo_inverted_vs_symmetric.png`)

**Figure 3.** Sample efficiency — environment-steps to 90% of asymptotic IQM (IQM + 95% CI),
PPO vs DQN; the two families' distributions overlap, the one outlier being a *slow* PPO
config. (`figures/p1_sample_efficiency_ppo_vs_dqn.png`)

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
(Figures 4–5).

**Figure 4.** P(disabling blackout ≥1 per episode) versus flicker probability `p` at `k=4`,
one curve per room using its real BFS path length `H`, with the "hard-but-learnable" band
shaded. (`figures/p2_rung3_blackout_vs_p.png`)

**Figure 5.** Model-predicted (distance-`H`-based) per-room difficulty vs observed per-room
difficulty at `p=0.5/0.7/0.8`; the observed ordering does not follow `H` — difficulty is
geometry-, not distance-, driven. (`figures/p2_rung3_observed_vs_predicted.png`)

**Claim-grade results.** [P2-MS] For a *standard* (symmetric) PPO agent — Exp 1
hyperparameters verbatim, 10 seeds, rliable IQM + 95% CI — every cell either ceilings
or fractures; none is hard-but-learnable (Table 1; Figure 6):

**Table 1.** Phase 2 claim-grade ladder cells — symmetric PPO, 10 seeds, rliable IQM + 95% CI.

| Cell | obs | steps | IQM eval return (95% CI) | per-room SR (k/bed/bath) | outcome |
|---|---|---|---|---|---|
| A-STRICT | 13-D | 200k | 27.24 [27.21, 27.27] | 1.00/1.00/1.00 | ceiling |
| Aliasing | 10-D | 200k | 27.17 [27.04, 27.37] | 1.00/1.00/1.00 | ceiling |
| Prox-noise q=0.3 | 13-D | 200k | 26.78 [26.41, 26.93] | 1.00/1.00/1.00 | ceiling |
| Flicker p=0.7 | 52-D | 500k | 26.14 [25.42, 26.32] | 1.00/1.00/1.00 | ceiling (slow) |
| Flicker p=0.8 | 52-D | 200k | 8.95 [6.29, 13.01] | 0.00/1.00/0.33 | **fracture** |

**Figure 6.** The ladder cells of Table 1 plotted: per-cell IQM eval return + 95% CI — four
ceilings clustered near the 27.8 reference vs the p=0.8 fracture isolated at 8.95.
(`figures/p2_ladder_iqm.png`)

Three observations are decisive. **First, the ceilings are real ceilings, not slow
learners or wall-avoidance floors:** the four ceiling cells reach 100% success on all
three rooms with tight CIs, prox-noise does so at a collision rate of 0.00 (the noise
is absorbed, not lethal), and even the hardest *learnable* flicker (`p=0.7`) reaches
the ceiling given budget — 9/10 seeds at full success (the tenth, seed 9, plateaus
partway with bathroom SR 0.33; the IQM trims it, but we note it exists), the ~1.6 gap
below 27.8 being a path-length tax from occasional blackouts, not sub-ceiling difficulty. **Second, the
one cell that leaves the ceiling does not become hard-but-learnable — it fractures:**
at `p=0.8`, IQM collapses to 8.95 and the failure is a *systematic give-up*, robust
across all 10 seeds — **kitchen is abandoned in 10/10 seeds and bedroom solved in
10/10** (Figure 7), with failures being timeouts, not collisions.
That is a degenerate local optimum, not the graceful partial competence a fair
architecture test needs. **Third, this exhausts the axis:** across removal, flicker,
aliasing, and sensor noise — the pre-registered cap on single-map observation
degradation — no rung meets the precondition. On this task, the H1 asymmetry
hypothesis is **untestable**: the null of §4 is a fact about the task, not (on this
evidence) about the hypothesis.

**Figure 7.** The p=0.8 fracture, per seed — left: outcome split (success / timeout /
collision), showing failures are timeouts not collisions; right: per-room success-rate
heatmap across the 10 seeds, kitchen abandoned 10/10 and bedroom solved 10/10.
(`figures/p2_flicker08_perseed.png`)

---

## 6. The protocol: testing whether a task can test an architecture hypothesis

The navigation study is one instance of a problem that recurs whenever architectures
are compared empirically. An architecture hypothesis almost always has the form
*"architectural property `A` improves performance because it suits some structure `S`
of the problem"* — inverted actor–critic asymmetry suits a hard-to-represent policy;
convolution suits spatial locality; attention suits long-range dependency; extra depth
suits compositional structure. Such a hypothesis can only be tested on a task that
actually *contains* `S`. If the chosen benchmark lacks `S`, then `A` has nothing to act
on and the hypothesis predicts **no** effect *by construction* — so a null result is
uninformative, and, just as dangerously, a positive result on one seed is unfalsifiable
noise. Before claiming a task confirms or refutes such a hypothesis, one should check
that the task is *able* to test it. We distil our study into a reusable procedure for
that check. It is stated below in task- and hypothesis-agnostic terms; §§4–5 are the
worked example that instantiates it.

> **Protocol — can this task test this architecture hypothesis?**
> *Given a hypothesis "architectural property `A` helps because it suits problem
> structure `S`":*
>
> 1. **State the precondition.** Write down `S` explicitly — the property the task must
>    have for `A` to matter — and the observable signature of `S` being present versus
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
>    (report IQM + CI, not single-seed point estimates). Stage compute if needed
>    (decision-grade first, archival-grade before publication).
> 5. **Read the ladder — and attribute the hardness.** If some rung is
>    **hard-but-learnable** — baseline sub-ceiling but improving — do not stop at "it's
>    hard." Verify the hardness is attributable to `S` *specifically*, and not to a
>    confounding difficulty source (an optimisation pathology such as a give-up local
>    optimum, or raw sensory deprivation that starves the agent rather than complicating
>    the policy). A rung can be hard the *wrong* way: if it is, an architecture gap
>    measured there reflects "which architecture escapes the pathology more often," not
>    "`A` suits `S`," and the comparison silently tests the wrong thing. Only once the
>    hardness is `S`-attributable does the rung qualify — run the architecture comparison
>    **on that rung**. If **no** rung reaches the precondition in a learnable,
>    `S`-attributable regime, the task *cannot* test the hypothesis. Report that, with the
>    mechanism that explains why the manipulation never induced `S`.
>
>    *(Our own p=0.8 flicker rung is the cautionary example: it is hard-but-not-ceiling,
>    yet it became hard the wrong way — a systematic give-up local optimum, not a
>    hard-to-represent policy — so an architecture comparison there would have measured
>    local-optimum escape, not `S`. It is disqualified by this guard, not by its
>    difficulty.)*

Two design choices in this procedure are themselves part of the methodological
contribution, not incidental lab habit. **Pre-registration** of the ladder, the
hypotheses, and — critically — a per-outcome *interpretation table* fixed before any
run, prevents the search from silently degenerating into architecture-fishing: with the
readings committed in advance, "we tried variants until one favoured our architecture"
is structurally impossible, and every escalation is a dated amendment with a frozen
meaning. **The gate/multi-seed split** is what makes exhaustive laddering affordable: a
single-seed learnability gate is one training run, so many candidate rungs can be
screened for the price of one claim-grade cell, and expensive seeds are spent only where
a conclusion actually rests. Together they turn "our architecture idea didn't win" into
a falsifiable, cheap-to-run diagnostic about the benchmark.

The test of whether this protocol is genuinely reusable is that its five steps never
mention navigation, observations, or reinforcement learning: a reader studying
width-versus-depth on a regression benchmark, or attention-versus-convolution on a
sequence task, instantiates `S` with their own structure (input frequency content;
dependency range) and follows the identical procedure. Our study fixes `A` = inverted
actor–critic asymmetry, `S` = a policy that is hard to represent while the value stays
smooth, the ladder = graded observation degradation, and finds no rung satisfies the
precondition — the concrete shape of one negative outcome the protocol can return.

### 6.1 A positive control: the protocol returns *yes* where the precondition is present

A procedure that has so far returned only its negative branch invites a fair objection: is
it a detector, or merely a rejection stamp? To show it also returns **yes**, we run it as a
**positive control** on a task where the precondition can be inserted and removed at will,
recovering a *known-true* effect rather than asserting a new one: `A` = convolution,
`S` = spatial locality. The ladder is a single fixed permutation of the MNIST pixels applied
to a fraction `q ∈ {0, 0.25, 0.5, 0.75, 1.0}` of positions (nested subsets, a derangement
within each, identity elsewhere). Every rung is a **bijection** on pixel positions — a fixed
relabeling of the input units — so **information is provably preserved** (the classes remain
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

**Table 2.** Positive-control dose-response — CNN vs parameter-matched MLP test accuracy
(IQM + 95% CI, 10 seeds) at each scramble fraction `q`.

**Figure 8.** The dose-response of Table 2 plotted: CNN and MLP test accuracy vs scramble
fraction `q`, with 95% CI bands and the shrinking gap; the CNN starts high, declines
monotonically, and crosses the flat MLP near `q≈0.55`. (`figures/pc_dose_response.png`)

Figure 8 shows the dose–response. The frozen predictions were: **P1**, the CNN degrades
monotonically as `q → 1`; **P2**, the MLP stays flat (it is invariant to a fixed permutation
of its inputs); **P3**, the gap is large at `q=0` and closes to ≈0 at `q=1`; **P4**, the
protocol therefore reports the task *can* test `A`-suits-`S` at `q=0` and *cannot* at `q=1`.
P1 held (CNN IQM 97.98 → 93.18, monotone). P4 held: at `q=0` the CNN leads by **+2.15 points
with disjoint 95% CIs** — a resolvable effect — while at `q=1` there is no positive gap to
find.

**P3 was partially wrong, in an informative direction, and we report it as such.** The gap
does shrink monotonically, but at `q=1` it does not settle at zero — it crosses to **−2.52**,
the CNN landing *below* the MLP with disjoint CIs. We had frozen "≈0"; the data say the
convolutional prior is not merely uninformative on non-local inputs but mildly *harmful* —
the CNN is constrained to seek local structure the scrambled task no longer contains, while
the position-agnostic MLP is unencumbered. This does not weaken the control; the effect still
appears and vanishes with `S`. But a prediction missed, and we let it stand rather than round
it to the number we pre-registered — the paper asks that discipline of the work it examines,
and owes it of itself.

The MLP curve is what licenses attributing the CNN's decline to `S` specifically. Because
each rung is a bijection, the MLP's **0.13-point** drift from `q=0` to `q=1` is optimization
noise, not information loss: the task did not get *harder*, only locality was removed, so the
CNN's fall can only be the loss of the structure convolution exploits. (Flatness is judged by
a pre-committed absolute threshold, `|Δ| < 1.0` point, which holds comfortably at 0.13 points.
The corroborating check — whether the `q=1` MLP IQM (95.70) lies inside the `q=0` MLP's 95%
CI, [95.72, 95.89] — misses at the lower bound by 0.02 points. This is an artifact of
precision, not a real shift: with 10 seeds that interval is only ±0.1 point wide, so a change
far smaller than any practical significance still falls outside it. We pre-committed the
absolute threshold precisely because CI-overlap is not a sound flatness criterion at that
resolution — pre-registration working as intended.) This is the §6 step-5 `S`-attribution guard
**succeeding** — the complement of the navigation `p=0.8` fracture, where the same guard
*disqualified* a rung that had become hard the wrong way.

The point is not about convolutions. It is that the protocol tracks the effect **appearing
and vanishing with `S`**: run one rung over, at `q=0`, and it recovers the CNN advantage in
full; run at `q=1`, and it correctly reports there is nothing to find. Where `S` can be dialed
in, the protocol says *yes*; on the navigation task of §§4–5, no manipulation could dial `S`
in at all, and it says *no*. **The instrument works; the benchmark doesn't.**

> A researcher who evaluated only the fully-scrambled task — who trained a CNN and an MLP on
> `q=1` MNIST and compared them — would not see the two merely tie. They would see the CNN
> lose by two and a half points and conclude that *convolution is actively harmful for image
> classification*. The conclusion is absurd, and it is absurd for exactly the reason the
> navigation null is: the task they measured has had its spatial locality permuted out of it,
> so it could never have shown the benefit of a locality prior, and a sound, multi-seed
> measurement of a meaningless comparison returns a confident wrong answer. That researcher
> stands in the identical structural position as one concluding "actor–critic asymmetry hurts
> in reinforcement learning" from the navigation result. The only difference is that here we
> can *see* the error, because the same protocol run one rung over recovers the effect in
> full — a luxury the navigation study, having no `q=0` to fall back on, does not afford.

This is a positive control for the protocol, not a contribution about convolutional
architectures: we claim nothing beyond the well-known fact that locality helps when it is
present, and use it only to show the instrument reads *yes* as readily as *no*.

## 7. Why this task resisted every manipulation

That no rung induced the precondition is not a failure of effort — four qualitatively
different mechanisms were tried and capped by pre-registration. It follows from three
concrete, mutually reinforcing properties of the task, each of which we can point to in
the data.

**Observation redundancy.** The state is over-determined by the observation: different
features encode the same distinction, so removing one leaves it recoverable from
another. The sharpest instance is the aliasing rung. We dropped the region one-hot
(in-room / hallway / doorway) expecting to make those cell-types confusable — yet the
agent ceilinged *immediately* (IQM 27.17, faster than the un-aliased A-STRICT). The
reason is that the five proximity sensors already encode the local wall pattern, which
distinguishes a doorway (opening on two sides) from a corridor from an open room —
**proximity, not the region one-hot, is the true de-aliaser**, and it was retained. To
alias the state one must degrade *every* redundant encoding of a distinction at once;
degrading any single one is absorbed.

**Reactive optimality.** A near-optimal action is a function of the *current* frame, not
of history. This is why frame-stacking neutralises flicker up to `p=0.7`: with `k=4`
stacked frames the probability that all four are simultaneously masked is small, so a
recent true frame is almost always available, and a policy that maps the most-recent
unmasked frame to an action suffices. Only when masking is aggressive enough that long
all-blank runs become common (`p=0.8`) does the reactive strategy break — and then the
policy does not gracefully degrade into a harder function, it *gives up* (the systematic
kitchen-abandonment fracture), because nothing in the task rewards the partial,
history-integrating competence a fair test would need.

**A memorisable fixed map.** The single, static layout means the agent never needs a
general navigation policy — it can encode *this* house. This is why proximity noise
(`q=0.3`) is shrugged off with collisions at 0.00: on a memorised map the agent routes
from the target one-hot and its own trajectory, and does not depend on reliable
proximity readings to avoid walls, so corrupting them changes little. Degrading an input
the optimal policy has learned not to rely on cannot make the policy harder.

Together these give a single diagnosis: information can be removed or corrupted
extensively before the *optimal policy* becomes a genuinely harder function, because the
task affords a cheap solution (reactive, memorised, redundantly cued) throughout. Value
stays as easy as policy, and asymmetry has no wedge.

## 8. What a task would need to test this hypothesis

The null is generative: the same analysis that shows this task cannot test the hypothesis
specifies what a task that *could* would look like. We state it as a property list a
future benchmark can be built against — the natural Phase 3 target — rather than as a
vague call for "harder tasks."

- **Genuine, unroutable partial observability.** Hidden state that (i) cannot be
  recovered from any single observation *and* (ii) cannot be memorised away — i.e.
  procedurally generated or changing layouts, so no fixed-map shortcut exists and the
  optimal policy must integrate history on every episode. (This directly negates the
  "reactive optimality" and "memorisable map" escapes of §7.)
- **Non-redundant observation.** Each policy-relevant distinction is encoded once, so a
  targeted removal actually removes it. (Negates "observation redundancy".)
- **A longer horizon with real credit assignment.** Enough temporal depth that the
  optimal policy must compose sub-decisions, so representing it is a compositional
  problem rather than a one-step lookup.
- **A policy that is a sharp function even under full information.** Ideally the hardness
  is intrinsic to the optimal *mapping* (high-frequency, many decision boundaries), not
  merely induced by hiding information — so that policy capacity is taxed while value
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
check affordable, and its outcome — which rung, if any, is hard-but-learnable — tells you
where (or whether) to spend a claim-grade comparison. Pre-registration is doing real work
here: it converts an open-ended "try architectures until one wins" into a bounded,
falsifiable procedure whose negative outcomes are as informative as its positive ones.

We are careful about the scope of our negative result. We have shown that **this task —
and, by the mechanism of §7, static single-map gridworld navigation with a redundant,
low-dimensional observation — cannot test the inverted-asymmetry hypothesis**, because no
degradation of its observation places the precondition in a learnable regime. We have
**not** shown that inverted actor–critic asymmetry fails to help in reinforcement
learning generally; the hypothesis may well hold on a task of the kind §8 specifies, and
nothing here bears on that. The contribution is precisely scoped on purpose: a rigorous
demonstration that a specific, widely-used *style* of task cannot test a specific
architectural hypothesis, together with a reusable protocol for detecting this failure
mode before it is mistaken for evidence either way. A single-seed positive result on such
a task — our own starting point here — is exactly what the protocol is designed
to catch.

A protocol that only ever returned *no* would be a rejection stamp, not an instrument;
§6.1 answers that objection directly. On a vision task where the precondition can be
inserted and removed at will, the same procedure returns a clean *yes* where spatial
locality is present and *no* where it is not, tracking the effect as it appears and
vanishes — the positive branch, demonstrated. The remaining limitation is specific and we
state it plainly: that positive instantiation is on image classification, not on
reinforcement learning. We have exhibited the protocol's positive branch (on vision) and
its negative branch (on RL navigation), but **not a substantive positive instantiation on
an RL task** — one of the kind §8 specifies, where the asymmetry precondition is present in
a learnable regime and the architecture comparison can actually run. Constructing that task
and testing inverted asymmetry where it *can* be tested is the natural next step; the
specification in §8 is written to make it buildable.
