# CLAUDE.md

This file is read at the start of every Claude Code session in this repo. It defines what this project is, the conventions to follow, and the decisions already made so we don't relitigate them every session.

---

## Project: one paragraph

This is the rework of a rejected DLI 2026 submission titled *"Policy-Hard, Value-Easy: Inverted Actor-Critic Asymmetry in Deep RL for Assistive Indoor Navigation."* The working framing is **architectural ablations for sparse-reward navigation**. The central hypothesis is that, in sparse-reward grid-world navigation, an actor network with greater capacity than the critic (inverted asymmetry) outperforms symmetric and conventional configurations at fixed parameter budget. The rejected version showed this on one map with one seed; this rework establishes it (or refutes it) with multi-seed evaluation, capacity-controlled ablations, procedural map generalisation, cross-environment validation on MiniGrid, and empirical Lipschitz measurements.

The full rework plan, reviewer mapping, and timeline live in `docs/rework_proposal.pdf`. Read it before making structural decisions about the codebase.

---

## Pre-registered hypotheses (do not rewrite without team discussion)

- **H1.** At fixed parameter budget, inverted asymmetry (actor:critic > 1) achieves higher IQM return than symmetric (1:1) and conventional (< 1) ratios.
- **H2.** The advantage is attributable to the ratio, not the total parameter count.
- **H3.** Inverted asymmetry generalises: smaller train-test gap on procedural layouts than conventional configurations.
- **H4.** Empirical Lip(π) / Lip(V) ≥ 2 across configurations.
- **H5.** On MiniGrid (FourRooms, MultiRoom, DoorKey), inverted asymmetry produces sample-efficiency gains over symmetric PPO.
- **H6.** Hybrid PPO+DQN achieves PPO's step count within 10% AND DQN's collision rate within 1 percentage point.

Each hypothesis has a falsification criterion in `docs/rework_proposal.pdf` §3.2.

---

## Phases and gates

- **Phase 1 — Multi-seed replication (gating).** 90 runs, 10 seeds each. Replicate the original Tables 1 and 2 under proper statistics. Decision gate after Phase 1: does inverted asymmetry survive? If yes → asymmetry stays as headline claim. If no → reframe paper.
- **Phase 2 — Capacity-ratio ablation + procedural maps + POMDP variants.**
- **Phase 3 — Cross-environment validation (MiniGrid: FourRooms, MultiRoom-N2-S4, DoorKey-6x6) + Lipschitz measurement + dynamic obstacles + sensor noise.**
- **Phase 4 — Hybrid safety architecture, figure regeneration, optional GRPO.**

Do not start Phase N+1 work until Phase N runs are committed and analysed. Premature parallelisation creates confounded results.

---

## Repository layout

```
Environment/
  custom_env.py         # paper-aligned Gym env (20x20, Discrete(5), 16-d obs)
  rendering.py          # matplotlib renderer + map data (one source of truth)
  procedural.py         # BSP-based layout generator (Phase 2)
  dynamic.py            # moving-NPC wrapper (Phase 3)
  noise.py              # sensor-noise wrapper (Phase 3)
Training/
  seeds.py              # seed manager: every run is (algo, config_id, seed) tagged
  dqn_training.py
  ppo_training.py
  baselines/
    double_dqn.py
    dueling_dqn.py
    a2c.py
Analysis/
  rliable_analysis.py   # IQM, performance profiles, prob-of-improvement (use # %% cells, not .ipynb)
  lipschitz.py          # empirical local Lipschitz on saved checkpoints
  astar.py              # shortest-path baseline for Path Optimality Ratio
  trajectories.py       # qualitative behaviour viz
hybrid/
  safety_filter.py      # PPO planner + DQN safety filter (Phase 4)
configs/
  *.yaml                # ALL hyperparameters live here, never hardcoded
figures/
  README.md             # naming convention + manifest of every figure
  *.png                 # all paper/slide figures, regenerable from Analysis/
results/
  csv/                  # per-seed raw numbers, one CSV per experiment
docs/
  rework_proposal.docx             # the active plan (editable source)
  rework_proposal.pdf              # the plan (PDF export of rework_proposal.docx)
  results_log.md                   # append-only structured results log (paper writes itself from this)
  rejected_submission_DLI2026.pdf  # the rejected Deep Learning Indaba 2026 paper (frozen reference)
  Final_Report.pdf                 # the pre-paper original report, written before the DLI submission (frozen reference)
main.py                 # rollout/visualisation entry point
requirements.txt
CLAUDE.md               # this file
```

Anything not under one of these directories needs a justification before it lands.

---

## Conventions (enforce these)

### Code

- **Python ≥ 3.10**, type-hinted where it isn't a chore.
- **Gymnasium**, not `gym`. SB3 ≥ 2.0.
- **All hyperparameters in `configs/*.yaml`.** Never hardcode learning rates, network sizes, batch sizes in scripts. If a script needs a number that isn't in a config, add it to one.
- **All training runs log to W&B.** Run names: `{algo}_{config_id}_seed{N}`. Project: `arch-ablations-sparse-reward`.
- **Checkpoint every 25k env steps.** Resume-from-checkpoint must work. We will lose Colab sessions; assume it.
- **Seed everything.** Numpy, torch, env. Every script takes `--seed` as a CLI arg. The seed manager in `Training/seeds.py` is the source of truth.
- **No silent failures.** If a run errors mid-training, raise. Don't swallow exceptions to "keep going."
- **Analysis as `.py` with `# %%` cell markers, not `.ipynb`** — diffs are reviewable, version control is sane. Notebooks are for one-off exploration only and live under `Notebooks/scratch/` (gitignored).

### Statistical reporting

- **rliable** (Google Research) is the project-wide evaluation library. IQM with 95% stratified-bootstrap CIs.
- **10 seeds for main results, 5 for exploratory, 3 only for sanity checks.** Never 1.
- **X-axis is environment steps, never episodes.** Cross-algorithm comparisons in episodes are misleading; we promised reviewers we wouldn't do this.
- **Every aggregate number in the paper carries a 95% CI.** No exceptions.
- **Per-seed raw numbers go to a public CSV** alongside the code.

### Git

- **Branch protection on `main`.** All work goes through PRs with at least one review.
- **One PR per atomic change.** A PR that touches the training script and the analysis pipeline is two PRs.
- **Commit messages reference the phase and hypothesis** when relevant: e.g., `Phase 2: H2 capacity sweep config (200k budget)`.
- **Tag releases at phase boundaries:** `phase1-complete`, `phase2-complete`, etc.

---

## What Claude Code should do by default

- Read `docs/rework_proposal.pdf` if a request touches scope or framing.
- Prefer reading `Environment/rendering.py` first when working on env code (it owns the map data; `custom_env.py` imports from it — do not duplicate).
- Write tests alongside code, even for "obvious" things like the procedural generator. The reviewers will look at test coverage.
- For new scripts: add a CLI with `argparse`, a `--seed` flag, a `--config` flag pointing to a YAML, and a smoke-test block under `if __name__ == "__main__"`.
- When unsure whether something should be a config field or a CLI flag: it's a config field. CLI flags are reserved for things that vary per-invocation (seed, output path, debug toggle).
- When generating plots: use seaborn defaults, the project palette (tab10 categorical / viridis sequential), axis labels with units, 300 DPI for paper figures, and save to `figures/` with a script in `Analysis/` whose filename matches the figure. **Add a manifest entry in `figures/README.md` in the same change.** See `figures/README.md` for the full convention.
- **When a training run or analysis pass produces results worth reporting** (anything with seeds, CIs, or a comparison), **append a structured entry to `docs/results_log.md`** following the template at the top of that file. Headline numbers, interpretation paragraph, W&B link, generating script. Per-seed raw CSVs go to `results/csv/{phase}_{exp_id}.csv`, not into the log file.

## What Claude Code should NOT do

- Do **not** restart unsupervised long training runs. The user runs the first seed of every new configuration and inspects curves before delegating the remaining seeds.
- Do **not** add a dependency without justification. Check `requirements.txt` first; if it isn't there, ask.
- Do **not** introduce new hyperparameters in code without a corresponding YAML field.
- Do **not** change the reward function, observation space, action space, or grid size without explicit user confirmation. These are the experimental contract.
- Do **not** add continuous action spaces, VLM integration, or sim-to-real hardware code. These are out of scope for this paper (see proposal §1, "What this rework deliberately will NOT do").
- Do **not** rewrite or reframe the pre-registered hypotheses. They were committed deliberately to prevent narrative drift.
- Do **not** use the old `gym` package, `MlpPolicy` with Dict observations, or `MultiInputPolicy` with flat Box observations. The env emits a flat 16-d Box; use `MlpPolicy`.

---

## Team

Two people. Default ownership:

- **Samuel** — lead author. Asymmetry hypothesis. Phase 1 multi-seed runs and statistical analysis. Phase 2 capacity sweep. Phase 3A MiniGrid runs (training queue). Paper rewrite. Final call on narrative shifts.
- **Collaborator** — co-author. Environment subsystem (procedural generator, dynamic and noise wrappers, MiniGrid integration: CNN encoder + obs wrappers). Phase 2 procedural-generalisation runs. Phase 3 Lipschitz pipeline, dynamic and sensor-noise ablations. Phase 4 hybrid safety filter and figure generation.

This split puts Samuel on the experiments-and-claims path and the collaborator on the infrastructure-and-analysis path. Whoever finishes a task first picks up the next available item; rigid ownership is less important than shipping. When in doubt about who owns a change, look at recent commits in the touched directory.

Compute reality: **one Colab Pro account between the two of us.** Training is serial, not parallel. Schedule runs on a single shared queue (W&B run table is the source of truth). Hand off the GPU between sleep cycles where possible (one person trains overnight in their timezone, the other during their day).

---

## Decisions already made (do not relitigate)

- **Grid stays 20×20.** Paper-aligned. Reviewer 3's "21×20" was based on a typo in the earlier draft, since corrected.
- **Action space is `Discrete(5)`** — Up, Down, Left, Right, Wait. Wait is a no-op that advances the clock and pays the step penalty.
- **Observation is a flat 16-d Box,** not a Dict. Layout: proximity (5) + target one-hot (4) + nav state (7).
- **Reward function:** −0.1 step, −5.0 collision (terminates), +1.0 doorway (once per cell per episode), +15 + 0.1·t_rem on target, −3.0 timeout.
- **Episode horizon T_max = 150.**
- **Agent spawn is the living-room corner `(0,0)`** (`AGENT_START` in `Environment/rendering.py`), and **the target room is sampled from the rooms other than the spawn room** (`{kitchen, bedroom, bathroom}`). *Signed-off contract change, 2026-06-24.* The reworked env had previously spawned at the hallway centre `(10,10)`, which put every room only 2–4 steps away — collapsing the task (mean optimal episode ≈ 3 steps) and erasing the architecture signal H1 depends on (symmetric and inverted configs both trivially hit the reward ceiling `R(L)=30−0.2·L ≈ 29.4`). The corner spawn restores cross-house navigation: mean optimal episode ≈ 14.7 steps (11–22), matching the paper's 14–55 / 14.3–28.6 regime. Note `(1,1)` (the original env's spawn) is a Sofa obstacle in the reworked map, so `(0,0)` is used. Excluding the spawn room from targets prevents the degenerate "already in the target room" episode. See `docs/results_log.md` (Phase 1 sanity-check).
- **Collision is termination,** not silent move cancellation. Strict paper semantics.
- **POMDP framing is being fixed** in Phase 2 (Variants A, B, C per proposal §5.2). Until then, the existing 16-d obs is what we use.
- **Continuous actions: out of scope.** Mentioned as future work in the paper, not implemented.
- **GRPO: optional Phase 4 stretch only.** Do not start without Samuel's go-ahead.
- **`Samuel_Babalola_RL_Summative` (the original repo)** is frozen at tag `v1.0-DLI2026-submission` and is the reference for the rejected submission. This repo (`Architectural_Ablations_Sparse_Reward_Nav`) is the active rework.
- **Team is 2 people (Samuel + 1 collaborator), not 4.** Compute is a single Colab Pro account, so training is serial. The two streams (experiments and infrastructure) run in parallel but the GPU is the binding constraint.
- **Timeline is 20 weeks**, not 16 — extended to keep cross-environment validation in scope.
- **MiniGrid cross-environment validation (FourRooms, MultiRoom, DoorKey) is in scope** as Phase 3A. This is a deliberate choice over a shorter timeline because cross-env evidence is what makes the asymmetry claim publishable beyond a single-domain study.
- **Phase 1 configs are verbatim ports of the paper's Table 1 (DQN ×5) and Table 2 (PPO ×4)** in `configs/{dqn,ppo}_exp{N}.yaml`. Each experiment has its own hyperparameters (do not copy one config's values to another). DQN Exp 1/3/4 and PPO Exp 2 exist **only in the paper tables, not in any committed code**, so they cannot be cross-checked against a script; DQN Exp 2/5 and PPO Exp 1/3/4 match the original notebook exactly.
- **Phase 1 seed tiers:** the 9 table configs run at 10 seeds (MAIN), the 3 baselines (Double DQN, Dueling DQN, A2C) at 5 seeds (EXPLORATORY) → **90 + 15 = 105 runs**. A baseline is promoted to 10 seeds only if its 5-seed IQM lands close enough to PPO Exp 1/Exp 4 to affect the conclusion. Baselines are reported as exploratory context, not central H1 evidence.
- **A2C is a "standard alternative" baseline, not a controlled ablation.** Its architecture is matched to the symmetric PPO Exp 1 (so it doesn't confound the asymmetry test), but it uses A2C's own design-appropriate rollout (`n_steps=5`, the SB3 default), so **PPO-vs-A2C differs in both algorithm and rollout length** — not a single-variable comparison. Double/Dueling DQN are matched to DQN Exp 5 (the best DQN config).

---

## When this file is out of date

If a session's work materially changes one of the items above — for example, Phase 1 results refute H1 and we reframe the paper — the change should land as a PR to this file in the same commit that makes the underlying decision. Future Claude Code sessions read this file; out-of-date guidance here is worse than no guidance.
