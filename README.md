# Testable or Not? A Pre-Registered Validity Protocol for Architecture Comparisons

This repository accompanies the paper *Testable or Not? A Pre-Registered Validity Protocol for Architecture Comparisons.* It contains the pre-registrations, per-seed data, analysis code, environment, and paper source needed to reproduce every claim in the paper under a seeded bootstrap.

## What this paper does

Empirical architecture comparisons rest on an implicit assumption of testability: that the benchmark can actually arbitrate the hypothesis. When it cannot, statistical rigor measures the wrong thing. We give a pre-registered protocol for checking whether a task can arbitrate an architecture hypothesis at all, and demonstrate it on a case where the answer is no — including the mechanism that explains why. A positive control on image classification confirms the protocol also returns yes when the precondition is met.

## Repository origin

This repository originated as the *Policy-Hard, Value-Easy* actor–critic study (unpublished, DLI 2026, rejected). The current paper is a construct-validity re-examination of that earlier work: it audits the original single-seed claims, finds they do not survive multi-seed evaluation, and abstracts the process into a reusable protocol. The old README and single-seed numbers have been superseded; see [`docs/rejected_submission_DLI2026.pdf`](docs/rejected_submission_DLI2026.pdf) for the original submission if needed.

Concretely, the claims that did not survive re-examination were:

| Original single-seed claim | Under 10-seed evaluation |
|---|---|
| Inverted asymmetry is the superior PPO architecture | IQM 27.56 vs 27.66, overlapping 95% CIs, bootstrap *P*(inverted > symmetric) = 0.25 |
| PPO is ≈12.5× more sample-efficient than DQN | Step-to-90% distributions overlap; the slowest configuration in either family is a PPO one |

The more useful finding is *why* the first null is uninformative: the task never contained the structure the hypothesis is about, so no comparison run on it could have arbitrated the claim either way.

## Repository structure

- **`Environment/`** — custom sparse-reward navigation env plus the four observability-degradation wrappers (pure removal, flicker + frame-stack, targeted aliasing, proximity noise).
- **`Training/`** — training entry points (PPO, DQN), the sweep runners, per-seed reproducibility scaffolding (`seeds.py`, `trainer_common.py`), and baselines (A2C, Double DQN, Dueling DQN).
- **`Analysis/`** — `rliable_analysis.py` (main aggregation, includes Section G for Phase 2), `positive_control.py` (MNIST CNN vs MLP), `plot_p2_ladder.py`, `rung3_difficulty_model.py`.
- **`configs/`** — YAML configs for all nine Phase-1 cells, three baselines, and six Phase-2 ladder cells. Every hyperparameter lives here; none are hardcoded.
- **`results/csv/`** — per-seed CSVs for every claim-grade cell (19 files). Every aggregate in the paper regenerates from these under the seeded bootstrap.
- **`figures/`** — the eight paper figures at final `REPS=50000` seeded settings. [`figures/README.md`](figures/README.md) is the figure manifest.
- **`paper/`** — LaTeX source for TMLR (`main.tex`), the TAE workshop (`tae/`), and the NewInML workshop (`newinml/`). Each is self-contained and builds independently.
- **`docs/`** — the full paper draft in markdown ([`paper_draft.md`](docs/paper_draft.md)), the two frozen pre-registrations, the append-only [results log](docs/results_log.md), and the outline.

## Reproducing the paper

**Prerequisites.** Python 3.10+. Clone and install with:

```bash
git clone https://github.com/SammyGbabs/Pre-Registered-Validity-Protocol-for-Architecture-Comparisons.git
cd Pre-Registered-Validity-Protocol-for-Architecture-Comparisons
pip install -r requirements.txt
```

**Regenerate all Phase-1 and Phase-2 aggregates and figures:**

```bash
python Analysis/rliable_analysis.py
```

Reads from `results/csv/`, writes updated figures to `figures/`. Uses `BOOTSTRAP_SEED=20260817` (overridable via `ARCH_ABLATIONS_BOOTSTRAP_SEED`); every aggregate reported in the paper regenerates to the digit. Useful environment variables: `ARCH_ABLATIONS_CSV_DIR` (input directory), `ARCH_ABLATIONS_FIG_DIR` (figure output), `ARCH_ABLATIONS_REPS` (bootstrap resamples), and `ARCH_ABLATIONS_WRITE_FIGS=0` for a numbers-only pass that writes no figures.

**Regenerate the Phase-2 ladder figures:**

```bash
python Analysis/plot_p2_ladder.py
```

**Re-run experiments from scratch** (Colab-friendly, GPU recommended):

```bash
python -m Training.run_sweep --output-dir <path>          # Phase 1
python -m Training.run_phase2_sweep --output-dir <path>   # Phase 2 ladder
python Analysis/positive_control.py run                   # MNIST positive control
```

Each cell writes per-seed CSVs and W&B logs. Runs are resumable via `.done` markers, so re-issuing the same command skips finished seeds and resumes partial ones from their latest checkpoint. Pass `--dry-run` to print the queue without launching, or `--only <config_id>` to run a single Phase-2 cell in isolation.

## Pre-registrations

Both are frozen and were committed before any load-bearing run:

- [**`docs/PHASE2_POMDP_PREREGISTRATION.md`**](docs/PHASE2_POMDP_PREREGISTRATION.md) — the observability ladder, four mechanisms, per-outcome interpretation table, plus Amendments 1–4 documenting each extension.
- [**`docs/POSITIVE_CONTROL_PREREGISTRATION.md`**](docs/POSITIVE_CONTROL_PREREGISTRATION.md) — MNIST CNN-vs-MLP dose-response, four frozen predictions (P1–P4).

One of those predictions (P3) turned out to be partially wrong. It is reported as a miss in the paper rather than quietly revised, which is the point of freezing them.

## Reproducibility guarantee

Every numerical claim in the paper — every IQM, every 95% CI, every per-room success rate, every distinct-pattern count — regenerates from `results/csv/` under the seeded bootstrap. A 71-check numerical sweep confirmed this before final submission; see [`docs/results_log.md`](docs/results_log.md) for the audit trail, including a corrected per-room success rate that the sweep caught.

Evidence tiers are kept strictly separate and are labelled throughout: claim-grade multi-seed results (10 seeds, IQM with 95% stratified-bootstrap CIs, released per-seed data) versus single-seed learnability gates, which are design and calibration steps only and are never cited as results.

## Citation

```bibtex
@article{babalola2026testable,
  title  = {Testable or Not? A Pre-Registered Validity Protocol for Architecture Comparisons},
  author = {Babalola, Samuel and Odero, Anjeline Noel and Sumba, Branis Mabumba and Ogore, Marvin and Nsabiyumva, Simeon},
  year   = {2026},
  note   = {Under review at TMLR; preprint at arXiv:XXXX.XXXXX}
}
```

(Update the arXiv ID once posted.)

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

We thank Eunice Adewusi for her contributions to the earlier unpublished version of this work, including the ethics review and writing of the ethics section.
