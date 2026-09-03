# Positive Control for the §6 Protocol — Pre-Registration

**Status:** FROZEN before any run. Do not edit predictions, the ladder, or the
interpretation table after the first run is launched. Design changes require a dated
amendment at the bottom, not an in-place edit.

**Author:** Samuel Babalola. **Date registered:** 2026-07-09.
**Relation to main paper:** a *positive control* for the protocol of §6 of
`docs/paper_draft.md`. It is not a contribution about convolutions; it exists only to
show the protocol can return its **positive** branch, on a task where the hypothesis's
precondition can be dialed in and out.

---

## 0. Why this exists

The navigation study exercises the §6 protocol on a case where it returns **negative**
("this task cannot test the hypothesis"). A reviewer will reasonably ask whether the
protocol is a detector that *only ever says no* — whether its ladder can return
**positive** at all. This control answers that: on a task where we can insert and remove
the precondition at will, the protocol must say **yes** where the precondition is present
and **no** where it is absent, tracking the effect as it appears and vanishes.

We deliberately choose an `A`-suits-`S` pairing that is **uncontroversial and known-true**,
so we are *recovering* a well-established effect, not asserting a new one:

- **`A` = convolution.**  **`S` = spatial locality** (neighbouring input dimensions are
  semantically related; a translation-equivariant local-receptive-field prior helps).

---

## 1. Design (frozen)

**Dataset:** MNIST (one dataset — see scope limits, §6). Standard train/test split.

**Ladder — graded pixel permutation.** A **single, fixed** permutation regime applied
identically to every image and to both train and test, parameterised by a scramble
fraction `q ∈ {0, 0.25, 0.5, 0.75, 1.0}`:

- Fix, once (seed-fixed, shared by all images and both splits), a nested family of pixel
  position-subsets `M_0 ⊂ M_0.25 ⊂ M_0.5 ⊂ M_0.75 ⊂ M_1`, with `|M_q| = round(q · 784)` and
  `M_1` = all 784 positions.
- Fix, once, a derangement of each `M_q` (identity outside `M_q`), giving a bijection
  `σ_q` on the 784 pixel positions. `σ_0` = identity (natural images); `σ_1` = a fixed full
  derangement of all pixels.
- Every image `x` at rung `q` is presented as `x[σ_q]` — the same pixel relabeling for all
  images, train and test.

**Why fixed, and why this preserves information (the essential control property).** Because
`σ_q` is a fixed bijection on positions, each rung is a *fixed relabeling of the input
units*: no information is lost (the classes remain perfectly separable — an MLP's first
layer can in principle invert any fixed permutation), and label-relevant content is
untouched. What *is* progressively destroyed as `q → 1` is **spatial locality** — the very
structure `S` that convolution exploits. A **per-image** permutation would instead destroy
information (different scramble per sample) and confound "locality removed" with "task made
unlearnable"; we do **not** do that.

**Architectures (2).** A small **CNN** (local receptive fields + weight sharing) and an
**MLP**, with **roughly matched parameter counts** (target: within ~10% of each other).
Everything else is held fixed across both architectures and all five rungs: optimiser
(Adam), learning rate, batch size, epoch/step budget, weight-init scheme, and data
pipeline. Only the architecture and `q` vary.

**Training budget (fixed, sub-asymptotic — pre-committed).** The convolutional advantage is
widest *before* convergence; trained to saturation, both models approach the same MNIST
ceiling and `gap(q=0)` shrinks toward the noise floor. To keep the gap resolvable we fix a
modest budget: **2 epochs** over the MNIST training set (Adam, lr `1e-3`, batch size 128),
**identical** across both architectures, all five rungs, and all seeds. Neither architecture
is trained to saturation.

**Seeds & aggregation.** **10 seeds** per (architecture × rung) cell — matching the
navigation study. Metric: **test accuracy**. Aggregate across seeds with **IQM + 95%
stratified-bootstrap CI** (rliable), as elsewhere in the paper.

---

## 2. Pre-registered predictions (frozen BEFORE running)

- **P1 (CNN).** CNN accuracy is high at `q=0` (expect ~98–99%) and **decreases
  monotonically** as `q → 1`, approaching the MLP.
- **P2 (MLP).** MLP accuracy is **flat** across `q` — an MLP is invariant to a fixed
  permutation of its inputs (a relabeling of input units), so scrambling locality should not
  change what it can learn. **Operationalised (pre-committed threshold, so the confound
  branch can fire cleanly rather than being judged post-hoc):** *P2 holds iff*
  `|IQM_acc(MLP, q=1) − IQM_acc(MLP, q=0)| < 1.0 percentage point`. A drop of ≥1.0pp from
  `q=0` to `q=1` means P2 **fails** — the permutation destroyed information, not only
  locality (a bug, not a result; see §4). As corroboration we additionally expect the `q=1`
  MLP IQM to lie within the 95% CI of the `q=0` MLP IQM.
- **P3 (gap).** `gap(q) = IQM_acc(CNN) − IQM_acc(MLP)` is **large with a 95% CI excluding 0
  at `q=0`**, **shrinks monotonically** with `q`, and is **≈0 (CI includes 0) at `q=1`**.
- **P4 (protocol verdict per rung).** Applying §6 step 5: at **`q=0` the task CAN test
  `A`-suits-`S`** (S present; the architecture gap is real and resolvable); at **`q=1` the
  task CANNOT** (S absent by construction, `gap ≈ 0`, so a null there is uninformative).
  Intermediate `q` are graded — the protocol's verdict moves with the presence of `S`.

---

## 3. Why the MLP baseline is load-bearing (the S-attribution point)

The MLP curve is not decoration — it is what **licenses attribution of the CNN's decline to
`S` specifically**. Its flatness (P2) demonstrates that the task did **not** get harder as
`q` rose (information is preserved; a permutation-invariant model is unaffected); therefore
the CNN's fall can only be due to the removal of the structure convolution exploits, i.e.
`S`, and not to some confounding increase in difficulty. This is precisely the **§6 step-5
S-attribution guard** exercised on a case where attribution **succeeds** — the complement of
the navigation `p=0.8` fracture, where a rung became hard *the wrong way* and attribution
failed. Together the two cases show the guard doing its job in both directions.

---

## 4. Interpretation table (frozen)

| Outcome | Meaning |
|---|---|
| CNN high→MLP (monotone), MLP flat, gap large-and-CI-excludes-0 at q=0 → ≈0 at q=1 | **Positive control passes.** On one task, the protocol returns **positive** where S is present and **negative** where S is absent — it tracks the effect with S. Exactly the reviewer answer. |
| MLP **not** flat (accuracy falls with q) | **Confound, not a result.** The permutation destroyed information, not only locality (bug in the bijection / pipeline). Fix and re-run; the control is invalid until P2 holds. |
| gap non-monotone, or gap at q=1 **not** ≈0 | Partial: the dose-response is not clean. Report honestly; inspect whether the CNN retains a locality-independent edge (e.g. capacity) — would qualify, not overturn, the control. |
| CNN ≈ MLP even at q=0 | The chosen A/S instance did not exploit S even when present (unexpected for conv/MNIST) — falsifies this *example*, not the protocol; pick a cleaner A/S. |

---

## 5. Falsification criteria

- **P2 fails** (MLP degrades with `q`) ⇒ information not preserved ⇒ the control is
  confounded and **falsified as constructed**; must be fixed (verify `σ_q` is a bijection and
  applied identically to train/test) before any claim.
- **P3 fails** (gap does not shrink monotonically, or is not ≈0 at `q=1`) ⇒ the positive-
  control prediction is not met; reported as such, not massaged.
- **P1 fails** (CNN does not lead at `q=0`) ⇒ the A/S example is wrong; report and reconsider.

**Saturation contingency (pre-committed, 2026-07-09 — decided before any data).** MNIST is
nearly saturated for both models, so `gap(q=0)` may be too small to resolve, and the whole
control hinges on `gap(q=0)` clearly excluding 0. **Decided in advance:** if `gap(q=0)`'s
95% CI **includes 0** on MNIST (with the fixed 2-epoch budget above), MNIST is too saturated
to serve as a positive control; we **escalate to CIFAR-10 and re-run the full ladder** —
identical design (same five `q` rungs, CNN vs param-matched MLP, 10 seeds, sub-asymptotic
budget). This is a dated, pre-registered contingency, **not** a post-hoc dataset swap.

---

## 6. Scope discipline (HARD limits)

**One** dataset (MNIST), **two** architectures (CNN, MLP), **five** rungs
(`q ∈ {0,0.25,0.5,0.75,1.0}`), **10** seeds, **one** figure, **~half a page** of prose.
This is a positive control, **not** a contribution about convolutions or image
classification. **Do not** add datasets, architecture variants, hyperparameter sweeps, or
ablations. **If a fourth experiment suggests itself, STOP and ask.**

---

## 7. Deliverables

1. This pre-registration (committed **before** any run).
2. The ladder + training implementation (fixed-permutation transform; CNN/MLP; runner).
3. Multi-seed results: test-accuracy IQM + 95% CI per (architecture × rung).
4. **One** figure: the dose–response — CNN and MLP accuracy vs `q`, with the shrinking gap.
5. A draft `§`-insert for the paper: *the protocol tracks the effect appearing and vanishing
   with `S`; where `S` can be dialed in, it says yes; on the navigation task, no manipulation
   could dial `S` in at all — the instrument works, the benchmark doesn't.*
6. A short rhetorical paragraph (the "q=1-only researcher" analogy).

---

## 8. Commit protocol

- Commit this file **before** any run: `Positive control: pre-register the convolution /
  spatial-locality ladder (frozen before any run)`.
- Code under `Analysis/` or a clearly-scoped module; results CSV under `results/csv/`;
  figure under `figures/` with a manifest entry. Frozen predictions above; any design change
  is a dated amendment here, not an in-place edit.

---

## Amendments

**Pre-run tightening — 2026-07-09 (before any run; approved).** Two operational tightenings
applied to the frozen design *before* building or collecting any data: (1) **P2 given a
falsifiable threshold** — `|Δ IQM_MLP(q=0→q=1)| < 1.0 pp`, so the confound branch fires by a
pre-committed rule rather than a post-hoc flatness judgement (§2, §4); (2) a **saturation
contingency** — escalate to CIFAR-10 if `gap(q=0)`'s 95% CI includes 0 (§5) — plus a **fixed
sub-asymptotic 2-epoch budget** (§1) chosen so neither model saturates and the gap stays
resolvable. No data existed at amendment time; predictions P1/P3/P4, the ladder, and the
interpretation table are unchanged.
