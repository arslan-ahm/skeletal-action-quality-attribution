# Results

Every table below is generated from the CSV named next to it by
`scripts/render_docs.py`, which reads `results/tables/`. No number in this file
was typed by hand; `python scripts/render_docs.py --check` fails if any of them
has drifted from its artefact.

The machine that produced them is a **4-core Windows laptop with no GPU, capped
to 2 torch threads, running other jobs at the same time.** That constraint is
visible in the scale of these experiments and it is not hidden.

<!-- SUMMARY -->

## What was run, and what was not

Being precise about this matters more than the numbers.

| claim | status | where |
|---|---|---|
| Parameters, MACs, latency for eight architectures | **measured** | `results/tables/efficiency.csv` |
| DTW cost is quadratic in sequence length, the model's is linear | **measured, exponents fitted** | `results/tables/cost_vs_length.csv` |
| Quality regression, every method, one dataset and one loop | **measured** | `results/tables/method_comparison.csv` |
| Paired significance against the reference approach | **measured** | `results/tables/statistical_tests.csv` |
| DTW baseline swept over 12 configurations, chosen on validation | **measured** | `results/tables/dtw_sweep.csv` |
| Attribution scored against exact ground truth | **measured** | `results/tables/attribution_fidelity.csv` |
| Attribution vs random and vs an untrained model | **measured** | same table |
| Monotonicity-violation rate on severity ladders | **measured** | `results/tables/monotonicity.csv` |
| Ordinal rank inconsistency | **measured, and it is 0 by construction** | same table |
| Interval coverage, sharpness, risk-coverage | **measured** | `results/tables/uncertainty.csv` |
| Seed-to-seed variation, 3 seeds | **measured** | `results/tables/seed_variance.csv` |
| Component ablation, one variable at a time | **measured, shorter schedule** | `results/tables/ablation_components.csv` |
| Leakage: random vs cross-subject vs cross-degradation | **measured** | `results/tables/split_comparison.csv` |
| Label-budget curve | **measured** | `results/tables/data_efficiency.csv` |
| **Accuracy of the 0.76M and 3.02M reference ST-GCNs** | **not run** — over the per-run budget | `notebooks/05_colab_full_scale.ipynb` |
| Agreement with a human judge's score | **not possible here** — the label is synthetic | §7 |
| Any real dataset | **not run** | `scripts/download_real.py` |

There are no placeholder numbers in this repository. Anything absent from
`results/` was not run, and is not claimed.

## 1. Efficiency

Parameters and MACs are inputs to a cost argument; wall-clock is the argument.
All three are reported, along with the achieved MACs-per-millisecond, because a
FLOP reduction does not convert to time one-for-one.

<!-- table:efficiency -->
_not measured_
<!-- /table -->

Measured with 10 warm-up iterations and 30 timed repeats, median and IQR, on 2
torch threads at 48 frames. **Read the IQR column.** This machine was shared, and
a row whose IQR approaches its median is not trustworthy — an earlier run of this
same table had `stgcn_dense` (43.8 MMACs) measuring *slower* than
`stgcn_reference` (167.5 MMACs), which is arithmetically impossible and was pure
contention.

### 1.1 The cost that actually separates the two approaches

The reference approach pays `O(T²)` twice — once to build the frame cost matrix
and once in the dynamic program. The graph model is `O(T)`. This is measured
across sequence lengths and the exponent is fitted on the log-log slope rather
than asserted:

<!-- table:cost_curve -->
_not measured_
<!-- /table -->

This is the efficiency claim that matters against the reference repository, and
it is a claim about *asymptotics that are visible at realistic lengths*, not a
constant factor.

## 2. Quality regression

Every arm trained through one loop, on one dataset, with one schedule and one
seed. `model.architecture` is the only thing that differs between the neural
arms.

<!-- table:method -->
_not measured_
<!-- /table -->

### 2.1 The reference approach, given a fair fight

Twelve DTW configurations — three distance functions x banded/unbanded x
canonical/exemplar reference — each with per-action isotonic calibration onto the
quality scale, **selected on validation**:

<!-- table:dtw_sweep -->
_not measured_
<!-- /table -->

The single largest fairness fix was per-action calibration. Distances are not
comparable across action classes, so one global monotone map conflates "which
action" with "how good". Measured on a 300-sequence pilot: **ρ = −0.12 without
it, ρ = +0.44 with it.** Reporting the first number would have been a strawman,
and it is the number a careless implementation produces.

### 2.2 Is the difference significant?

<!-- table:statistics -->
_not measured_
<!-- /table -->

**What a paired test here does and does not say.** It conditions on **one
trained model per method** and asks whether the difference is consistent across
test sequences — which it answers correctly. But "this method is better" treats
the **training run** as the sampling unit, and there is one run per method. No
number of test sequences fixes an n of 1. That is why §5 exists and why every
claim in this file is placed against the seed-study noise scale before it is
believed.

Spearman is a set-level statistic with no per-item value, so it cannot be tested
with a Wilcoxon test at all. It is tested by resampling *sequence indices* and
recomputing the metric for both methods on each resample.

### 2.3 Per action class

<!-- table:per_action -->
_not measured_
<!-- /table -->

## 3. Attribution fidelity

This is what the project is for. The explanation is scored against the exact
cause, and against two controls that a convincing-looking heat map must beat:
a uniform-random attribution, and **the same method applied to an untrained
model** — the parameter-randomisation sanity check of Adebayo et al. (2018).

<!-- table:attribution -->
_not measured_
<!-- /table -->

`n_joint_iou` is the number of test sequences that contributed: clean sequences
have no cause to attribute and are excluded rather than scored as perfect.

<!-- ATTRIBUTION_READING -->

## 4. Monotonicity and calibrated uncertainty

### 4.1 Does adding a defect ever raise the score?

Severity ladders fix the subject, the action, the defect target and all of the
defect's internal randomness, and vary only the severity — with the sensor model
switched off, so a violation cannot be caused by a noise realisation.

<!-- table:monotonicity -->
_not measured_
<!-- /table -->

`rank_inconsistency` is a **structural** property of the shared-latent ordinal
head and is 0 by construction; the independent-logit variant produces crossings
on over half of all inputs
(`test_independent_variant_does_produce_crossings`). `violation_rate` is the
*empirical* property, and the head does not guarantee it — which is exactly why
it is measured.

### 4.2 Intervals

<!-- table:uncertainty -->
_not measured_
<!-- /table -->

Coverage alone cannot distinguish a useful interval from a constant-width one, so
`width_ratio_wrong_right` is reported next to it: above 1 means the interval
widens on the sequences the model gets wrong, which is the property that makes
abstention work. `e_aurc` is AURC minus the oracle AURC, which separates "the
confidence signal is good" from "the model is accurate".

## 5. Seed variance — read this before any ablation

Three training seeds, identical configuration. The seed changes initialisation,
batch order **and** the split, which is the conservative choice: it measures the
variation a reader reproducing the pipeline would see.

<!-- table:seeds -->
_not measured_
<!-- /table -->

Two independent runs differ with standard deviation `sqrt(2) * sd`. Any claimed
improvement smaller than that column is **noise**, and is labelled as such below.

## 6. Ablations

One variable at a time, one shared dataset, and a **shorter schedule** (8 epochs
against the headline 10) so that nine variants fit in the budget. The table
carries its own `full` reference row, so it is internally consistent; do not
compare its absolute numbers against §2.

<!-- table:ablation -->
_not measured_
<!-- /table -->

Every delta is divided by the §5 noise scale before it is interpreted. A verdict
of `inside noise` means the ablation did not measure anything at this scale, not
that the component does nothing.

## 7. Splits, and the leakage this repository measures rather than assumes

<!-- table:splits -->
_not measured_
<!-- /table -->

The gap between `random` and `subject` is the leakage estimate. A random split
puts the same subject — same limb lengths, same style factor — on both sides;
`combination` additionally holds out every multi-defect sequence containing a
compensatory pattern, so the model must generalise to an interaction it has
never seen.

## 8. Label efficiency

Reference-free quality assessment needs a human-scored corpus, and the scoring is
the expensive part. This is the practically important efficiency axis.

<!-- table:data_efficiency -->
_not measured_
<!-- /table -->

## 9. Limitations, and what would change them

**The label is synthetic, and that bounds every accuracy claim here.** `w_k` is a
modelling choice, not a measured human judgement. What this repository
demonstrates is that a model can recover a *known* score function and a *known*
cause, and how well. Whether the same machinery predicts a diving judge is a
different question that needs MTL-AQA or FineDiving; the loader and the download
script are provided and neither was run.

**One training run per arm.** Three seeds for the headline configuration, one for
each ablation. Settling an accuracy ranking needs 5–10 per configuration, which
is tens of CPU-hours here and under an hour on the GPU that
`notebooks/05_colab_full_scale.ipynb` targets.

**The reference-scale architectures were not trained.** Their cost is measured;
their accuracy is not claimed.

**48 frames.** The ST-GCN regime is 300. The temporal receptive field ablation is
therefore testing a smaller range of kernel sizes than it would at full length.

**Attribution ground truth is a displacement, not a causal counterfactual in the
model's own terms.** It answers "what did this defect do to the movement", which
is the right target for a coaching explanation, but a model could in principle
score well on it while using different features internally.

<!-- RETRACTIONS -->
