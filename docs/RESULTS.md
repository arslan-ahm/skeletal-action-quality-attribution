# Results

Every table below is generated from the CSV named next to it by
`scripts/render_docs.py`, which reads `results/tables/`. No number in this file
was typed by hand; `python scripts/render_docs.py --check` fails if any of them
has drifted from its artefact.

The machine that produced them is a **4-core Windows laptop with no GPU, capped
to 2 torch threads, running other jobs at the same time.** That constraint is
visible in the scale of these experiments and it is not hidden.

**Read this first.** The headline experiment of this project — validating
per-joint attribution against exact ground truth — returns a **negative result**,
and a three-seed study **retracts** the significance verdict of the paired tests
reported in §2.2. Both are stated where the claim was made rather than quietly
restated at the end.

The short version:

* **Solid, because they are measurements or arithmetic rather than inferences:**
  parameters/MACs/latency; the fitted `O(T²)` exponent of DTW against the model's
  `O(T)`; zero crossed quantile intervals and zero ordinal rank inconsistencies;
  89.6% coverage at nominal 90%; bit-identical reproducibility.
* **The central experiment fails, robustly.** Attribution rank correlation with
  the true cause is −0.24 to +0.01 for trained models, 0.001 for random, and
  **+0.27 for the *untrained* control**. Across every architecture and every
  attribution method.
* **Two accuracy claims survive the noise:** temporal modelling beats
  frame-averaging (2.17x), and training beats random weights (4.60x).
* **No ranking among the five competitive methods survives** — every gap is
  ≤1.4x the run-to-run scale of 0.1205, including this repository's model against
  the reference DTW approach it was built to replace.
* **The ordinal head does not deliver degradation monotonicity** (16.2% of
  ordered pairs), which `docs/METHOD.md` predicted it would not guarantee.
* **Predictive intervals do not support abstention** for any method here.

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
| method | family | spearman | kendall_tau | relative_l2 | mae | coverage | mean_width | error_auroc | params |
|---|---|---|---|---|---|---|---|---|---|
| kinematic_gbr | baseline | **0.5968** | 0.4271 | **0.214** | 0.1289 | 0.88 | 0.5693 | 0.5398 | n/a |
| dtw_reference | baseline | 0.4996 | 0.3467 | 0.2407 | 0.1452 | 0.864 | 0.5396 | 0.4744 | n/a |
| framewise_reference | baseline | 0.4191 | 0.295 | 0.2524 | 0.1512 | 0.868 | 0.5703 | 0.5193 | n/a |
| saqa_stgcn | neural | 0.3842 | 0.2641 | 0.2498 | 0.1526 | 0.896 | 0.6021 | 0.5347 | 66791 |
| lstm | neural | 0.3569 | 0.2423 | 0.2498 | 0.1561 | 0.8 | 0.6037 | 0.4985 | 1.60e+05 |
| stgcn_dense | neural | 0.3507 | 0.238 | 0.2518 | 0.1541 | 0.86 | 0.6309 | 0.5086 | 1.94e+05 |
| tcn | neural | 0.3305 | 0.2281 | 0.2521 | 0.1557 | 0.864 | 0.5768 | 0.5491 | 1.23e+05 |
| frame_average | neural | 0.2381 | 0.1633 | 0.2626 | 0.16 | 0.932 | 0.5999 | 0.4971 | 25965 |
| untrained_stgcn | control | -0.0547 | -0.0358 | 0.6319 | 0.4353 | 1 | 1.4094 | 0.5552 | 66791 |
<!-- /table -->

### 2.0 Why the absolute numbers are modest: the model hedges

A concrete, reproducible mechanism rather than a shrug. Across all three seeds
the predictions use about **half** the label range:

| seed | label sd | prediction sd | ratio | prediction range |
|---|---|---|---|---|
| 0 | 0.2005 | 0.0960 | 0.479 | [0.388, 0.892] |
| 7 | 0.1889 | 0.0968 | 0.512 | [0.313, 0.874] |
| 1337 | 0.1961 | 0.1052 | 0.537 | [0.374, 0.918] |

The labels span [0.147, 1.000]; the model never predicts above 0.92 or below
0.31. Its **worst per-combination error is on `clean` sequences (MAE 0.250)** —
it will not commit to a perfect score. This is regression to the mean under a
short schedule, and it is the direct cause of the relative-L2 numbers; it does
*not* affect Spearman, which is scale-free, so the two metrics are telling
different parts of the same story.

It is also evidence that the arms are under-trained rather than saturated: the
selected checkpoint was epoch **6-9 of 10** for every arm, i.e. validation
Spearman was still improving when the schedule ended. The schedule was set by the
compute budget, not by convergence, and that is a limitation of this study rather
than a property of the methods.

### 2.1 The reference approach, given a fair fight

Twelve DTW configurations — three distance functions x banded/unbanded x
canonical/exemplar reference — each with per-action isotonic calibration onto the
quality scale, **selected on validation**:

<!-- table:dtw_sweep -->
| config | val_spearman | test_spearman | test_relative_l2 | test_mae |
|---|---|---|---|---|
| dtw_l2_band0.15_canonical | **0.5972** | 0.4996 | 0.2407 | 0.1452 |
| dtw_l2_unbanded_canonical | 0.5972 | 0.4996 | 0.2407 | 0.1452 |
| dtw_l2_band0.15_exemplar | 0.5499 | 0.3988 | 0.2536 | 0.1549 |
| dtw_l2_unbanded_exemplar | 0.5499 | 0.3988 | 0.2536 | 0.1549 |
| dtw_per_joint_norm_band0.15_exemplar | 0.4765 | 0.3979 | 0.256 | 0.1562 |
| dtw_per_joint_norm_unbanded_exemplar | 0.4765 | 0.3979 | 0.256 | 0.1562 |
| dtw_per_joint_norm_band0.15_canonical | 0.4371 | 0.3964 | 0.2517 | 0.153 |
| dtw_per_joint_norm_unbanded_canonical | 0.4371 | 0.3964 | 0.2517 | 0.153 |
| dtw_velocity_cosine_band0.15_canonical | 0.4021 | 0.3244 | 0.2553 | 0.1538 |
| dtw_velocity_cosine_unbanded_canonical | 0.4021 | 0.3244 | 0.2553 | 0.1538 |
| dtw_velocity_cosine_band0.15_exemplar | 0.3492 | 0.3191 | 0.2622 | 0.1583 |
| dtw_velocity_cosine_unbanded_exemplar | 0.3492 | 0.3191 | 0.2622 | 0.1583 |
<!-- /table -->

The single largest fairness fix was per-action calibration. Distances are not
comparable across action classes, so one global monotone map conflates "which
action" with "how good". Measured on a 300-sequence pilot: **ρ = −0.12 without
it, ρ = +0.44 with it.** Reporting the first number would have been a strawman,
and it is the number a careless implementation produces.

### 2.2 Is the difference significant?

<!-- table:statistics -->
| name_a | mean_a | mean_b | difference | ci_lower | ci_upper | p_value | p_adjusted | effect_size | significant |
|---|---|---|---|---|---|---|---|---|---|
| saqa_stgcn.spearman | 0.3842 | 0.4996 | -0.1154 | -0.2301 | -0.0092 | n/a | n/a | n/a | yes |
| stgcn_dense.spearman | 0.3507 | 0.4996 | -0.1489 | -0.2689 | -0.0373 | n/a | n/a | n/a | yes |
| tcn.spearman | 0.3305 | 0.4996 | -0.1691 | -0.2781 | -0.0627 | n/a | n/a | n/a | yes |
| lstm.spearman | 0.3569 | 0.4996 | -0.1427 | -0.2474 | -0.04 | n/a | n/a | n/a | yes |
| frame_average.spearman | 0.2381 | 0.4996 | -0.2615 | -0.3847 | -0.1367 | n/a | n/a | n/a | yes |
| kinematic_gbr.spearman | 0.5968 | 0.4996 | 0.0973 | -0.0041 | 0.208 | n/a | n/a | n/a | no |
| framewise_reference.spearman | 0.4191 | 0.4996 | -0.0805 | -0.1638 | -0.0049 | n/a | n/a | n/a | yes |
| untrained_stgcn.spearman | -0.0547 | 0.4996 | -0.5543 | -0.708 | -0.3949 | n/a | n/a | n/a | yes |
<!-- /table -->

### 2.2.1 And is it bigger than the noise?

The paired test above answers "is this difference consistent across test
sequences". This one answers the question that actually matters: "is it bigger
than re-rolling the seed". Deltas are against the reference DTW baseline, divided
by the `sqrt(2) * sd` from the three-seed study in §5.

<!-- table:verdicts -->
| method | value | reference_value | delta | noise_scale | ratio_to_noise | verdict |
|---|---|---|---|---|---|---|
| kinematic_gbr | 0.5968 | 0.4996 | 0.0973 | 0.1205 | 0.807 | inside noise |
| framewise_reference | 0.4191 | 0.4996 | -0.0805 | 0.1205 | 0.6681 | inside noise |
| saqa_stgcn | 0.3842 | 0.4996 | -0.1154 | 0.1205 | 0.9573 | inside noise |
| lstm | 0.3569 | 0.4996 | -0.1427 | 0.1205 | 1.184 | suggestive |
| stgcn_dense | 0.3507 | 0.4996 | -0.1489 | 0.1205 | 1.2354 | suggestive |
| tcn | 0.3305 | 0.4996 | -0.1691 | 0.1205 | 1.403 | suggestive |
| frame_average | 0.2381 | 0.4996 | -0.2615 | 0.1205 | 2.1695 | survives |
| untrained_stgcn | -0.0547 | 0.4996 | -0.5543 | 0.1205 | 4.5996 | robust |
<!-- /table -->

**And this retracts the paired test's verdict.** §2.2 reports that every neural
arm is *significantly* worse than the DTW baseline: the paired bootstrap
intervals on the Spearman difference exclude zero. Placed against the run-to-run
scale of 0.1205, that is **not** supportable:

| comparison against `dtw_reference` | delta | ratio to noise | verdict |
|---|---|---|---|
| `saqa_stgcn` (ours) | -0.115 | 0.96x | **inside noise** |
| `kinematic_gbr` | +0.097 | 0.81x | **inside noise** |
| `framewise_reference` | -0.081 | 0.67x | **inside noise** |
| `lstm` / `stgcn_dense` / `tcn` | -0.14 to -0.17 | 1.2-1.4x | suggestive |
| `frame_average` | -0.261 | 2.17x | survives |
| `untrained_stgcn` | -0.554 | 4.60x | robust |

The two tests are not in conflict; they answer different questions. The paired
bootstrap correctly reports that *these particular weights* rank worse than the
DTW baseline consistently across the 250 test sequences. The seed study reports
that re-rolling the seed moves Spearman by 0.12, which is as large as the gap.
**The claim "this method is worse than the reference approach" is about the
method, so the sampling unit is the training run, and there were three.**

So the supportable reading of the whole comparison is narrow:

* **Nothing separates** the DTW baseline, the kinematic GBR, the graph model, the
  LSTM and the temporal CNN at this scale. Every pairwise gap among them is at or
  below the run-to-run scale.
* **Temporal structure matters:** frame-averaging is worse than everything with a
  temporal model, at 2.17x the noise.
* **Training matters:** every trained arm clears the untrained control at 4.6x.

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
| method | gait | lunge | overhead_press | squat | throw |
|---|---|---|---|---|---|
| saqa_stgcn | 0.5567 | 0.3555 | 0.363 | 0.5013 | 0.2595 |
| stgcn_dense | 0.4982 | 0.2481 | 0.3109 | 0.4549 | 0.2817 |
| tcn | 0.0363 | 0.4522 | 0.3506 | 0.5642 | 0.3012 |
| lstm | 0.1172 | 0.4074 | 0.3639 | 0.5539 | 0.3402 |
| frame_average | 0.2574 | 0.1889 | 0.4217 | 0.3864 | 0.4513 |
| kinematic_gbr | 0.5338 | 0.6172 | 0.6209 | 0.7673 | 0.5301 |
| dtw_reference | 0.4222 | 0.5123 | 0.5773 | 0.4735 | 0.4985 |
| framewise_reference | 0.3201 | 0.4092 | 0.515 | 0.594 | 0.3619 |
| untrained_stgcn | -0.1839 | -0.2291 | -0.4349 | -0.0559 | 0.0706 |
<!-- /table -->

## 3. Attribution fidelity

This is what the project is for. The explanation is scored against the exact
cause, and against two controls that a convincing-looking heat map must beat:
a uniform-random attribution, and **the same method applied to an untrained
model** — the parameter-randomisation sanity check of Adebayo et al. (2018).

<!-- table:attribution -->
| attribution | joint_precision | joint_recall | joint_iou | joint_rank_corr | joint_top1_hit | joint_overlap | frame_localisation_error |
|---|---|---|---|---|---|---|---|
| integrated_gradients | 0.5095 | 0.5095 | 0.4038 | -0.2405 | 0 | 0.4752 | 0.0476 |
| occlusion | 0.5282 | 0.5282 | 0.4256 | -0.1738 | 0 | 0.3143 | 0.1163 |
| gradient | 0.5275 | 0.5275 | 0.4181 | -0.106 | 0.0583 | 0.5028 | 0.0505 |
| random | 0.5551 | 0.5551 | 0.4442 | 9.85e-04 | 0.0333 | 0.5455 | 0.0405 |
| integrated_gradients_untrained | 0.6902 | 0.6902 | 0.5639 | 0.2699 | 0.475 | 0.6781 | 0.0561 |
| edge_importance_global | 0.5267 | 0.5267 | 0.4098 | -0.2249 | 0 | 0.5973 | n/a |
<!-- /table -->

`n_joint_iou` is the number of test sequences that contributed: clean sequences
have no cause to attribute and are excluded rather than scored as perfect.

### 3.1 The attributions fail their own sanity check

This is the project's central experiment and it returns a **negative result**,
consistently, across every architecture and every attribution method:

* the trained models' attributions have rank correlation with the ground-truth
  joint importance between **-0.24 and 0.01** — that is, at or *below* chance,
  and often systematically anti-correlated;
* a **uniform-random** attribution scores 0.0010;
* the **same method applied to an untrained network** scores **+0.15 to +0.29**,
  and hits the single most-responsible joint 47.5% of the time against 0.0-19.2%
  for the trained models and 3.3% for random.

An untrained network explains this task's ground truth better than a trained one.
That is precisely the failure mode Adebayo et al. (2018) described: a saliency
map that reflects input geometry rather than anything the model learned. The
control was cheap to include and it is the only reason this is visible; without
it, the trained model's top-k precision of ~0.51 would have looked like a result,
because it *is* the same order as the random baseline's 0.56 and nobody would
have computed the random baseline.

Two secondary observations:

**The DTW baseline's explanation fails differently.** Its per-joint deviation
along the optimal path also anti-correlates overall (-0.12), but it has the best
`top1_hit` of any method (0.333): it finds the single most-affected joint a third
of the time while ranking the rest wrongly. That is consistent with its
mechanism — the alignment minimises whole-body cost, so it sees the largest
deviation clearly and everything below it poorly.

**The model's own attention fails too.** `edge_importance_global`, the learned
adjacency mask, scores -0.22. It is also a *global* quantity and cannot explain a
single sequence, so it is reported separately and never scored as if it could.

**It is not an implementation bug.** Integrated gradients satisfies its own
completeness axiom on these models to within a mean absolute residual of
**8.7e-4** (max 4.9e-3) at 24 steps, against a score range of 0.95
(`results/tables/ig_completeness.csv`). The attributions sum to what they are
supposed to sum to; they are simply pointing at the wrong joints. Three methods
with different failure modes -- a path integral, a forward-pass occlusion, and a
single gradient -- agree, and the occlusion method uses no gradients at all.

**What this does not say.** It does not say integrated gradients is a bad method,
and it does not say the ground truth is wrong — the ground truth is exact by
construction and is concentrated rather than uniform (top-3 joint mass 0.49
against 0.18 for a uniform vector, §1 of `notebooks/01`). It says that *these*
models, trained to this accuracy on this budget, have not learned features whose
gradients point at the true cause. The honest reading is that the attribution
machinery and its validation harness work, and the thing being validated failed.


## 4. Monotonicity and calibrated uncertainty

### 4.1 Does adding a defect ever raise the score?

Severity ladders fix the subject, the action, the defect target and all of the
defect's internal randomness, and vary only the severity — with the sensor model
switched off, so a violation cannot be caused by a noise realisation.

<!-- table:monotonicity -->
| method | violation_rate | ladders_with_any_violation | max_increase | rank_inconsistency |
|---|---|---|---|---|
| saqa_stgcn | 0.1617 | 0.35 | 0.0092 | 0 |
| stgcn_dense | 0.29 | 0.3833 | 0.0207 | 0 |
| tcn | **0.115** | 0.1667 | 0.0093 | 0 |
| lstm | 0.2783 | 0.45 | 0.008 | 0 |
| frame_average | 0.2833 | 0.4 | 0.006 | 0 |
<!-- /table -->

`rank_inconsistency` is a **structural** property of the shared-latent ordinal
head and is 0 for every arm, as the parameterisation guarantees; the
independent-logit variant produces crossings on over half of all inputs
(`test_independent_variant_does_produce_crossings`). `violation_rate` is the
*empirical* property, and the head does not guarantee it.

**And it does not hold.** Between 11.5% and 29.0% of ordered severity pairs are
violations — the model scored a *more degraded* execution *higher* — and between
16.7% and 45.0% of individual executions contain at least one such reversal. The
graph model sits at 16.2% of pairs and 35.0% of ladders.

The violations are small in magnitude (`max_increase` 0.006-0.021, against a
score range of 0.95), so this is hedging near the decision boundary rather than
gross inversion. It is still a real defect for the stated use case: a system that
sometimes rewards a worse execution cannot be put in front of a learner, however
good its rank correlation is.

**This is the difference between a guarantee and a hope, made measurable.** The
ordinal head buys rank consistency — provably, and it is 0 — and buys nothing at
all for degradation monotonicity. `docs/METHOD.md` §4.4 says so in advance; this
table is what makes that statement checkable rather than a hedge written after
the fact.

### 4.2 Intervals

<!-- table:uncertainty -->
| method | coverage | mean_width | crossing_rate | width_ratio_wrong_right | aurc | aurc_oracle | e_aurc | error_auroc |
|---|---|---|---|---|---|---|---|---|
| saqa_stgcn | 0.896 | 0.6021 | 0 | 1.0143 | 0.1422 | 0.068 | 0.0742 | 0.5347 |
| stgcn_dense | 0.86 | 0.6309 | 0 | 0.9995 | 0.1488 | 0.0704 | 0.0784 | 0.5086 |
| tcn | 0.864 | 0.5768 | 0 | 1.0146 | 0.1438 | 0.0726 | 0.0712 | 0.5491 |
| lstm | 0.8 | 0.6037 | 0 | 1.009 | 0.1556 | 0.0742 | 0.0814 | 0.4985 |
| frame_average | 0.932 | 0.5999 | 0 | 0.9977 | 0.1602 | 0.0717 | 0.0885 | 0.4971 |
| untrained_stgcn | 1 | 1.4094 | 0 | 1.0001 | 0.4396 | 0.2629 | 0.1768 | 0.5552 |
| kinematic_gbr | 0.88 | 0.5693 | 0 | 1.0107 | 0.1252 | 0.0571 | 0.0681 | 0.5398 |
| dtw_reference | 0.864 | 0.5396 | 0 | 1 | 0.1485 | 0.0633 | 0.0852 | 0.4744 |
| framewise_reference | 0.868 | 0.5703 | 0 | 1 | 0.1511 | 0.0641 | 0.087 | 0.5193 |
<!-- /table -->

Coverage alone cannot distinguish a useful interval from a constant-width one, so
`width_ratio_wrong_right` is reported next to it: above 1 means the interval
widens on the sequences the model gets wrong, which is the property that makes
abstention work. `e_aurc` is AURC minus the oracle AURC, which separates "the
confidence signal is good" from "the model is accurate".

**Marginal calibration is good. Conditional calibration is worthless.** Those are
two different claims and the table separates them:

* **Good:** the graph model covers 89.6% of labels at a nominal 90%, the closest
  of any arm, and **no method produces a single crossed interval** — the
  non-crossing parameterisation does what it claims. The untrained control's
  interval is 2.3x wider (1.409 against 0.602), so the width does at least track
  gross incompetence.
* **Negative, and it is robust:** `width_ratio_wrong_right` is 0.998-1.015 for
  **every** method, and error-detection AUROC is 0.474-0.555 — chance. The
  intervals carry essentially no information about *which* sequences the model
  gets wrong. `e_aurc` (0.068-0.089) is comparable to `aurc_oracle` itself
  (0.057-0.074), meaning the achievable improvement from a perfect confidence
  signal is roughly as large as what the model achieves in total.

**So the risk-coverage story does not work here**, and that applies to the
conformal baselines exactly as much as to the quantile head — the DTW and
framewise baselines have constant-width conformal intervals and score a ratio of
exactly 1.000 by construction. The quantile head is *allowed* to vary its width
with the input and essentially declines to. A system built on this should not
offer selective abstention; it can offer a calibrated marginal interval and
nothing more.

## 5. Seed variance — read this before any ablation

Three training seeds, identical configuration. The seed changes initialisation,
batch order **and** the split, which is the conservative choice: it measures the
variation a reader reproducing the pipeline would see.

<!-- table:seeds -->
| metric | mean | sd | min | max | range | noise_scale | n_runs |
|---|---|---|---|---|---|---|---|
| spearman | 0.3585 | 0.0852 | 0.2634 | 0.428 | 0.1645 | 0.1205 | 3 |
| kendall_tau | 0.2481 | 0.057 | 0.1849 | 0.2955 | 0.1106 | 0.0806 | 3 |
| relative_l2 | 0.2456 | 0.0076 | 0.2368 | 0.2502 | 0.0134 | 0.0108 | 3 |
| mae | 0.1466 | 0.0056 | 0.1415 | 0.1526 | 0.0111 | 0.008 | 3 |
| coverage | 0.9134 | 0.0232 | 0.896 | 0.9398 | 0.0438 | 0.0328 | 3 |
| mean_width | 0.5997 | 0.0127 | 0.5859 | 0.6109 | 0.025 | 0.018 | 3 |
| aurc | 0.1411 | 0.0136 | 0.1269 | 0.154 | 0.0271 | 0.0192 | 3 |
| error_auroc | 0.518 | 0.0365 | 0.4761 | 0.5431 | 0.067 | 0.0516 | 3 |
<!-- /table -->

Two independent runs differ with standard deviation `sqrt(2) * sd`. Any claimed
improvement smaller than that column is **noise**, and is labelled as such below.

## 6. Ablations

One variable at a time, one shared dataset, and a **shorter schedule** (8 epochs
against the headline 10) so that nine variants fit in the budget. The table
carries its own `full` reference row, so it is internally consistent; do not
compare its absolute numbers against §2.

<!-- table:ablation -->
| variant | change | spearman | spearman_delta | spearman_ratio_to_noise | spearman_verdict | params |
|---|---|---|---|---|---|---|
| full | - | 0.3599 | 0 | 0 | inside noise | 66791 |
| partitions_uniform | model.partitions=uniform | 0.4588 | 0.0989 | 0.8209 | inside noise | 43135 |
| partitions_identity | model.partitions=identity | 0.4022 | 0.0423 | 0.3512 | inside noise | 43135 |
| no_edge_importance | model.edge_importance=False | 0.3565 | -0.0034 | 0.028 | inside noise | 63323 |
| dense_temporal | model.separable=False | 0.351 | -0.0089 | 0.0735 | inside noise | 1.94e+05 |
| temporal_kernel_3 | model.temporal_kernel=3 | 0.3828 | 0.0229 | 0.1904 | inside noise | 65399 |
| head_regression | model.head=regression | 0.3903 | 0.0304 | 0.2525 | inside noise | 66782 |
| norm_group | model.norm=group | 0.0746 | -0.2853 | 2.3676 | survives | 66791 |
| uncertainty_heteroscedastic | model.uncertainty=heteroscedastic | 0.3533 | -0.0066 | 0.0548 | inside noise | 66694 |
<!-- /table -->

Every delta is divided by the §5 noise scale before it is interpreted. A verdict
of `inside noise` means the ablation did not measure anything at this scale, not
that the component does nothing.

### 6.1 One component matters, and it is the normalisation

**`norm_group` is the only variant that survives**: Spearman 0.075 against 0.360
for the default, a delta of −0.285 at **2.37x the run-to-run scale**. This
reproduces, like-for-like at the shipped configuration, the finding recorded in
`docs/METHOD.md` §4.3 — GroupNorm removes the per-sample channel scale that the
global average-pooling readout reads, and the model becomes nearly unable to
learn. It is the single largest effect anywhere in this repository.

### 6.2 Everything else is inside the noise, including the graph

That has to be said plainly, because it undercuts the architecture this project
proposes:

* **`partitions_identity`** — self-loops only, *no message passing between joints
  at all* — scores **0.402 against the full model's 0.360**, using 43k parameters
  instead of 67k. At 0.35x the noise scale this is not evidence that removing the
  graph *helps*; it is decisive evidence that at this scale the graph is not
  measurably helping either.
* **`partitions_uniform`** (one undirected partition instead of three) scores
  0.459, 0.82x the noise. The centripetal/centrifugal split of Yan et al. (2018)
  buys nothing measurable here and costs 55% more parameters.
* **`dense_temporal`** costs 2.9x the parameters for a delta of −0.009 (0.07x
  noise). This is the one *supportive* reading available: the separable temporal
  stage is 2.9x cheaper at no measurable cost in accuracy — though "no measurable
  cost" at this noise level is a weak statement, and it is the same statement in
  both directions.
* **`head_regression`** scores 0.390 against the ordinal head's 0.360 (0.25x
  noise). The ordinal head is not bought for accuracy; it is bought for the rank
  consistency in §4.1, which it does deliver.
* **`temporal_kernel_3`**, **`no_edge_importance`** and
  **`uncertainty_heteroscedastic`** are all ≤0.26x the noise.

**The honest summary of this table is that one variable — the normalisation —
accounts for everything measurable, and the graph structure that motivates the
architecture does not show up above the noise floor at this scale.** With one run
per variant and a noise scale of 0.12, this experiment can only detect effects
larger than about 0.24 Spearman, and only one is.

## 7. Splits, and the leakage this repository measures rather than assumes

<!-- table:splits -->
| split | spearman | kendall_tau | relative_l2 | mae | coverage | n_train | n_test |
|---|---|---|---|---|---|---|---|
| random | 0.3333 | 0.2344 | 0.266 | 0.1556 | 0.832 | 638 | 250 |
| subject | 0.3842 | 0.2641 | 0.2498 | 0.1526 | 0.896 | 638 | 250 |
| combination | 0.1679 | 0.1149 | 0.5936 | 0.2648 | 0.5101 | 682 | 198 |
<!-- /table -->

### 7.1 The subject-leakage gap did not appear

The expectation was that `random` — which puts the same subject, with the same
limb lengths and style factor, on both sides — would score *higher* than
`subject`. **It does not.** `random` scores 0.333 against `subject`'s 0.384, a
gap of −0.051, which is **0.42x the run-to-run scale and therefore inside the
noise.**

So this repository does **not** demonstrate subject leakage, and the expectation
stated when the three regimes were designed is not supported. The most likely
explanation is that the canonicalisation already removes most of what makes a
subject identifiable — translation is subtracted and every sequence is divided by
its own body-height estimate (`docs/METHOD.md` §3, and
`test_canonicalise_scales_out_subject_size` measures the residual at <0.02 body
heights between a 0.85x and a 1.20x subject). If so, the nuisance factor was
handled by the representation rather than by the split. That is a plausible
account, not a measured one, and it is not claimed as a finding.

### 7.2 Cross-degradation generalisation fails, and the calibration fails with it

The `combination` regime — where every multi-defect sequence containing a
compensatory pattern is held out, so the model must score an *interaction* it has
never seen — is a different story, and this one is robust:

| metric | `subject` | `combination` | delta | ratio to noise | verdict |
|---|---|---|---|---|---|
| Spearman | 0.3842 | 0.1679 | −0.216 | 1.79x | suggestive |
| relative L2 | 0.2498 | 0.5936 | +0.344 | 31.9x | **robust** |
| coverage (nominal 0.90) | 0.8960 | 0.5101 | −0.386 | 11.8x | **robust** |

**The most important number in this table is the coverage.** A predictive
interval that covers 89.6% of labels in-distribution covers **51.0%** when the
defect combination is unseen — a nominally 90% interval delivering barely half of
that. The interval is calibrated to the training distribution and carries no
warning at all when it leaves it.

The failure is structured, not random. Coverage falls monotonically with the
number of simultaneous unseen defects (`results/runs/split_combination/per_item.csv`):

| held-out combination | n | coverage | MAE |
|---|---|---|---|
| `compensation+instability` | 28 | 0.714 | 0.176 |
| `asymmetry+compensation` | 23 | 0.565 | 0.219 |
| `compensation+instability+rom` | 7 | 0.429 | 0.312 |
| `asymmetry+compensation+jerk` | 10 | 0.300 | 0.401 |
| `asymmetry+compensation+rom` | 10 | **0.000** | 0.487 |

Ten sequences where the nominally 90% interval contains the true score **zero
times**, and the interval width does not grow to compensate. The model's error
grows with the number of co-occurring unseen defects and its stated uncertainty
does not move — which is the worst possible combination for a system meant to
know when to decline.

Taken with §4.2 — where the interval also fails to indicate *which* individual
sequences are wrong — the uncertainty story of this repository is: marginal
coverage is accurate exactly when it is least needed, and uninformative or
actively misleading when it matters. That is a negative result about the
formulation, not about this implementation of it: the conformal baselines share
the first failure by construction.

## 8. Label efficiency

Reference-free quality assessment needs a human-scored corpus, and the scoring is
the expensive part. This is the practically important efficiency axis.

<!-- table:data_efficiency -->
| n_train | frame_average | kinematic_gbr | saqa_stgcn | tcn |
|---|---|---|---|---|
| 100 | 0.0253 | 0.2624 | 0.0113 | 0.0984 |
| 250 | 0.0472 | 0.4075 | 0.3092 | 0.2273 |
| 500 | 0.1887 | 0.6173 | 0.3658 | 0.303 |
| 638 | 0.2381 | 0.5968 | 0.3842 | 0.3305 |
<!-- /table -->

**The handcrafted-feature baseline dominates at every budget**, and the gap is
widest where it matters most: at 100 labelled sequences it reaches Spearman
0.262 while the graph model manages **0.011** -- indistinguishable from nothing.
The neural arms need roughly 250 labels to reach what the feature baseline gets
from 100, and at 638 they have still not caught it.

That ordering is not surprising and it is not a criticism of learned features in
general. The kinematic feature set was written *with the degradation taxonomy in
hand* (`docs/METHOD.md` section 5): joint-angle ranges for range-of-motion
defects, left-right differences for asymmetry, dimensionless jerk for smoothness,
pelvis path length for stability. It is being handed the vocabulary of the label
function, and forty-odd such features with a gradient-boosted regressor is a
strong estimator at these sample sizes. The learned models must discover the same
structure from 100-638 examples, and at this budget they do not.

**What the curve says about the future is limited, and worth stating rather than
extrapolating.** All four methods are still rising at 638, so none has plateaued;
that is consistent with the neural arms closing the gap given more labels, and
equally consistent with them not. This study cannot distinguish the two. The
honest statement is: **at every label budget this study can afford, handcrafted
kinematics beat the graph model.**


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

## 10. Retractions and negative results

Collected in one place. Each is stated at the point of the claim as well.

### 10.1 Retracted: "the neural arms are significantly worse than DTW"

§2.2's paired bootstrap gives Spearman-difference intervals that exclude zero for
every neural arm against `dtw_reference`. §2.2.1 shows the run-to-run scale is
0.1205 and the `saqa_stgcn` gap is 0.115 — **0.96x the noise**. The paired test
was computed correctly and answers a question about *two sets of weights*; the
claim was about *methods*, whose sampling unit is the training run. The table is
left intact and the retraction sits next to it.

The same correction removes the opposite claim: the kinematic GBR's +0.097
advantage over DTW is 0.81x the noise and is **not** evidence that handcrafted
features beat the reference approach either.

### 10.2 Negative: the attribution fidelity experiment failed

The project's central claim was that per-joint, per-phase attribution could be
*validated*. The harness works; the attributions do not. Trained-model
attributions score at or below a random baseline, and below an untrained network,
on every architecture and every method (§3.1). This is reported as the primary
finding rather than buried, because a validation harness that detects a failure
is worth more than one that never could have.

### 10.3 Negative: the ordinal head buys rank consistency and nothing else

Rank inconsistency is exactly 0, as designed and as `docs/METHOD.md` §4.4 states
in advance. Degradation monotonicity — the property a *user* would care about —
is violated on 11.5-29.0% of ordered severity pairs. The design note was written
before the measurement and is unchanged by it; what changed is that the property
is now quantified instead of hoped for.

### 10.4 Negative: predictive intervals do not support abstention

Coverage is good and crossings are zero, but `width_ratio_wrong_right` is
0.998-1.015 and error-detection AUROC is 0.474-0.555 across **every** method,
including the conformal baselines. The risk-coverage machinery is implemented and
tested; on this task it has nothing to rank.

### 10.5 Weakened: the efficiency claim against DTW is asymptotic, with a crossover

DTW's fitted cost exponent is 1.98 and the model's MAC count is linear in ``T``
(asserted by test). But at the 48 frames used throughout these experiments, the
DTW baseline is **cheaper in wall-clock** than the graph model. The scaling
advantage is real and it only pays off at longer sequences. The parameter and MAC
reductions against the reference-scale ST-GCN (45x, 52x) are unaffected — those
are architecture-to-architecture, not approach-to-approach.

### 10.6 Design errors found and fixed during the build

Kept because they are the kind of thing that silently invalidates results:

* **GroupNorm made the model unable to fit its own training set** (train Spearman
  0.147 vs 0.855). It removes the per-sample channel scale that global average
  pooling reads. `docs/METHOD.md` §4.3.
* **A generic `model.channels` in the config silently overrode the named
  architectures**, collapsing three comparison arms into one identical model.
  Visible only because their attribution scores matched to six decimals. Guarded
  by `SHAPE_OWNING_ARCHITECTURES` and a regression test.
* **The DTW baseline scored ρ = −0.12 before per-action calibration** and +0.44
  after. Shipping the first number would have been a strawman comparison.
* **The Sakoe-Chiba band was anchored off-diagonal**, making the distance
  asymmetric for equal-length sequences. Caught by a symmetry test.
* **`velocity_cosine` gave a non-zero self-distance** because the first frame's
  velocity is zero by construction; two stationary joints were being scored as
  maximally dissimilar.
* **A full-repetition phase window still applied an edge taper**, which shifted a
  range-of-motion defect's temporal mean and turned it partly into a posture
  defect. Caught by a mean-preservation test.

### 10.7 What would settle the open questions

Five to ten seeds per configuration (tens of CPU-hours here, under an hour on a
GPU), the reference-scale architectures actually trained, sequences at 192-300
frames, and a real judge-scored dataset for the one question this design cannot
answer at all. `notebooks/05_colab_full_scale.ipynb` targets exactly that.
