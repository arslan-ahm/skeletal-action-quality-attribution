# Method

## 1. The problem with a similarity score

The reference approach — the one this repository is built to replace — extracts
3D pose from two videos, aligns the sequences with dynamic time warping
(Sakoe & Chiba, 1978), and reports how similar they are. It has three structural
problems, and none of them is fixed by a better distance function.

**A single number says nothing actionable.** "0.83" does not tell a learner to
push their knees out, or that they rushed the concentric phase. The information
a coach gives is *where* and *when*, and a scalar cannot carry it.

**The number is not calibrated to anything.** Is 0.83 good? The question has no
answer without a population to compare against, and the similarity score has no
units. It is not a probability, not a rank, not a distance in any interpretable
space.

**It requires a reference performance.** For a scripted drill you might have
one. For free-form assessment — an athlete's own technique, a patient's gait, a
movement nobody has demonstrated — there is no reference, and the entire
formulation is unavailable. This is the structural limitation: the method cannot
be *applied*, not merely that it performs worse.

## 2. What this repository does instead

**Reference-free, calibrated action-quality regression with per-joint,
per-phase causal attribution.** The output is three things:

| output | mechanism | what it answers |
|---|---|---|
| a quality score | ordinal head over quality bands | how good was this? |
| a predictive interval | non-crossing quantile head | how sure are we? |
| a `(joint, phase)` attribution map | integrated gradients / occlusion | what should change? |

and — the part that is usually skipped — the attribution is **scored against
exact ground truth**, because the data generator knows which joints it degraded.

## 3. The data, and why it is synthetic

A synthetic generator is usually a compromise. Here it is the *enabling
condition* for the central experiment, and that is worth being precise about.

To measure whether an explanation is correct you need to know the true cause.
No human-annotated action-quality dataset carries that: MTL-AQA and FineDiving
have judge scores, not "the left knee at 40% of the dive cost 0.3 points".
Without ground truth, attribution work can only show heat maps and assert that
they look reasonable — which Adebayo et al. (2018) showed is not evidence of
anything, since saliency maps that are *independent of the model's parameters*
can look equally reasonable.

So the generator is built so the cause is known by construction:

1. **Five action classes** (`squat`, `overhead_press`, `lunge`, `gait`, `throw`)
   are defined as parametric joint-angle trajectories over a phase variable.
2. **Forward kinematics** maps angles onto a fixed 17-joint tree, then grounds
   the skeleton so the lower ankle sits on the floor. Grounding matters: without
   it, bending the knees leaves the pelvis pinned in space and a squat becomes a
   levitation.
3. **Degradations** are named, severity-parameterised edits to those
   trajectories.
4. **The quality label** is an explicit function of the applied severities.
5. **The attribution ground truth** is obtained by running steps 1–3 twice —
   with and without each defect — and measuring what moved.

### 3.1 The degradation taxonomy

Six kinds, each with a per-unit-severity quality cost `w_k`:

| kind | mechanism | `w_k` |
|---|---|---|
| `rom` | reduced range of motion at a DOF **and its mirror** (bilateral) | 0.40 |
| `asymmetry` | the same reduction on **one side only** | 0.35 |
| `tempo` | monotone reparameterisation of time (rushing / dragging) | 0.25 |
| `jerk` | band-limited high-frequency oscillation added to a DOF | 0.20 |
| `instability` | low-frequency sway added to the pelvis trajectory | 0.30 |
| `compensation` | a DOF loses range **and a second DOF gains range to hide it** | 0.45 |

Three design decisions in that table are load-bearing.

**`rom` is bilateral and `asymmetry` is unilateral.** If both were one-sided
they would be *mechanically identical* while carrying different costs — the
label would be unlearnable, and silently so. A generator can produce an
impossible task without any error surfacing, which is the most dangerous failure
mode a synthetic benchmark has.
→ `test_rom_is_bilateral_and_asymmetry_is_not`

**`compensation` is the most expensive.** That is the coaching reality: a shallow
squat hidden behind lumbar flexion is worse than a shallow squat, not better. It
is also the hardest case for attribution, because the *observable* change is at
the compensating joint while the *cause* is at the restricted one.

**Degradations shrink a DOF toward its temporal mean**, not toward zero.
Shrinking toward zero would change the posture's offset as well as its range,
which is a different defect.

### 3.2 The quality label

```
q = 1 - clip( sum_d  w_{k(d)} * s_d ,  0 , 0.95 )
```

with `s_d` the severity of defect `d`. Linear in each severity, hence **monotone
non-increasing in each by construction** — which is exactly what makes the
model-side monotonicity test unambiguous. The floor at `q = 0.05` stops the
worst decile piling up on a hard zero, which would inflate rank correlation.

**The limitation, plainly.** `w_k` is a modelling choice. It is not a measured
human judgement, and no amount of care in choosing it makes it one. This
validates the *mechanism* — can a model recover a known score function and a
known cause? — and says nothing about agreement with a real judge. The optional
MTL-AQA / FineDiving path exists precisely because that is a different question.

### 3.3 Attribution ground truth

For each defect `d`, forward kinematics runs with the full defect set and again
with `d` removed. The per-joint displacement between the two canonicalised
sequences is *what that defect did to the movement*, weighted by `d`'s
contribution to the quality loss:

```
truth_joint[v] = sum_d  (w_d * s_d) * mean_t || x_full[t,v] - x_without_d[t,v] ||
```

normalised to sum to 1. Three choices:

**Leave-one-out, not leave-one-in.** This measures each defect's marginal effect
*in the presence of the others*, which is the quantity an attribution method
applied to the full sequence could possibly recover. Leave-one-in would measure
an effect in a context the model never saw.

**The counterfactuals share the defects' randomness.** The RNG is re-seeded
identically for every leave-one-out call, so the jerk waveform, the sway
realisation and the tempo direction are unchanged. Otherwise the counterfactual
would differ by noise as well as by the removed defect and the ground truth
would be garbage.

**A clean sequence gets an all-zero truth, not a uniform one.** A clean sequence
has no cause to attribute. A uniform vector would silently reward a model that
always says "everything", and every fidelity metric returns `NaN` there rather
than a flattering 1.0.

Two ground-truth views are reported and they differ:

* **declared** — the joints whose Euler angles were literally edited;
* **displacement** — every joint the edit actually moved through the kinematic
  chain and floor contact.

A restricted knee moves the pelvis, because the body is grounded. The
displacement view is the honest target; the declared view is reported so a
reader can see the gap.

### 3.4 The sensor model is not a defect

Jitter (AR(1), ρ = 0.8), i.i.d. noise, and short joint dropouts filled by linear
interpolation are applied **after** the quality label and the attribution ground
truth are computed. Sensor noise is a nuisance the model must survive, not a
fault it should penalise. Conflating the two is the easiest way to build a
benchmark that rewards the wrong thing.
→ `test_sensor_model_perturbs_but_does_not_relabel`

## 4. The model

### 4.1 Spatio-temporal graph convolution, hand-rolled

Following Yan et al. (2018). For input `x` of shape `(N, C, T, V)`:

```
y[n,c,t,i] = sum_k sum_j  Â[k,i,j] * (W_k x)[n,c,t,j]
```

with `Â[k]` a row-normalised adjacency partition. Three partitions
(*spatial configuration partitioning*): identity (self), **centripetal**
(child → parent, toward the pelvis) and **centrifugal** (parent → child).

Row normalisation rather than the symmetric `D^{-1/2} A D^{-1/2}` keeps each
partition an *average* over neighbours, so a high-degree joint (thorax, pelvis)
does not shout louder than a leaf (wrist, ankle).

**Why no `torch-geometric` or `mmaction2`.** The skeleton graph has 17 nodes and
a fixed adjacency, so message passing is one `einsum` against a `(K, V, V)`
constant. Scatter-gather machinery designed for million-node dynamic graphs is
strictly slower here than a dense matmul, and both libraries are heavy and
version-brittle. The whole convolution is fifteen lines and is verified against
a hand-computed reference in
`test_graph_conv_matches_hand_computed_message_passing`. That is a deliberate
engineering choice, not an omission.

### 4.2 Separable temporal convolution

The reference ST-GCN block applies a dense `(kt, 1)` temporal convolution costing
`C_out² · kt` parameters. Here the temporal stage is **depthwise**
(`groups=C_out`, `C_out · kt` parameters) followed by a `1×1` pointwise mix
(`C_out²`) — the MobileNet/Xception factorisation applied along time. At
`kt = 9` this removes roughly a factor of `kt` from the temporal stage.

`stgcn_dense` is the *identical* architecture with `separable=False`, so the
ablation isolates separability from width and depth. **Measured:** it costs 2.9x
the parameters for a Spearman delta of −0.009, which is 0.07x the run-to-run
noise scale — the separable stage is much cheaper at no cost this study can
detect. It is also the **trained**
stand-in for the dense-temporal-convolution design of the reference lineage:
2.9x the parameters of ours, 2.6x the MACs, same graph, same head, same
schedule, same data, same seed.

Two genuinely reference-scale models — `stgcn_reference` (0.76M) and
`stgcn_reference_large` (3.02M, the nine-block 64/128/256 stack of Yan et al.)
— are **benchmarked for parameters, MACs and latency but not trained here.**
That is the efficiency claim and it is measured; the accuracy claim is not made,
because a single training run of either exceeds this project's 20-minute per-run
budget on a shared four-core CPU by a wide margin, and a shortened schedule
would produce a number about the schedule rather than about the architecture.
`notebooks/05_colab_full_scale.ipynb` trains them on a GPU.

### 4.3 BatchNorm, and why it is not a preference

The first version of this model used `GroupNorm` everywhere, reasoning that
batch statistics are noisy at batch size 32 and that a batch-independent norm
makes a batch-1 latency benchmark mean the same thing as training.

**That model could not fit its own training set** — train Spearman 0.147 against
0.855 for the BatchNorm version, on identical data, schedule and seed (600
sequences, 64 frames, 25 epochs, temporal CNN trunk). The `norm_group` row of the
ablation table in `docs/RESULTS.md` repeats the comparison like-for-like at the
shipped configuration.

The cause is an interaction with the readout. GroupNorm normalises each sample
over `(channel group, T, V)`, removing that sample's per-channel scale at every
layer. The readout is a global average pool over `(T, V)` — which reads
precisely that scale. BatchNorm normalises over `(N, T, V)` per channel, so
*relative* differences between samples survive to the pool. Yan et al. use
BatchNorm; this is why, and it is a harder constraint than a stability
preference. Both are kept and both appear in the ablation.

### 4.4 The ordinal head, and what it does *not* guarantee

Quality is ordinal. A judge decides an execution is better than one band and
worse than another. Following CORAL (Cao et al., 2020), `K` binary sub-problems
"is the score above band `k`?" share **one** scalar latent `z` and differ only in
an ordered bias:

```
P(y > k | x) = sigmoid( z(x) - b_k ),     b_1 < b_2 < ... < b_K
```

The thresholds are parameterised as `b_1` plus a cumulative sum of
`softplus(δ_i)`, so the ordering holds for **every value the optimiser can
reach**. The common implementation stores `K` free biases and hopes they stay
ordered; they cross during training.
→ `test_ordinal_thresholds_stay_ascending_under_arbitrary_parameters`

**What this guarantees:** the predicted survival function is non-increasing in
`k`, so it is a valid ordered CDF. Rank inconsistency is exactly 0. A head with
`K` independent linear outputs is not — the shipped `ordinal_independent` variant
produces crossings on over half of all inputs, which is what makes "structural
guarantee" a measured claim rather than a description.

**What this does not guarantee, and the documentation says so:** it does *not*
guarantee that adding a degradation lowers the predicted score. That would
require monotonicity in the *input*, and no pooling-plus-graph-convolution
architecture provides it. The score is a strictly increasing function of `z`, so
degradation monotonicity reduces entirely to whether `z` decreases when a defect
is added — an empirical question about learned features. This repository
**measures** the violation rate on severity ladders instead of asserting the
property. A structural guarantee you have tested is worth more than one you
asserted; a guarantee you have *not* tested is worth nothing.

The training target is a **soft** level indicator (linear interpolation across
the band edge it straddles) rather than a hard one, because a hard indicator
discards the within-band position of the target — a tenth of the whole scale
here.

### 4.5 Uncertainty

**Quantile head (default).** Pinball loss (Koenker & Bassett, 1978) at levels
`(0.05, 0.5, 0.95)`. Quantiles are predicted as a base plus a cumulative sum of
`softplus` increments, so `q₁ ≤ q₂ ≤ q₃` for every parameter value — a crossed
interval has negative width, which is not an interval.
→ `test_quantiles_never_cross_for_any_parameters`

**Heteroscedastic head (ablation).** Gaussian mean/log-variance with the
Gaussian NLL. It assumes symmetry, which is wrong near the top of a bounded
score scale.

The interval is deliberately **not** derived from the ordinal distribution: a
discretised ordinal CDF has a resolution floor of one band (0.095 of the scale
here), which would make width uninformative below that.

The uncertainty loss is weighted at 0.5 so the uncertainty head cannot steer the
shared trunk hard enough to move the point estimate. Otherwise the interval is
bought with accuracy and the comparison against a point-only model stops being
about uncertainty.

### 4.6 Attribution

Three methods, because they fail differently:

**Integrated gradients** (Sundararajan et al., 2017). The score is **negated**
so a positive attribution means "cost quality" — the direction a coach needs.
Getting this backwards produces a plausible-looking map that names the joints
doing the *work* rather than the joints doing it *wrong*. The Riemann sum uses
the **midpoint** rule, not the left endpoint, which systematically
under-integrates and breaks completeness by several percent at small step
counts. Completeness — `sum(IG) = f(baseline) - f(x)` — is checked, not cited.

**The baseline choice matters and is stated.** The obvious all-zeros baseline is
a body collapsed to the origin, which is not a *movement*; IG then explains "why
is this a skeleton" and the map is dominated by whichever joints sit furthest
from the pelvis. Instead each joint is frozen at its own temporal mean: the pose
is kept, the *motion* is removed, and motion is what quality is a function of.

**Occlusion of joint groups.** The same intervention (replace a group's motion in
a time window by its temporal mean), so a disagreement between the two methods is
about the method and not about the counterfactual. Groups rather than single
joints: the skeleton is a chain, and occluding one joint while its neighbours
stay put creates an anatomically impossible pose far outside the training
distribution — the model's response to that is extrapolation, not attribution.

**Input × gradient.** The single-pass control. If it matches integrated
gradients, the extra 24 forward-backward passes bought nothing, and that is worth
knowing.

Reduction to per-joint and per-phase vectors takes the **positive part** before
summing. A joint with a large positive attribution in one phase and a large
negative one in another has a real, localised cost, and a signed sum would cancel
it to zero.

### 4.7 Fidelity metrics, and the controls

* **Top-k precision / recall / F1 / IoU**, with `k` derived from how concentrated
  the *truth* is (the number of joints carrying 80% of the mass). A fixed `k`
  could be tuned into looking good.
* **Rank correlation and histogram overlap** over the whole vector.
* **Temporal localisation error** — distance between the attribution's and the
  truth's temporal centres of mass, in units of sequence length.
* **Two controls that must be beaten**: a uniform-random attribution, and the
  *same method applied to an untrained model with the same architecture* — the
  parameter-randomisation sanity check of Adebayo et al. (2018). An attribution
  that does not clear both is reflecting input statistics, not explaining a
  model. Where this repository's attributions fail that test, it is reported as
  a negative result.

## 5. The baselines

**DTW to a reference (the reference approach).** Given every advantage:

* three distance functions — raw L2, per-joint-normalised L2 (each joint divided
  by its own spread in the reference, so a wrist does not drown a pelvis), and
  velocity cosine;
* banded (Sakoe-Chiba, 0.15) and unbanded warping — an unbounded warp can absorb
  the very tempo defect the score should penalise, so unbanded is not
  automatically stronger;
* a **canonical** reference (the generator's ideal) or an **exemplar** reference
  (the medoid clean training sequence for that action — a real performance by a
  real subject);
* **per-action isotonic calibration** of distance onto the quality scale, fitted
  on training data only.

The configuration is chosen on **validation** Spearman, never on test.

Per-action calibration is the single largest fairness fix. Distances are not
comparable across action classes — a throw and a gait cycle have different
spatial extents — so one global monotone map conflates "which action" with "how
good". Without it the baseline scored ρ = −0.12; with it, +0.44 on the same data.
Reporting the first number would have been a strawman.

Its explanation is per-joint deviation along the DTW-optimal path — the best
that formulation can produce, and structurally weak for a reason: the alignment
minimises *whole-body* cost, so a defect that DTW absorbed into the warp leaves
no per-joint trace.

**Handcrafted kinematics + gradient boosting.** Joint-angle ranges, left-right
symmetry, jerk and dimensionless jerk, pelvis path length and sway, energy
centroid and spread. The feature set is deliberately aligned with the defect
taxonomy — it is handed the *vocabulary* of the degradations. That makes it a
strong and slightly favoured baseline, and if a learned model cannot beat
features designed with knowledge of the label function, that is the finding.

**Graph-free neural controls.** `tcn` (dilated temporal convolutions over
flattened joints — keeps time, removes the graph), `lstm` (same, recurrent), and
`frame_average` (mean and per-joint temporal std, then an MLP — removes temporal
order entirely). The std is included so it is a "no temporal order" baseline
rather than a "no motion" one.

**An untrained network.** Not a strawman: a random projection of movement
amplitude already correlates with defect severity, so this arm scores a
non-trivial Spearman. Any trained model that does not clear it has learned
nothing the architecture and input statistics did not already provide.

## 6. Splits

Three regimes, all reported:

* `random` — **leaks**. The same subject and defect combination appear on both
  sides. Included so the leakage can be measured rather than asserted.
* `subject` — whole subjects held out (default).
* `combination` — every multi-defect sequence whose kind set contains
  `compensation` goes to test. Compensation alone is seen in training;
  compensation *interacting* with another defect is not.

## 7. References

* Yan, Xiong & Lin (2018). *Spatial Temporal Graph Convolutional Networks for
  Skeleton-Based Action Recognition.* AAAI.
* Shi, Zhang, Cheng & Lu (2019). *Two-Stream Adaptive Graph Convolutional
  Networks for Skeleton-Based Action Recognition.* CVPR.
* Parmar & Morris (2019). *What and How Well You Performed? A Multitask Learning
  Approach to Action Quality Assessment.* CVPR. (MTL-AQA)
* Parmar & Morris (2019). *Action Quality Assessment Across Multiple Actions.*
  WACV.
* Tang et al. (2020). *Uncertainty-Aware Score Distribution Learning for Action
  Quality Assessment.* CVPR.
* Xu, Rao, Yu, Chen, Zhou & Lu (2022). *FineDiving: A Fine-grained Dataset for
  Procedure-aware Action Quality Assessment.* CVPR.
* Sundararajan, Taly & Yan (2017). *Axiomatic Attribution for Deep Networks.*
  ICML.
* Adebayo, Gilmer, Muelly, Goodfellow, Hardt & Kim (2018). *Sanity Checks for
  Saliency Maps.* NeurIPS.
* Sakoe & Chiba (1978). *Dynamic Programming Algorithm Optimization for Spoken
  Word Recognition.* IEEE TASSP.
* Koenker & Bassett (1978). *Regression Quantiles.* Econometrica.
* Cao, Mirjalili & Raschka (2020). *Rank Consistent Ordinal Regression for Neural
  Networks with Application to Age Estimation.* Pattern Recognition Letters.
* Shahroudy, Liu, Ng & Wang (2016). *NTU RGB+D: A Large Scale Dataset for 3D
  Human Activity Analysis.* CVPR.
* Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018). *Distribution-Free
  Predictive Inference for Regression.* JASA. (split conformal)
* Geifman & El-Yaniv (2017). *Selective Classification for Deep Neural
  Networks.* NeurIPS. (risk-coverage)
