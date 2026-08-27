![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13%20cpu-red)
![Tests](https://img.shields.io/badge/tests-482%20passing-brightgreen)
![Params](https://img.shields.io/badge/params-0.067M-orange)
![License](https://img.shields.io/badge/License-MIT-green)

# What Should I Change? — Skeletal Action-Quality Attribution

**Reference-free, calibrated action-quality regression with per-joint,
per-phase attribution that is *validated against ground truth*.**

The approach this replaces takes two videos, extracts 3D pose, aligns them with
DTW, and reports one similarity number. This repository argues that the number is
the wrong output — it is not actionable, not calibrated, and it cannot be
produced at all without a reference performance — and replaces it with a quality
estimate, a predictive interval, and a map of *which joints in which movement
phase cost the quality*. Then it does the thing attribution work usually skips:
**it measures whether that map is correct.**

> **Result, up front — and it is not the flattering one.**
>
> **The measurement machinery works and the mechanism guarantees hold.**
> Predicted quantiles never cross (0 crossings, all methods), the ordinal head's
> rank-inconsistency rate is exactly 0, coverage is 89.6% against a nominal 90%,
> and every trained arm clears an untrained-network control by 4.6x the
> run-to-run noise.
>
> **The central attribution experiment returns a negative result.** Integrated
> gradients, occlusion and input-gradient on the trained models all score *at or
> below chance* against the exact ground truth (rank correlation −0.24 to +0.01,
> versus 0.001 for a uniform-random attribution) — while the **same method
> applied to an untrained network scores +0.27 and finds the single most
> responsible joint 47.5% of the time against 0.0% for the trained model.** That
> is the Adebayo et al. (2018) failure mode, and it is only visible because the
> control was built in.
>
> **No method ranking survives the seed study.** A three-seed study puts the
> run-to-run scale of a Spearman difference at 0.1205. Every gap among the DTW
> baseline (0.500), the kinematic GBR (0.597), this repository's graph model
> (0.384), the LSTM and the temporal CNN is at or below that. What does survive:
> frame-averaging is worse than temporal modelling (2.17x noise), and training
> beats random weights (4.6x). **This retracts the paired-bootstrap verdict**
> reported alongside it, which called the neural arms significantly worse.
>
> **And the ordinal head does not deliver monotonicity** — 16.2% of ordered
> severity pairs are scored the wrong way round — which is exactly what
> `docs/METHOD.md` said it would not guarantee, now measured.
>
> The solid contributions here are a *validation harness for attribution that
> actually detects failure*, and an honest efficiency characterisation. Details
> and every retraction: [docs/RESULTS.md](docs/RESULTS.md).

---

## The one-paragraph argument

A learner is told their squat scored 0.83. Against what? And what should they do
differently? A similarity score answers neither question, and to produce it at
all you need a reference execution of the same movement — which, for free-form
assessment, does not exist. The information a coach actually gives is *where* and
*when*: "your left knee collapses at the bottom of the rep". That is a
`(joint, phase)` claim, and no scalar can carry it.

So this repository predicts quality directly from a single sequence, with a
calibrated interval, and localises the loss. The hard part is not producing an
attribution map — any gradient method does that. The hard part is knowing whether
the map is **right**, because no human-annotated action-quality dataset records
which joint caused which point of the deduction. That is why the data is
generated: a degradation here is a named, severity-parameterised edit to a
joint-angle trajectory, so *the cause is known exactly*, and attribution
precision and recall can be measured rather than eyeballed.

## What makes the attribution claim checkable

```
generator                          →  quality label   =  f(applied severities)   exact
run forward kinematics twice       →  joint truth     =  what each defect moved  exact
   (with and without each defect)     frame truth
```

Attribution is then scored against that truth **and against two controls that a
convincing-looking heat map has to beat**:

* a uniform-random attribution;
* the *same method applied to an untrained model* — the parameter-randomisation
  sanity check of Adebayo et al. (2018).

Where this repository's attributions fail those controls, that is reported as a
negative result rather than omitted. See [Results](#results).

## Efficiency, measured rather than quoted

<!-- table:efficiency -->
| architecture | params_M | MMACs | latency_bs1_ms | iqr_bs1_ms | latency_bs8_ms | macs_per_ms_bs1 | params_reduction | macs_reduction | latency_reduction |
|---|---|---|---|---|---|---|---|---|---|
| saqa_stgcn_tiny | 0.017 | 4.942 | 8.701 | 2.027 | 19.082 | 0.568 | 179.03 | 176.49 | 8.691 |
| saqa_stgcn | 0.067 | 16.924 | 12.23 | 2.702 | 33.158 | 1.384 | 45.278 | 51.535 | 6.183 |
| stgcn_dense | 0.194 | 43.82 | 11.526 | 1.456 | 39.41 | 3.802 | 15.588 | 19.904 | 6.561 |
| stgcn_reference | 0.759 | 167.542 | 20.57 | 2.19 | 112.029 | 8.145 | 3.985 | 5.206 | 3.676 |
| stgcn_reference_large | 3.024 | 872.194 | 75.62 | 5.391 | 477.009 | 11.534 | 1 | 1 | 1 |
| tcn | 0.123 | 5.834 | 4.888 | 0.304 | 7.78 | 1.193 | 24.645 | 149.499 | 15.469 |
| lstm | 0.16 | 7.545 | 5.148 | 2.126 | 12.694 | 1.466 | 18.918 | 115.594 | 14.688 |
| frame_average | 0.026 | 0.026 | 1.352 | 0.695 | 3.086 | 0.019 | 116.471 | 33900.577 | 55.915 |
<!-- /table -->

Ten warm-up iterations, thirty timed repeats, median and IQR, 2 torch threads at
48 frames. Ratios are against `stgcn_reference_large`, the nine-block 64/128/256
ST-GCN of Yan et al. (2018).

**The efficiency claim that actually separates the two approaches is asymptotic,
not a constant factor.** DTW to a reference is `O(T²)` — quadratic in the frame
cost matrix and quadratic again in the dynamic program. The graph model is
`O(T)`. Both exponents are *fitted from measurements*, not asserted:

<!-- table:cost_curve -->
| num_frames | dtw_total_ms | model_ms | speedup |
|---|---|---|---|
| 24 | 2.155 | 10.89 | 0.198 |
| 48 | 7.303 | 10.72 | 0.681 |
| 96 | 43.478 | 19.911 | 2.184 |
| 192 | 134.822 | 20.915 | 6.446 |
| 384 | 537.419 | 33.698 | 15.948 |
<!-- /table -->

## Results

Five neural arms and three baselines, one dataset, one training loop, one seed;
1000 sequences at 48 frames, cross-subject split, 250 test sequences.

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

**Read the ranking against the noise, not down the column.** A three-seed study
of the identical configuration gives a run-to-run scale of **0.1205 Spearman**:

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

### What survives

* **The structural guarantees, because they are arithmetic.** Zero crossed
  quantile intervals and zero ordinal rank inconsistencies across every arm; the
  independent-logit variant, shipped as a control, crosses on >50% of inputs.
* **Marginal calibration.** 89.6% coverage at a nominal 90%.
* **Training beats random weights** at 4.6x the noise scale, and **temporal
  modelling beats frame-averaging** at 2.17x. Those are the only two accuracy
  claims this study can support outright.
* **Suggestive, and mechanistically predicted:** on `gait` — the one action class
  whose quality signal is inter-limb coordination — the graph model beats the
  graph-free temporal CNN by 0.323 Spearman at 1.83x the noise, with
  **non-overlapping distributions across three seeds** (worst graph seed 0.465 vs
  best CNN seed 0.355). It is the only class of five above the noise floor, and it
  is the class the mechanism predicts. Three seeds is not a significance claim.
* **The efficiency measurements**, which are measurements rather than inferences.
* **Determinism.** Two separate invocations agree to 0.0 across 250 per-sequence
  scores.

### What does not

* **Any ranking among the DTW baseline, the kinematic GBR, the graph model, the
  LSTM and the temporal CNN.** All gaps ≤1.4x the run-to-run scale. In
  particular this repository's model does **not** demonstrate an advantage over
  the reference approach it was built to replace — and the handcrafted-feature
  baseline's apparent 0.10 lead over DTW is equally inside the noise.
* **The attribution fidelity claim**, which is the project's central experiment.
  It fails its own sanity check, robustly, across every architecture and method.
* **Degradation monotonicity**, at 11.5-29.0% of ordered pairs.
* **Selective abstention.** Interval width carries no information about which
  sequences are wrong — error-detection AUROC 0.474-0.555 for *every* method,
  baselines included.

### The efficiency claim, stated precisely

Three separate claims, and only two of them hold.

**Against the reference-scale ST-GCN: real, and measured.** 45.3x fewer
parameters (0.067M vs 3.024M), 51.5x fewer MACs, **6.18x lower latency**
(12.23 ms vs 75.62 ms, batch 1, 48 frames). Note the gap between 51.5x MACs and
6.18x time: the small model retires 1.38 MMAC/ms against the large one's 11.53,
because narrow convolutions are memory-bandwidth bound. MACs are a poor proxy for
latency and this table shows by how much.

**Against DTW: asymptotic, with a crossover on the wrong side of these
experiments.** DTW's fitted cost exponent is **2.013**; the model's MAC count is
linear. But the measured crossover sits between 48 and 96 frames — **at the 48
frames used throughout this study, DTW is 1.5x *faster*.** It only pays off
above ~96 frames (2.2x), reaching 15.9x at 384.

**The separable temporal convolution saves no time at all.** Against
`stgcn_dense`, the identical architecture with dense temporal convolutions:
2.9x fewer parameters, 2.59x fewer MACs, and **6% slower** (12.23 ms vs
11.53 ms). The depthwise convolution that produces the saving is memory-bound.
The parameter reduction is real; the latency reduction is not, and quoting the
MAC ratio alone would have been misleading.

Full tables, statistical tests, the seed study, ablations and the retractions:
**[docs/RESULTS.md](docs/RESULTS.md)**

## Quickstart

Runs end to end with **no dataset download, no API keys, no network**.

```bash
git clone https://github.com/HabibaSajid321/skeletal-action-quality-attribution
cd skeletal-action-quality-attribution

uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python ./.venv/Scripts/python.exe \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch
uv pip install --python ./.venv/Scripts/python.exe \
  numpy scipy pandas scikit-learn pyyaml matplotlib pytest ruff \
  nbformat nbconvert ipykernel
uv pip install --python ./.venv/Scripts/python.exe -e . --no-deps
```

```bash
# 1. Prove it works end to end (~1 minute; real training, real metrics)
python scripts/train.py --config configs/smoke.yaml

# 2. Cost: params, MACs, measured latency, and the DTW-vs-model cost curve
python scripts/benchmark_efficiency.py

# 3. The whole experiment matrix
python scripts/run_all.py

# 4. Tests
python -m pytest tests -q -m "not slow"
```

On Linux/macOS the interpreter is `./.venv/bin/python` and `make setup`,
`make smoke`, `make bench`, `make all`, `make test` do the same things.

> **Python 3.12, not 3.13/3.14.** The 3.14 torch wheels in this environment fail
> to import on a missing bundled `torchgen`, and the PyPI `torchgen` package is
> an unrelated stub that does not fix it.

## The data, and why it is generated

`saqa.data` is a procedural 3D skeleton motion generator: five kinematically
dissimilar actions (`squat`, `overhead_press`, `lunge`, `gait`, `throw`) built
from parametric joint-angle trajectories, mapped onto a fixed 17-joint tree by
forward kinematics, and grounded so the lower ankle sits on the floor.

Six degradation kinds are then applied, each with a known quality cost:

| kind | mechanism | cost per unit severity |
|---|---|---|
| `rom` | reduced range of motion at a DOF **and its mirror** | 0.40 |
| `asymmetry` | the same reduction **on one side only** | 0.35 |
| `tempo` | monotone reparameterisation of time | 0.25 |
| `jerk` | band-limited oscillation added to a DOF | 0.20 |
| `instability` | low-frequency sway added to the pelvis | 0.30 |
| `compensation` | a DOF loses range and a second gains range to hide it | 0.45 |

Three details that decide whether the benchmark measures anything:

**`rom` is bilateral, `asymmetry` is unilateral.** If both were one-sided they
would be *mechanically identical* while carrying different costs — an unlearnable
label, and a silent one. A generator can produce an impossible task without any
error surfacing.
→ `test_rom_is_bilateral_and_asymmetry_is_not`

**The sensor model runs after the labels.** Jitter, noise and occlusion dropout
are applied once the quality label and the attribution ground truth are already
computed. Sensor noise is a nuisance to survive, not a fault to penalise.
→ `test_sensor_model_perturbs_but_does_not_relabel`

**A clean sequence gets an all-zero attribution truth, not a uniform one.** It has
no cause to attribute, so every fidelity metric returns `NaN` there. Scoring it
as uniform would reward a model that always says "everything".

> **The limitation, stated plainly.** The cost weights are a modelling choice, not
> a measured human judgement. This validates the *mechanism* — can a model recover
> a known score function and a known cause? — and says nothing about agreement
> with a real judge. That is a different question, and it is why the optional
> MTL-AQA / FineDiving / NTU loader exists (`scripts/download_real.py`). No
> committed number uses it.

## The method in three pieces

**A compact spatio-temporal graph network, hand-rolled.** ST-GCN-style
convolution (Yan et al., 2018) with three adjacency partitions — identity,
centripetal, centrifugal — written in plain PyTorch. No `torch-geometric`, no
`mmaction2`: a 17-node fixed graph makes message passing one `einsum` against a
constant, so scatter-gather machinery built for million-node dynamic graphs is
strictly slower here, and both libraries are install-fragile. The whole
convolution is fifteen lines and is checked against a hand-computed reference
(`test_graph_conv_matches_hand_computed_message_passing`). The temporal stage is
depthwise-separable, which is the mechanism the parameter reduction rests on.

**An ordinal head whose guarantee is stated precisely.** CORAL-style
(Cao et al., 2020): `K` binary sub-problems share one scalar latent and differ by
ordered thresholds, parameterised so the ordering holds for *every* value the
optimiser can reach. That guarantees **rank consistency** — the predicted
survival function cannot cross itself, and the measured inconsistency rate is
exactly 0, against >0.5 for an independent-logit variant. It does **not**
guarantee that adding a degradation lowers the score; that would require
monotonicity in the input, which no pooled graph network has. So the repository
*measures* the violation rate on severity ladders instead of claiming it.

**Non-crossing quantile intervals.** Pinball loss (Koenker & Bassett, 1978) with
quantiles predicted as a base plus cumulative `softplus` increments, so
`q₀.₀₅ ≤ q₀.₅ ≤ q₀.₉₅` for every parameter value — a crossed interval has
negative width, which is not an interval.

Full derivations and design reasoning: **[docs/METHOD.md](docs/METHOD.md)**.

## What makes the comparison trustworthy

**One training loop for every architecture.** `model.architecture` is the only
thing that differs between arms. Separate scripts per method would let a
difference come from an incidental difference in schedule or checkpointing.

**The reference approach is given a fair fight — and it needed one.** The DTW
baseline is swept over twelve configurations (three distance functions ×
banded/unbanded × canonical/exemplar reference), each with per-action isotonic
calibration onto the quality scale, and the winner is chosen on **validation**.
Per-action calibration is not a detail: distances are not comparable across
action classes, so one global monotone map conflates "which action" with "how
good". On a 300-sequence pilot the baseline scored **ρ = −0.12 without it and
ρ = +0.44 with it**. The first number is what a careless implementation produces,
and reporting it would have been a strawman.

**An untrained network is a reported arm.** A random projection of movement
amplitude already correlates with defect severity, so random weights score a
non-trivial Spearman. Any trained model that does not clear that has learned
nothing the architecture and the input statistics did not already provide. This
control is cheap and easy to omit, which is exactly why it is here.

**Three splitting regimes, all reported.** `random` leaks — the same subject and
the same defect combination appear on both sides — and it is included so the
leakage can be *measured*. `subject` holds out whole subjects. `combination`
holds out every multi-defect sequence containing a compensatory pattern.

**Ablation deltas are divided by the run-to-run noise scale** from a three-seed
study before they are interpreted. A delta smaller than `sqrt(2)·sd` is noise and
is labelled as such.

## Repository layout

```
configs/            one YAML per experiment, composed through `_base_`
  base.yaml           shared defaults; smoke.yaml ~1 minute end to end
  saqa_stgcn / stgcn_dense / stgcn_reference / tcn / lstm / frame_average
  ordinal_off.yaml    the scoring-head ablation in isolation
  real_ntu.yaml       optional real-data path (not run here)

src/saqa/
  config.py           typed config + `_base_` inheritance + --set overrides
  data/
    skeleton.py         topology, adjacency partitions, joint groups
    kinematics.py       forward kinematics, grounding, canonicalisation
    actions.py          five parametric action classes
    degradations.py     the six defect kinds and the exact quality label
    generator.py        the pipeline, and the exact attribution ground truth
    dataset.py          three splitting regimes, severity ladders
    real.py             optional NTU / MTL-AQA / FineDiving loaders
  models/
    stgcn.py            hand-rolled graph conv, separable blocks, make_norm
    trunks.py           graph-free controls: temporal CNN, LSTM, frame average
    heads.py            ordinal (CORAL), regression, quantile, heteroscedastic
    quality.py          the assembled model; registry.py  named architectures
  baselines/
    dtw.py              the reference approach, three distances, banded DP
    kinematic.py        handcrafted biomechanical features
    fitted.py           calibrated, conformal-interval baseline wrappers
  attribution/methods.py  integrated gradients, occlusion, input-gradient
  metrics/
    regression.py  uncertainty.py  monotonicity.py  attribution.py  stats.py
  engine/trainer.py   ONE loop for every architecture and head
  pipelines/          core, analysis, experiments, sweeps
  report.py  viz.py  cli.py  utils/

scripts/            train, compare_methods, run_ablations, data_efficiency,
                    benchmark_efficiency, make_figures, build_notebooks,
                    render_docs, report_tables, download_real, run_all
notebooks/          01 data & ground truth · 02 train & compare · 03 ablations
                    & noise · 04 attribution & uncertainty · 05 Colab full scale
tests/              482 tests
docs/               METHOD.md · RESULTS.md · REPRODUCIBILITY.md
results/            tables/ figures/ runs/ — the evidence, committed
```

## Notebooks

| notebook | what it establishes | needs a GPU |
|---|---|---|
| `01_data_and_ground_truth.ipynb` | the generator is a real benchmark; the attribution ground truth is concentrated, not uniform | no |
| `02_train_and_compare.ipynb` | the reference approach given a fair fight, and what per-action calibration is worth | no |
| `03_ablations_and_noise.ipynb` | which component does the work, and the GroupNorm failure measured | no |
| `04_attribution_and_uncertainty.ipynb` | is the explanation correct, and does the model know when it is wrong | no |
| `05_colab_full_scale.ipynb` | the same code at 20000 sequences and 192 frames, with the 3.0M reference trained | yes |

## Configuration

One YAML fully describes an experiment, and every run saves its resolved config
next to its results. Configs compose through `_base_`, any leaf can be overridden
from the command line, and **unknown keys are rejected** — a silently-swallowed
typo would void an experiment.

```bash
python scripts/train.py --config configs/saqa_stgcn.yaml \
  --set optim.epochs=40 model.partitions=uniform data.split=combination
```

## Citation

```bibtex
@software{sajid2026saqa,
  author = {Habiba Sajid},
  title  = {What Should I Change? Reference-Free Calibrated Skeletal
            Action-Quality Regression with Validated Attribution},
  year   = {2026},
  url    = {https://github.com/HabibaSajid321/skeletal-action-quality-attribution}
}
```

Builds on: Yan et al. ST-GCN (2018); Shi et al. 2s-AGCN (2019); Parmar & Morris
on action quality assessment and MTL-AQA (2019); Tang et al. uncertainty-aware
score distribution learning (2020); Xu et al. FineDiving (2022); Sundararajan et
al. integrated gradients (2017); Adebayo et al. sanity checks for saliency maps
(2018); Sakoe & Chiba DTW (1978); Koenker & Bassett quantile regression (1978);
Cao et al. CORAL (2020). Full list in [docs/METHOD.md](docs/METHOD.md#7-references).

## License

MIT — see [LICENSE](LICENSE).
