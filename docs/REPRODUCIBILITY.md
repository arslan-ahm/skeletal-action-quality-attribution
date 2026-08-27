# Reproducibility

Every number in this repository comes from a command listed here, run on the
machine described here, with no network access and no dataset download.

## Environment

The committed results were produced with:

| component | version |
|---|---|
| Python | 3.12.13 |
| torch | 2.13.0+cpu |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| OS | Windows 10 (10.0.19045) |
| CPU | 4 cores, **capped to 2 torch threads** |
| GPU | none |

> **Python 3.12, not 3.13/3.14.** The 3.14 torch wheels available in this
> environment fail to import on a missing bundled `torchgen`, and the `torchgen`
> package on PyPI is an unrelated stub that does not fix it. `.python-version`
> pins 3.12.

### Setup

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

On Linux/macOS the interpreter is `./.venv/bin/python`, and `make setup` does
all of the above.

## Determinism

Every entry point calls `saqa.utils.seed.seed_everything(seed, threads)`, which
seeds Python, NumPy and torch, sets `PYTHONHASHSEED` and `OMP_NUM_THREADS`, calls
`torch.use_deterministic_algorithms(True, warn_only=True)`, and **pins the thread
count**. The last one matters as much as the seeds: some CPU reduction kernels
choose a different split by thread count, which changes the floating-point
summation order and therefore the last bits of a result.

Verified, not assumed:

```bash
python -m pytest tests/test_pipelines.py::test_two_identical_runs_are_bit_identical -q
```

This trains the same configuration twice in separate model instances and asserts
**max absolute difference 0.0** across every per-sequence score. It passes.

Determinism evidence measured on the committed run:

<!-- DETERMINISM_EVIDENCE -->

The data generator is deterministic independently of torch: every sample is a
pure function of one integer and the generator config
(`test_sample_is_a_pure_function_of_its_index`), so a dataset can be regenerated
from a seed without keeping any files.

## Data

**No downloads. No API keys. Nothing to register for.** The default and only
tested data source is `saqa.data.generator`, a procedural 3D skeleton motion
generator. See `docs/METHOD.md` §3 for why the synthetic path is the enabling
condition for the attribution experiment rather than a compromise.

An optional real-data path exists and is **not** used by any committed number:

```bash
python scripts/download_real.py          # prints retrieval steps and layout
python scripts/download_real.py --check  # reports what is present locally
```

## Reproducing the results

```bash
# 0. Prove the pipeline works end to end (~1 minute, real training)
python scripts/train.py --config configs/smoke.yaml

# 1. Cost: params, MACs, measured latency, and the DTW-vs-model cost curve
python scripts/benchmark_efficiency.py

# 2. The whole experiment matrix, in the order the results are read in
python scripts/run_all.py

# or stage by stage
python scripts/compare_methods.py --set data.num_sequences=1000 optim.epochs=10
python scripts/run_ablations.py --only seeds --set data.num_sequences=1000 optim.epochs=10
python scripts/run_ablations.py --only ablations --set data.num_sequences=1000 optim.epochs=8
python scripts/run_ablations.py --only splits --set data.num_sequences=1000 optim.epochs=10
python scripts/data_efficiency.py --set data.num_sequences=1000 optim.epochs=10

# 3. Figures, from the committed CSVs
python scripts/make_figures.py

# 4. Notebooks
python scripts/build_notebooks.py --execute

# 5. Tests
python -m pytest tests -q -m "not slow"   # fast suite
python -m pytest tests -q                 # everything
```

`make all` runs step 2. `make help` lists every target.

## Compute, and what it costs the results

<!-- TIMING_TABLE -->

**The scale is a constraint, and it is visible in the results rather than hidden
by them.** Four shared CPU cores and a 20-minute ceiling per training run mean:

* one training seed per ablation variant, which is why every ablation delta is
  divided by the seed-study noise scale before it is interpreted;
* three seeds for the seed study, where settling an accuracy claim needs 5–10;
* 48 frames per sequence, against the 300 of the ST-GCN paper's regime;
* the full 3.0M-parameter reference ST-GCN is **benchmarked for cost but not
  trained** — a single run is roughly 40 minutes here, over the per-run budget.
  Training it on a shortened schedule would produce a number that says more about
  the schedule than about the architecture, so no accuracy figure is claimed for
  it. `notebooks/05_colab_full_scale.ipynb` runs it on a GPU.

## Measurement pitfalls found while building this

Four things that produced wrong numbers before they were fixed. They are listed
because each is easy to repeat.

**Warm-up.** Fewer than about eight untimed iterations lets PyTorch's first-call
allocation and kernel selection dominate a small model's measurement. The
benchmark uses 10 warm-up iterations and 30 timed repeats and reports median and
IQR, never the mean.

**Contention.** This machine ran other jobs during the benchmark, and the first
latency table it produced was unusable: `stgcn_dense` (43.8 MMACs) measured
*slower* than `stgcn_reference` (167.5 MMACs), and IQRs reached 148 ms on a
median of 108 ms. The efficiency table is re-measured on a quiet machine and the
IQR is reported next to every median so a reader can see when a row is not
trustworthy.

**Normalisation and the readout.** GroupNorm removes each sample's per-channel
scale; global average pooling reads exactly that scale. The first model could
not fit its own training set (train Spearman 0.147 vs 0.855). See
`docs/METHOD.md` §4.3.

**Config overrides reaching named architectures.** A generic `model.channels` in
the config silently overrode the shapes of `stgcn_dense` and `stgcn_reference`,
collapsing three comparison arms into one identical model — visible only because
their attribution scores were identical to six decimal places. Guarded by
`SHAPE_OWNING_ARCHITECTURES` and
`test_named_variants_keep_their_own_shape`.

**DTW band anchoring.** The Sakoe-Chiba band was centred on `(i-1)*m/n` rather
than on the cell the diagonal actually passes through, shifting the band off the
diagonal. Caught by a symmetry test. The same test uncovered that
`per_joint_norm` is *deliberately* a directed distance (it normalises by the
reference's spread), which is now asserted so nobody "fixes" it into symmetry and
silently changes what the baseline measures.

## Provenance

* `results/tables/*.csv` — produced by the commands above; every table in the
  documentation is cross-checked against its CSV.
* `results/runs/<name>/` — `config.yaml` (the fully resolved config),
  `history.jsonl` (per-epoch), `per_item.csv` (per-sequence predictions and
  errors), `summary.json` (every metric), `analysis.json` where `train.py` was
  used.
* `results/figures/*.png` — drawn from the CSVs by `scripts/make_figures.py`.
* `results/run_all.log` — the console log of the matrix run that produced them.

Checkpoints are **not** committed (`.gitignore` excludes `*.pt`); `results/` is,
because that is the evidence.
