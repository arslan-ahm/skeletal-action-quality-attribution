"""Generate the notebooks programmatically, then optionally execute them.

    python scripts/build_notebooks.py            # write .ipynb files
    python scripts/build_notebooks.py --execute  # write and run, keeping outputs

Notebooks are generated rather than hand-written so that the code in them cannot
drift from the package: every cell calls into ``saqa``, and none of them
reimplements an experiment. Cell ids are set explicitly, which nbformat 4.5+
requires.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

HEADER = """import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
pd.set_option("display.width", 200)
import matplotlib.pyplot as plt
"""


def _nb(cells) -> nbformat.NotebookNode:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:03d}"
    return nb


def notebook_01():
    return _nb([
        new_markdown_cell(
            "# 1. The data, and why a synthetic generator is the point\n\n"
            "Reference-free action-quality assessment has a measurement problem: to "
            "check whether an explanation is *correct* you need to know what actually "
            "caused the quality loss, and no human-annotated dataset carries that.\n\n"
            "This generator does. A degradation is a named, severity-parameterised "
            "edit to a joint-angle trajectory; the quality label is an explicit "
            "function of the applied severities; and the per-joint attribution ground "
            "truth is obtained by running forward kinematics twice, with and without "
            "each defect.\n\n"
            "**The limitation, stated up front:** a synthetic label is not a human "
            "judge's rating. This validates the *mechanism*, not real-world agreement."
        ),
        new_code_cell(HEADER + "from saqa.data import build_dataset, GeneratorConfig, "
                      "label_stats, ACTION_CLASSES, KIND_WEIGHTS\n"
                      "from saqa.data.generator import generate_sample\n"
                      "cfg = GeneratorConfig(num_frames=48)\n"
                      "data = build_dataset(400, cfg)\n"
                      "print(ACTION_CLASSES)\nprint(KIND_WEIGHTS)\n"
                      "pd.Series(label_stats(data)).round(4)"),
        new_markdown_cell("## Five kinematically distinct actions"),
        new_code_cell(
            "from saqa.viz import plot_skeleton_frames\n"
            "for action in ACTION_CLASSES:\n"
            "    s = next(x for x in (generate_sample(i, cfg) for i in range(400))\n"
            "             if x.action == action and not x.degradations)\n"
            "    plot_skeleton_frames(s, frames=(0, 12, 24, 36), "
            "name=f'nb_skeleton_{action}.png')\n"
            "    plt.show()"
        ),
        new_markdown_cell(
            "## The label distribution\n\n"
            "The ceiling at 1.0 is populated on purpose (clean executions exist) and "
            "the floor is clipped at 0.05 so the bottom decile does not pile up, which "
            "would make rank correlation look better than it is."
        ),
        new_code_cell(
            "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n"
            "ax[0].hist(data.quality, bins=30, color='#2b6cb0')\n"
            "ax[0].set_xlabel('quality label'); ax[0].set_title('label distribution')\n"
            "counts = pd.Series(data.combinations).value_counts().head(12)\n"
            "ax[1].barh(counts.index[::-1], counts.values[::-1], color='#c05621')\n"
            "ax[1].set_title('defect combinations'); plt.tight_layout(); plt.show()"
        ),
        new_markdown_cell(
            "## Attribution ground truth is concentrated, not uniform\n\n"
            "If the truth were near-uniform over joints, attribution precision would "
            "be uninformative. It is not: the top three joints carry far more than the "
            "3/17 a uniform vector would give."
        ),
        new_code_cell(
            "jt = data.joint_truth[data.joint_truth.sum(1) > 0]\n"
            "top3 = np.sort(jt, axis=1)[:, -3:].sum(1)\n"
            "print(f'top-3 joint mass: mean {top3.mean():.3f} vs {3/17:.3f} if uniform')\n"
            "plt.hist(top3, bins=25, color='#2b6cb0')\n"
            "plt.axvline(3/17, color='k', ls='--', label='uniform')\n"
            "plt.xlabel('ground-truth mass in the top 3 joints'); plt.legend(); plt.show()"
        ),
        new_markdown_cell(
            "## The label is monotone in severity by construction\n\n"
            "This is what makes the model-side monotonicity test meaningful: on a "
            "severity ladder, nothing varies but the defect."
        ),
        new_code_cell(
            "from saqa.data import paired_degradation_sequences\n"
            "lad = paired_degradation_sequences(6, GeneratorConfig(num_frames=48), seed=0)\n"
            "for x in lad[:3]:\n"
            "    print(f\"{x['kind']:14s} {x['action']:16s} labels {np.round(x['quality'],3)}\")"
        ),
    ])


def notebook_02():
    return _nb([
        new_markdown_cell(
            "# 2. Train, and compare against the reference approach\n\n"
            "The reference repository's method is DTW alignment to a reference "
            "performance. It is implemented here properly and given every advantage: "
            "three distance functions, banded and unbanded warping, a canonical or a "
            "real exemplar reference, and a per-action isotonic calibration of its "
            "distance onto the quality scale. The configuration is selected on "
            "**validation**, never on test."
        ),
        new_code_cell(HEADER + "from saqa.config import load_config\n"
                      "from saqa.pipelines import make_splits, run_single, fit_baselines\n"
                      "cfg = load_config('../configs/base.yaml', "
                      "['data.num_sequences=600', 'optim.epochs=10', "
                      "'run.out_dir=../results/runs_nb'])\n"
                      "splits = make_splits(cfg); splits.sizes()"),
        new_markdown_cell(
            "## Per-action calibration is worth more than the alignment\n\n"
            "A single global isotonic map conflates 'which action' with 'how good', "
            "because a throw and a gait cycle have different spatial extents. Fixing "
            "that is the single largest change to the baseline's score, and it is the "
            "difference between a strawman and a real comparison."
        ),
        new_code_cell(
            "from saqa.baselines.fitted import DTWBaseline\n"
            "from saqa.metrics import spearman\n"
            "from saqa.pipelines import generator_config\n"
            "g = generator_config(cfg)\n"
            "rows = []\n"
            "for per_action in (False, True):\n"
            "    m = DTWBaseline(generator=g, per_action_calibration=per_action)"
            ".fit(splits.train)\n"
            "    s, _, _ = m.predict(splits.test)\n"
            "    rows.append({'per_action_calibration': per_action,\n"
            "                 'test_spearman': spearman(splits.test.quality, s)})\n"
            "pd.DataFrame(rows).round(4)"
        ),
        new_markdown_cell("## Train the graph model and the graph-free controls"),
        new_code_cell(
            "runs = {}\n"
            "for arch in ('saqa_stgcn', 'tcn', 'frame_average'):\n"
            "    runs[arch] = run_single(cfg, name=f'nb_{arch}', architecture=arch,\n"
            "                            splits=splits, save=False, verbose=False)\n"
            "    print(f\"{arch:16s} rho {runs[arch].metrics['spearman']:+.4f}  \"\n"
            "          f\"params {runs[arch].metrics['params']:.0f}\")"
        ),
        new_code_cell(
            "base = fit_baselines(splits, cfg, dtw_sweep=False)\n"
            "rows = [{'method': k, 'spearman': spearman(splits.test.quality, v.metrics['spearman'] "
            "if False else runs[k].pred['score'])} for k in runs]\n"
            "for k in ('kinematic_gbr', 'dtw_reference'):\n"
            "    rows.append({'method': k, 'spearman': "
            "spearman(splits.test.quality, base[k]['score'])})\n"
            "pd.DataFrame(rows).sort_values('spearman', ascending=False).round(4)"
        ),
        new_markdown_cell(
            "## The untrained control\n\n"
            "A random-weights network is **not** a trivial baseline here: a random "
            "projection of movement amplitude already correlates with defect severity. "
            "Any trained model that does not clear this has learned nothing the "
            "architecture and the input statistics did not already provide."
        ),
        new_code_cell(
            "from saqa.models import build_model\n"
            "from saqa.engine import predict\n"
            "torch.manual_seed(999)\n"
            "u = predict(build_model('saqa_stgcn'), splits.test.coords)['score']\n"
            "print(f'untrained saqa_stgcn: rho {spearman(splits.test.quality, u):+.4f}')"
        ),
        new_markdown_cell("## Committed results (from `make all`)"),
        new_code_cell(
            "for name in ('method_comparison', 'statistical_tests'):\n"
            "    p = pathlib.Path('../results/tables') / f'{name}.csv'\n"
            "    if p.exists():\n"
            "        print(f'--- {name} ---'); display(pd.read_csv(p).round(4))"
        ),
    ])


def notebook_03():
    return _nb([
        new_markdown_cell(
            "# 3. Ablations, read against the noise\n\n"
            "An ablation delta is meaningless without knowing how far two runs of the "
            "*same* configuration drift apart. The seed study is therefore run first "
            "and every delta is divided by `sqrt(2) * sd` before it is interpreted."
        ),
        new_code_cell(HEADER + "tables = pathlib.Path('../results/tables')\n"
                      "var = pd.read_csv(tables / 'seed_variance.csv') if "
                      "(tables / 'seed_variance.csv').exists() else None\n"
                      "var.round(4) if var is not None else 'run `make ablate` first'"),
        new_code_cell(
            "abl = pd.read_csv(tables / 'ablation_components.csv') if "
            "(tables / 'ablation_components.csv').exists() else None\n"
            "cols = ['variant', 'change', 'spearman', 'spearman_delta', "
            "'spearman_ratio_to_noise', 'spearman_verdict']\n"
            "abl[[c for c in cols if c in abl.columns]].round(4) if abl is not None else None"
        ),
        new_markdown_cell(
            "## The normalisation choice is load-bearing\n\n"
            "The first version of this model used GroupNorm everywhere, for "
            "batch-size independence. It could not fit its own training set. "
            "GroupNorm removes each sample's per-channel scale at every layer, and the "
            "readout is a global average pool over (T, V) -- which reads precisely "
            "that scale. This is measured below, not asserted."
        ),
        new_code_cell(
            "from saqa.config import load_config\n"
            "from saqa.pipelines import make_splits, run_single\n"
            "from saqa.metrics import spearman\n"
            "from saqa.engine import predict\n"
            "cfg = load_config('../configs/base.yaml', "
            "['data.num_sequences=500', 'optim.epochs=10', 'run.out_dir=../results/runs_nb'])\n"
            "sp = make_splits(cfg)\n"
            "for norm in ('batch', 'group'):\n"
            "    c = load_config('../configs/base.yaml', "
            "['data.num_sequences=500', 'optim.epochs=10', f'model.norm={norm}', "
            "'run.out_dir=../results/runs_nb'])\n"
            "    r = run_single(c, name=f'nb_norm_{norm}', splits=sp, save=False, verbose=False)\n"
            "    tr = spearman(sp.train.quality, predict(r.model, sp.train.coords)['score'])\n"
            "    print(f'{norm:6s} train rho {tr:+.4f}   test rho "
            "{r.metrics[\"spearman\"]:+.4f}')"
        ),
        new_markdown_cell(
            "## Splitting regimes: the leakage is measured, not assumed\n\n"
            "A random split lets the same subject and the same defect combination "
            "appear on both sides. The gap between it and the cross-subject split is "
            "the leakage estimate."
        ),
        new_code_cell(
            "p = tables / 'split_comparison.csv'\n"
            "pd.read_csv(p)[['split', 'spearman', 'kendall_tau', 'relative_l2', 'mae']]"
            ".round(4) if p.exists() else 'run `make splits` first'"
        ),
    ])


def notebook_04():
    return _nb([
        new_markdown_cell(
            "# 4. Attribution fidelity and calibrated uncertainty\n\n"
            "This is the notebook the project exists for. Two questions:\n\n"
            "1. **Is the explanation correct?** Scored against exact ground truth, and "
            "against two controls -- a uniform-random attribution, and the same method "
            "applied to an *untrained* model (the parameter-randomisation sanity check "
            "of Adebayo et al., 2018).\n"
            "2. **Does the model know when it is wrong?** Interval coverage, sharpness, "
            "and a risk-coverage curve for abstention."
        ),
        new_code_cell(HEADER + "tables = pathlib.Path('../results/tables')\n"
                      "fid = pd.read_csv(tables / 'attribution_fidelity.csv') if "
                      "(tables / 'attribution_fidelity.csv').exists() else None\n"
                      "cols = ['model', 'attribution', 'joint_precision', 'joint_recall',\n"
                      "        'joint_iou', 'joint_rank_corr', 'joint_top1_hit',\n"
                      "        'frame_localisation_error', 'n_joint_iou']\n"
                      "fid[[c for c in cols if c in fid.columns]].round(4) if fid is not None "
                      "else 'run `make compare` first'"),
        new_markdown_cell(
            "**How to read this.** The rows to compare against are `random` and "
            "`integrated_gradients_untrained`. An attribution method that does not "
            "beat both is not explaining the model -- it is reflecting the input "
            "statistics. Where that happens it is reported as a negative result."
        ),
        new_code_cell(
            "from saqa.viz import plot_attribution_fidelity\n"
            "p = plot_attribution_fidelity('nb_attribution_fidelity.png')\n"
            "print(p); plt.show()"
        ),
        new_markdown_cell("## A single sequence, explained"),
        new_code_cell(
            "from saqa.config import load_config\n"
            "from saqa.pipelines import make_splits, run_single, compute_attribution\n"
            "from saqa.viz import plot_attribution_map\n"
            "cfg = load_config('../configs/base.yaml', "
            "['data.num_sequences=500', 'optim.epochs=10', 'run.out_dir=../results/runs_nb'])\n"
            "sp = make_splits(cfg)\n"
            "res = run_single(cfg, name='nb_attr', splits=sp, save=False, verbose=False)\n"
            "k = int(np.argmax(sp.test.joint_truth.sum(1) > 0))\n"
            "attr = compute_attribution(res.model, sp.test.coords[k:k+1], "
            "'integrated_gradients', cfg)[0]\n"
            "plot_attribution_map(attr, sp.test.joint_truth[k], sp.test.frame_truth[k],\n"
            "                     name='nb_attribution_map.png'); plt.show()\n"
            "print(sp.test.samples[k].action, [d.kind for d in sp.test.samples[k].degradations])"
        ),
        new_markdown_cell(
            "## Monotonicity: does adding a defect ever *raise* the score?\n\n"
            "The ordinal head guarantees rank consistency (its survival function cannot "
            "cross itself), but it does **not** guarantee monotonicity in the input. "
            "That is an empirical question and it is measured, not claimed."
        ),
        new_code_cell(
            "from saqa.pipelines import monotonicity_check, ordinal_consistency\n"
            "rep, ladders, preds = monotonicity_check(res.model, cfg, num_ladders=24)\n"
            "print({k: round(v, 4) for k, v in rep.items() if not k.startswith('n__')})\n"
            "print(ordinal_consistency(res.model, sp.test))"
        ),
        new_code_cell(
            "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
            "for lad, pr in list(zip(ladders, preds))[:8]:\n"
            "    ax.plot(lad['severity'], pr, 'o-', alpha=0.7, label=lad['kind'])\n"
            "ax.set_xlabel('defect severity'); ax.set_ylabel('predicted quality')\n"
            "ax.set_title('every line should slope down'); ax.grid(alpha=0.3)\n"
            "ax.legend(fontsize=7); plt.show()"
        ),
        new_markdown_cell(
            "## Intervals: coverage next to sharpness\n\n"
            "Coverage alone cannot distinguish a useful interval from a constant-width "
            "one, so the width conditional on the model being wrong is reported "
            "alongside. A ratio above 1 means the interval widens where it should."
        ),
        new_code_cell(
            "p = tables / 'uncertainty.csv'\n"
            "pd.read_csv(p).round(4) if p.exists() else 'run `make compare` first'"
        ),
        new_code_cell(
            "from saqa.viz import plot_risk_coverage\n"
            "p = plot_risk_coverage('../results/runs/saqa_stgcn/per_item.csv',\n"
            "                       name='nb_risk_coverage.png')\n"
            "print(p); plt.show() if p else print('run `make compare` first')"
        ),
    ])


def notebook_05():
    return _nb([
        new_markdown_cell(
            "# 5. Full-scale run on a GPU (Colab / Kaggle)\n\n"
            "The committed results were produced on **two CPU threads with no GPU**, "
            "and that constraint is visible in their scale: 48 frames, 1400 sequences, "
            "16 epochs, one seed per ablation. This notebook runs the *identical code* "
            "at a scale that settles the questions the CPU budget could not.\n\n"
            "Nothing here has been run locally. No number in `results/` comes from this "
            "notebook."
        ),
        new_code_cell(
            "# Colab setup. Skip if you already have the repository.\n"
            "# !git clone https://github.com/HabibaSajid321/skeletal-action-quality-attribution\n"
            "# %cd skeletal-action-quality-attribution\n"
            "# !pip install -q torch numpy scipy pandas scikit-learn pyyaml matplotlib\n"
            "# !pip install -q -e . --no-deps"
        ),
        new_code_cell(HEADER + "print('cuda available:', torch.cuda.is_available())"),
        new_markdown_cell(
            "## What to run at full scale, and why\n\n"
            "| experiment | CPU budget here | what a GPU settles |\n"
            "|---|---|---|\n"
            "| seeds per configuration | 3 | 10+, which is what an accuracy claim needs |\n"
            "| the 3.0M reference ST-GCN | **not trained** (~40 min/run) | "
            "its accuracy, not just its cost |\n"
            "| sequence length | 48 frames | 300 frames, the ST-GCN paper's regime |\n"
            "| dataset size | 1400 | 20000, where the data-efficiency curve flattens |"
        ),
        new_code_cell(
            "from saqa.config import load_config\n"
            "from saqa.pipelines import method_comparison, make_splits\n"
            "cfg = load_config('configs/base.yaml', [\n"
            "    'data.num_sequences=20000', 'data.num_frames=192', 'data.batch_size=64',\n"
            "    'data.num_subjects=120', 'optim.epochs=60',\n"
            "])\n"
            "# arms includes the full 3.0M reference architecture, which is out of\n"
            "# budget on CPU and is the point of running this here.\n"
            "arms = ('saqa_stgcn', 'stgcn_reference', 'stgcn_reference_large', 'tcn',\n"
            "        'lstm', 'frame_average')\n"
            "# table, runs, baselines = method_comparison(cfg, make_splits(cfg), arms)\n"
            "# table.round(4)"
        ),
        new_markdown_cell(
            "## The real-data path\n\n"
            "`saqa.data.real` loads NTU RGB+D skeletons and MTL-AQA / FineDiving style "
            "score annotations. Both need a manual download; "
            "`scripts/download_real.py` prints the steps and the expected layout. "
            "FineDiving is the one public dataset that can validate the *temporal* half "
            "of the attribution claim against human annotation rather than against a "
            "generator."
        ),
        new_code_cell(
            "# from saqa.data.real import load_aqa_directory\n"
            "# coords, scores, ids = load_aqa_directory('data/raw/mtl_aqa', num_frames=192)\n"
            "# print(coords.shape, scores.min(), scores.max())"
        ),
    ])


BUILDERS = {
    "01_data_and_ground_truth.ipynb": notebook_01,
    "02_train_and_compare.ipynb": notebook_02,
    "03_ablations_and_noise.ipynb": notebook_03,
    "04_attribution_and_uncertainty.ipynb": notebook_04,
    "05_colab_full_scale.ipynb": notebook_05,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    names = args.only or list(BUILDERS)
    for name in names:
        nb = BUILDERS[name]()
        path = OUT / name
        path.write_text(nbformat.writes(nb), encoding="utf-8", newline="\n")
        print(f"  wrote {path}")

    if args.execute:
        from nbclient import NotebookClient

        for name in names:
            if name.startswith("05_"):
                print(f"  skipping {name} (needs a GPU; cells are commented out)")
                continue
            path = OUT / name
            nb = nbformat.read(path, as_version=4)
            print(f"  executing {path} ...", flush=True)
            client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                    resources={"metadata": {"path": str(OUT)}})
            client.execute()
            path.write_text(nbformat.writes(nb), encoding="utf-8", newline="\n")
            print(f"  executed {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
