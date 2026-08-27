"""Figures, drawn from committed artefacts rather than from live objects.

Every figure reads a CSV in ``results/tables/`` so that the picture and the
number in the documentation cannot drift apart. The one exception is the
skeleton/attribution panel, which needs a model.

The matplotlib backend is **never** forced at import time. Doing so inside a
notebook silently renders every plot blank, which is a failure mode that looks
exactly like "the cell produced no output".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIGURES = Path("results/figures")
TABLES = Path("results/tables")


def _plt():
    """Import pyplot, choosing Agg only outside a notebook."""
    import matplotlib

    if "ipykernel" not in sys.modules and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return path


def _read(name: str):
    import pandas as pd

    path = TABLES / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_method_comparison(name: str = "method_comparison.png") -> Path | None:
    """Spearman and relative L2 per method, families colour-coded."""
    df = _read("method_comparison.csv")
    if df is None:
        return None
    plt = _plt()
    colours = {"neural": "#2b6cb0", "baseline": "#c05621", "control": "#718096"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = df.sort_values("spearman", ascending=False)
    for ax, metric, better in ((axes[0], "spearman", "higher is better"),
                               (axes[1], "relative_l2", "lower is better")):
        sub = order.sort_values(metric, ascending=(metric != "spearman"))
        ax.barh(sub["method"], sub[metric],
                color=[colours.get(f, "#999") for f in sub["family"]])
        ax.set_xlabel(f"{metric} ({better})")
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(0, color="k", lw=0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    axes[0].legend(handles, colours.keys(), loc="lower right", fontsize=8)
    fig.suptitle("Quality regression: every method, one dataset, one training loop")
    fig.tight_layout()
    return _save(fig, name)


def plot_attribution_fidelity(name: str = "attribution_fidelity.png") -> Path | None:
    """Joint-attribution fidelity against the random and untrained controls."""
    df = _read("attribution_fidelity.csv")
    if df is None:
        return None
    plt = _plt()
    metric = "joint_rank_corr"
    if metric not in df.columns:
        return None
    sub = df[df["model"].isin(["saqa_stgcn", "dtw_reference"])].copy()
    if sub.empty:
        sub = df.copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [f"{m}\n{a}" for m, a in zip(sub["model"], sub["attribution"], strict=True)]
    is_control = ["random" in a or "untrained" in a for a in sub["attribution"]]
    ax.bar(range(len(sub)), sub[metric],
           color=["#a0aec0" if c else "#2b6cb0" for c in is_control])
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("rank correlation with ground-truth joint attribution")
    ax.set_title("Attribution fidelity (grey = controls that must be beaten)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, name)


def plot_cost_curve(name: str = "cost_vs_length.png") -> Path | None:
    """DTW's quadratic cost against the model's linear cost."""
    df = _read("cost_vs_length.csv")
    if df is None:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(df["num_frames"], df["dtw_total_ms"], "o-", label="DTW to a reference")
    ax.loglog(df["num_frames"], df["model_ms"], "s-", label="graph model, batch 1")
    ax.set_xlabel("sequence length (frames)")
    ax.set_ylabel("wall-clock per sequence (ms)")
    ax.set_title(
        f"measured exponents: DTW {df['dtw_exponent'].iloc[0]:.2f}, "
        f"model {df['model_exponent'].iloc[0]:.2f}"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_efficiency(name: str = "efficiency.png") -> Path | None:
    """Parameters against measured latency, with the MACs-per-ms annotation."""
    df = _read("efficiency.csv")
    if df is None:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df["params"], df["latency_bs1_ms"], s=60, c="#2b6cb0")
    for _, row in df.iterrows():
        ax.annotate(row["architecture"], (row["params"], row["latency_bs1_ms"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("median latency, batch 1 (ms)")
    ax.set_title("Cost: parameters vs measured CPU latency")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return _save(fig, name)


def plot_data_efficiency(name: str = "data_efficiency.png") -> Path | None:
    """Spearman against the labelled-sequence budget."""
    df = _read("data_efficiency.csv")
    if df is None:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("n_train")
        ax.plot(sub["n_train"], sub["spearman"], "o-", label=method)
    ax.set_xlabel("labelled training sequences")
    ax.set_ylabel("test Spearman")
    ax.set_title("Label efficiency: the axis that costs money in practice")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def plot_risk_coverage(per_item_csv: str, name: str = "risk_coverage.png") -> Path | None:
    """Risk-coverage curve against the oracle, from a run's per-item CSV."""
    import pandas as pd

    path = Path(per_item_csv)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "width" not in df.columns:
        return None
    from .metrics.uncertainty import risk_coverage_curve

    plt = _plt()
    err = df["abs_error"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, conf in (("interval width", -df["width"].to_numpy()),
                        ("oracle (true error)", -err)):
        covs, risks = risk_coverage_curve(err, conf)
        ax.plot(covs, risks, label=label)
    ax.set_xlabel("coverage (fraction of sequences graded)")
    ax.set_ylabel("mean absolute error of the graded set")
    ax.set_title("When should the system decline to grade?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_skeleton_frames(sample, frames: tuple[int, ...] = (0, 8, 16, 24),
                         name: str = "skeleton_frames.png") -> Path:
    """Stick-figure snapshots of one generated execution, in the sagittal plane."""
    from .data.skeleton import PARENTS

    plt = _plt()
    pos = np.asarray(sample.positions)
    frames = tuple(f for f in frames if f < pos.shape[0])
    fig, axes = plt.subplots(1, len(frames), figsize=(2.6 * len(frames), 4.2),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    # One set of limits for every panel, from the whole sequence. Per-panel
    # autoscaling makes a squat and a stand look identical, which defeats the
    # purpose of a motion figure.
    z, y = pos[:, :, 2], pos[:, :, 1]
    pad = 0.08
    span = max(np.ptp(z), np.ptp(y)) / 2 + pad
    zc, yc = (z.max() + z.min()) / 2, (y.max() + y.min()) / 2
    for ax, f in zip(axes, frames, strict=True):
        for j, par in enumerate(PARENTS):
            if par < 0:
                continue
            ax.plot([pos[f, par, 2], pos[f, j, 2]], [pos[f, par, 1], pos[f, j, 1]],
                    "-o", color="#2b6cb0", ms=3, lw=1.6)
        ax.set_title(f"frame {f}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(zc - span, zc + span)
        ax.set_ylim(yc - span, yc + span)
        ax.set_xticks([round(zc - span / 2, 1), round(zc + span / 2, 1)])
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("height (body heights)", fontsize=8)
    fig.suptitle(
        f"{sample.action}  |  quality {sample.quality:.2f}  |  "
        f"{sample.combination}", fontsize=11
    )
    fig.tight_layout()
    return _save(fig, name)


def plot_attribution_map(attribution: np.ndarray, truth_joint: np.ndarray,
                         truth_frame: np.ndarray, name: str = "attribution_map.png"):
    """A single sequence's (T, V) attribution next to its ground truth."""
    from .data.skeleton import JOINT_NAMES

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             gridspec_kw={"width_ratios": [2.2, 1, 1]})
    a = np.clip(np.asarray(attribution), 0, None)
    im = axes[0].imshow(a.T, aspect="auto", cmap="magma", origin="lower")
    axes[0].set_yticks(range(len(JOINT_NAMES)))
    axes[0].set_yticklabels(JOINT_NAMES, fontsize=6)
    axes[0].set_xlabel("frame")
    axes[0].set_title("predicted attribution (joint x phase)")
    fig.colorbar(im, ax=axes[0], fraction=0.03)

    pred_joint = a.sum(axis=0)
    pred_joint = pred_joint / max(pred_joint.sum(), 1e-12)
    idx = np.arange(len(JOINT_NAMES))
    axes[1].barh(idx - 0.2, pred_joint, height=0.4, label="predicted")
    axes[1].barh(idx + 0.2, truth_joint, height=0.4, label="ground truth")
    axes[1].set_yticks(idx)
    axes[1].set_yticklabels(JOINT_NAMES, fontsize=6)
    axes[1].set_title("per joint")
    axes[1].legend(fontsize=7)

    pred_frame = a.sum(axis=1)
    pred_frame = pred_frame / max(pred_frame.sum(), 1e-12)
    axes[2].plot(pred_frame, label="predicted")
    axes[2].plot(truth_frame, label="ground truth")
    axes[2].set_xlabel("frame")
    axes[2].set_title("per phase")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, name)


def all_figures(run_dir: str | Path = "results/runs/saqa_stgcn") -> list[Path]:
    """Draw every table-driven figure that has data behind it.

    Args:
        run_dir: The run whose ``per_item.csv`` supplies the risk-coverage curve.
            Parameterised rather than hard-coded so a caller can point at another
            run -- and so the tests can point at an empty directory and assert
            that a missing artefact produces no figure instead of a blank one.

    Returns:
        The paths written. Empty when no table has data behind it yet.
    """
    out = []
    for fn in (plot_method_comparison, plot_attribution_fidelity, plot_cost_curve,
               plot_efficiency, plot_data_efficiency):
        path = fn()
        if path is not None:
            out.append(path)
    rc = plot_risk_coverage(str(Path(run_dir) / "per_item.csv"))
    if rc is not None:
        out.append(rc)
    return out
