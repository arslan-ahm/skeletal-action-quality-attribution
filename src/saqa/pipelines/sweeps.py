"""Ablations, split-regime comparison, data efficiency and the cost benchmark."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..baselines.dtw import dtw_similarity, frame_cost_matrix
from ..config import Config
from ..metrics.regression import spearman
from ..models.registry import build_model
from ..utils.complexity import count_macs, count_parameters, model_cost
from .core import make_splits, run_single
from .experiments import _with_seed, _write

#: Each entry changes **one** thing relative to the default configuration.
ABLATIONS: dict[str, dict[str, object]] = {
    "full": {},
    "partitions_uniform": {"model.partitions": "uniform"},
    "partitions_identity": {"model.partitions": "identity"},
    "no_edge_importance": {"model.edge_importance": False},
    "dense_temporal": {"model.separable": False},
    "temporal_kernel_3": {"model.temporal_kernel": 3},
    "head_regression": {"model.head": "regression"},
    "norm_group": {"model.norm": "group"},
    "uncertainty_heteroscedastic": {"model.uncertainty": "heteroscedastic"},
}


def run_ablations(
    cfg: Config, names: tuple[str, ...] | None = None, verbose: bool = True
) -> pd.DataFrame:
    """Train one variant per ablation entry on a shared dataset.

    The dataset is built once and shared, so an ablation difference cannot come
    from a different draw of sequences.
    """
    from ..config import load_config

    splits = make_splits(cfg)
    rows = []
    for name in names or tuple(ABLATIONS):
        overrides = [f"{k}={v}" for k, v in ABLATIONS[name].items()]
        variant = load_config(None, overrides)
        _copy_defaults(cfg, variant, skip=set(ABLATIONS[name]))
        if verbose:
            print(f"[ablate] {name}: {overrides or 'default'}")
        result = run_single(
            variant, name=f"ablation_{name}", splits=splits, verbose=False
        )
        rows.append({"variant": name, "change": ";".join(overrides) or "-",
                     **result.metrics})
    return _write(pd.DataFrame(rows), "ablation_components.csv")


def _copy_defaults(src: Config, dst: Config, skip: set[str]) -> None:
    """Copy every field of ``src`` into ``dst`` except the ablated ones."""
    for section in ("data", "model", "optim", "eval", "run"):
        s, d = getattr(src, section), getattr(dst, section)
        for key, value in vars(s).items():
            if f"{section}.{key}" in skip:
                continue
            setattr(d, key, value)


def split_comparison(cfg: Config, verbose: bool = True) -> pd.DataFrame:
    """Train the same model under all three splitting regimes.

    The gap between ``random`` and ``subject``/``combination`` is the leakage
    estimate. Reporting only the random split -- which is what a lot of synthetic
    benchmarks do -- would inflate every number in this repository.
    """
    rows = []
    for mode in ("random", "subject", "combination"):
        c = _with_seed(cfg, cfg.run.seed)
        c.data.split = mode
        if verbose:
            print(f"[splits] {mode}")
        result = run_single(c, name=f"split_{mode}", verbose=False)
        rows.append({"split": mode, **result.metrics})
    return _write(pd.DataFrame(rows), "split_comparison.csv")


def data_efficiency(
    cfg: Config,
    budgets: tuple[int, ...] = (100, 250, 500, 0),
    methods: tuple[str, ...] = ("saqa_stgcn", "tcn", "frame_average", "kinematic_gbr"),
    verbose: bool = True,
) -> pd.DataFrame:
    """Spearman as a function of the number of labelled training sequences.

    This is the axis that matters in practice: reference-free quality assessment
    needs a human-scored corpus, and scoring is the expensive part. A method that
    reaches a usable correlation from 100 labels is worth more than one that needs
    1000, whatever their asymptotes are.

    ``0`` in ``budgets`` means "all available training sequences".
    """
    from ..baselines.fitted import KinematicGBRBaseline
    from ..data.actions import ACTION_CLASSES

    rows = []
    for budget in budgets:
        c = _with_seed(cfg, cfg.run.seed)
        c.data.train_limit = int(budget)
        splits = make_splits(c)
        n_train = len(splits.train)
        for method in methods:
            if verbose:
                print(f"[data-eff] {method} @ n_train={n_train}")
            if method == "kinematic_gbr":
                model = KinematicGBRBaseline(
                    action_classes=ACTION_CLASSES, seed=c.run.seed
                ).fit(splits.train)
                score, _, _ = model.predict(splits.test)
                rho = spearman(splits.test.quality, score)
                rows.append({"method": method, "n_train": n_train, "spearman": rho,
                             "train_seconds": float("nan")})
                continue
            result = run_single(
                c, name=f"dataeff_{method}_{n_train}", architecture=method,
                splits=splits, verbose=False, save=False,
            )
            rows.append(
                {
                    "method": method,
                    "n_train": n_train,
                    "spearman": result.metrics["spearman"],
                    "train_seconds": result.metrics["train_seconds"],
                }
            )
    return _write(pd.DataFrame(rows), "data_efficiency.csv")


def efficiency_benchmark(
    architectures: tuple[str, ...] = (
        "saqa_stgcn_tiny", "saqa_stgcn", "stgcn_dense", "stgcn_reference",
        "stgcn_reference_large", "tcn", "lstm", "frame_average",
    ),
    num_frames: int = 48,
    num_joints: int = 17,
    warmup: int = 10,
    repeats: int = 30,
    threads: int = 2,
) -> pd.DataFrame:
    """Params, MACs and measured latency for every architecture.

    Ratios are taken against ``stgcn_reference_large`` -- the full architecture of
    Yan et al. (2018) -- because that is the thing the efficiency claim is about.
    """
    torch.set_num_threads(threads)
    rows = []
    for name in architectures:
        model = build_model(name)
        cost = model_cost(model, (3, num_frames, num_joints), (1, 8), warmup, repeats)
        rows.append({"architecture": name, **cost})
    df = pd.DataFrame(rows)
    ref = df[df["architecture"] == "stgcn_reference_large"]
    if not ref.empty:
        for col, out in (("params", "params_reduction"), ("macs", "macs_reduction"),
                         ("latency_bs1_ms", "latency_reduction")):
            df[out] = float(ref[col].iloc[0]) / df[col].replace(0, np.nan)
    return _write(df, "efficiency.csv")


def dtw_cost_curve(
    lengths: tuple[int, ...] = (24, 48, 96, 192, 384),
    num_joints: int = 17,
    repeats: int = 5,
    model_name: str = "saqa_stgcn",
    threads: int = 2,
) -> pd.DataFrame:
    """DTW's quadratic cost against the model's linear cost, across lengths.

    The reference approach pays ``O(T^2)`` in the dynamic program *and* ``O(T^2)``
    to build the frame cost matrix. The graph model is ``O(T)``. This measures
    both rather than asserting the exponents, and fits the empirical exponent by
    least squares on the log-log slope so a reader can check the claim.
    """
    torch.set_num_threads(threads)
    rng = np.random.default_rng(0)
    rows = []
    for t in lengths:
        q = rng.normal(0, 0.2, size=(t, num_joints, 3))
        r = rng.normal(0, 0.2, size=(t, num_joints, 3))
        t0 = time.perf_counter()
        for _ in range(repeats):
            dtw_similarity(q, r, "per_joint_norm", 0.15)
        dtw_ms = (time.perf_counter() - t0) / repeats * 1e3

        t0 = time.perf_counter()
        for _ in range(repeats):
            frame_cost_matrix(q, r, "per_joint_norm")
        cost_ms = (time.perf_counter() - t0) / repeats * 1e3

        model = build_model(model_name)
        model.eval()
        x = torch.zeros(1, 3, t, num_joints)
        with torch.no_grad():
            for _ in range(8):
                model(x)
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(x)
        model_ms = (time.perf_counter() - t0) / repeats * 1e3
        rows.append(
            {
                "num_frames": t,
                "dtw_total_ms": dtw_ms,
                "dtw_cost_matrix_ms": cost_ms,
                "model_ms": model_ms,
                "speedup": dtw_ms / model_ms if model_ms > 0 else float("nan"),
                "model_macs": float(count_macs(model, (3, t, num_joints))),
                "model_params": float(count_parameters(model)),
            }
        )
    df = pd.DataFrame(rows)
    log_t = np.log(df["num_frames"].to_numpy(dtype=float))
    for col, out in (("dtw_total_ms", "dtw_exponent"), ("model_ms", "model_exponent")):
        slope = np.polyfit(log_t, np.log(df[col].to_numpy(dtype=float)), 1)[0]
        df[out] = slope
    return _write(df, "cost_vs_length.csv")


def ensure_dirs() -> None:
    """Create the results tree so a fresh clone can run any script."""
    for sub in ("tables", "figures", "runs"):
        Path("results", sub).mkdir(parents=True, exist_ok=True)
