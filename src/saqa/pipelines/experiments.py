"""The experiment matrix: comparison, ablations, seeds, splits, data efficiency.

Every function here writes a CSV into ``results/tables/`` and returns the same
frame, so a notebook and a script get identical numbers from identical code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data.dataset import SequenceDataset
from ..engine.trainer import predict
from ..metrics.regression import per_item_absolute_error, regression_summary, spearman
from ..metrics.stats import (
    Comparison,
    bootstrap_metric_difference,
    compare,
    holm_bonferroni,
    noise_scale,
    verdict,
)
from ..models.registry import build_model
from .core import RunResult, Splits, evaluate_predictions, make_splits, run_single

TABLES = Path("results/tables")

#: The neural arms of the headline comparison. ``stgcn_reference_large`` is not
#: here: it is benchmarked for cost only (see the registry docstring).
NEURAL_ARMS: tuple[str, ...] = (
    "saqa_stgcn",
    "stgcn_dense",
    "stgcn_reference",
    "tcn",
    "lstm",
    "frame_average",
)


def _write(df: pd.DataFrame, name: str) -> pd.DataFrame:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False, lineterminator="\n")
    return df


def fit_baselines(
    splits: Splits, cfg: Config, dtw_sweep: bool = True
) -> dict[str, dict[str, np.ndarray]]:
    """Fit every non-neural baseline and return their test predictions.

    The DTW arm is swept over distance function, band and reference source when
    ``dtw_sweep`` is on, and **the best configuration by validation Spearman is
    promoted to the headline arm**. Selecting it on validation rather than test is
    the only way the sweep is a fair advantage rather than a test-set peek.
    """
    from ..baselines.fitted import DTWBaseline, KinematicGBRBaseline
    from ..data.actions import ACTION_CLASSES
    from .core import generator_config

    gcfg = generator_config(cfg)
    out: dict[str, dict[str, np.ndarray]] = {}

    gbr = KinematicGBRBaseline(action_classes=ACTION_CLASSES, seed=cfg.run.seed,
                               level=cfg.eval.interval_level).fit(splits.train)
    score, lo, hi = gbr.predict(splits.test)
    out["kinematic_gbr"] = {"score": score, "lower": lo, "upper": hi}
    out["kinematic_gbr__importance"] = gbr.feature_importance()  # type: ignore[assignment]

    configs = (
        [
            dict(distance=d, band=b, reference_source=r)
            for d in ("l2", "per_joint_norm", "velocity_cosine")
            for b in (0.15, None)
            for r in ("canonical", "exemplar")
        ]
        if dtw_sweep
        else [dict()]
    )
    rows, best, best_rho = [], None, -np.inf
    for kw in configs:
        model = DTWBaseline(generator=gcfg, level=cfg.eval.interval_level, **kw).fit(
            splits.train, seed=cfg.run.seed
        )
        val_score, _, _ = model.predict(splits.val)
        rho_val = spearman(splits.val.quality, val_score)
        test_score, tlo, thi = model.predict(splits.test)
        rows.append(
            {
                "config": model.name,
                "val_spearman": rho_val,
                "test_spearman": spearman(splits.test.quality, test_score),
                **{f"test_{k}": v for k, v in
                   regression_summary(splits.test.quality, test_score).items()},
            }
        )
        if np.isfinite(rho_val) and rho_val > best_rho:
            best_rho, best = rho_val, (model.name, test_score, tlo, thi)
    if rows:
        _write(pd.DataFrame(rows), "dtw_sweep.csv")

    # The no-alignment control, always reported: it isolates what the warping buys.
    fw = DTWBaseline(generator=gcfg, aligned=False, level=cfg.eval.interval_level).fit(
        splits.train, seed=cfg.run.seed
    )
    s, l, u = fw.predict(splits.test)
    out["framewise_reference"] = {"score": s, "lower": l, "upper": u}

    assert best is not None
    name, score, lo, hi = best
    out["dtw_reference"] = {"score": score, "lower": lo, "upper": hi, "config": name}
    return out


def method_comparison(
    cfg: Config, splits: Splits | None = None, arms: tuple[str, ...] = NEURAL_ARMS,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict[str, RunResult], dict[str, dict]]:
    """Train every neural arm and fit every baseline on one shared dataset.

    Returns:
        ``(table, runs, baselines)``.
    """
    sp = splits if splits is not None else make_splits(cfg)
    runs: dict[str, RunResult] = {}
    rows: list[dict[str, object]] = []

    for arch in arms:
        if verbose:
            print(f"[compare] training {arch}")
        result = run_single(cfg, name=arch, architecture=arch, splits=sp, verbose=False)
        runs[arch] = result
        rows.append({"method": arch, "family": "neural", **result.metrics})

    # An untrained network with the same architecture. It is not a strawman: a
    # random projection of movement amplitude already correlates with defect
    # severity, so this arm can score a non-trivial Spearman, and any trained
    # model that does not clear it has learned nothing that the architecture and
    # the input statistics did not already provide. Cheap to run and easy to
    # omit, which is exactly why it is here.
    torch.manual_seed(cfg.run.seed + 999)
    untrained = build_model(
        arms[0], head=cfg.model.head, uncertainty=cfg.model.uncertainty,
        num_bins=cfg.model.num_bins,
    )
    untrained_pred = predict(untrained, sp.test.coords, cfg.data.batch_size)
    rows.append(
        {
            "method": "untrained_stgcn",
            "family": "control",
            **evaluate_predictions(
                sp.test, untrained_pred["score"], untrained_pred.get("lower"),
                untrained_pred.get("upper"), level=cfg.eval.interval_level,
            ),
            "params": float(sum(p.numel() for p in untrained.parameters())),
        }
    )

    baselines = fit_baselines(sp, cfg)
    baselines["untrained_stgcn"] = untrained_pred
    for name in ("kinematic_gbr", "dtw_reference", "framewise_reference"):
        pred = baselines[name]
        rows.append(
            {
                "method": name,
                "family": "baseline",
                **evaluate_predictions(
                    sp.test, pred["score"], pred.get("lower"), pred.get("upper"),
                    level=cfg.eval.interval_level,
                ),
                "params": float("nan"),
            }
        )
    table = _write(pd.DataFrame(rows), "method_comparison.csv")
    return table, runs, baselines


def statistical_tests(
    test: SequenceDataset,
    predictions: dict[str, np.ndarray],
    reference: str = "dtw_reference",
    n_resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired tests of every method against the reference approach.

    Two families, kept separate because they answer different questions and
    mixing them into one multiplicity correction would be wrong:

    * **per-sequence absolute error** -- Wilcoxon signed-rank plus a paired
      bootstrap, Holm-corrected across methods;
    * **Spearman**, a set-level statistic with no per-item value, tested by the
      paired bootstrap of :func:`~saqa.metrics.stats.bootstrap_metric_difference`.
    """
    truth = np.asarray(test.quality, dtype=np.float64)
    comparisons: list[Comparison] = []
    for name, score in predictions.items():
        if name == reference:
            continue
        comparisons.append(
            compare(
                per_item_absolute_error(truth, score),
                per_item_absolute_error(truth, predictions[reference]),
                f"{name}.abs_error",
                f"{reference}.abs_error",
                n_resamples,
                seed,
            )
        )
    holm_bonferroni(comparisons)

    rows = [{"family": "abs_error", **c.to_dict()} for c in comparisons]
    for name, score in predictions.items():
        if name == reference:
            continue
        iv = bootstrap_metric_difference(
            truth, score, predictions[reference], spearman, n_resamples, seed=seed
        )
        rows.append(
            {
                "family": "spearman",
                "name_a": f"{name}.spearman",
                "name_b": f"{reference}.spearman",
                "mean_a": spearman(truth, score),
                "mean_b": spearman(truth, predictions[reference]),
                "difference": iv.estimate,
                "ci_lower": iv.lower,
                "ci_upper": iv.upper,
                "p_value": float("nan"),
                "p_adjusted": None,
                "effect_size": float("nan"),
                "n": iv.n,
                "significant": bool(iv.lower > 0 or iv.upper < 0),
            }
        )
    return _write(pd.DataFrame(rows), "statistical_tests.csv")


def seed_study(
    cfg: Config, seeds: tuple[int, ...] = (0, 7, 1337), architecture: str = "saqa_stgcn",
    verbose: bool = True
) -> pd.DataFrame:
    """Same configuration, several training seeds. Run this before any claim.

    The seed changes the model initialisation, the batch order **and** the split,
    which is the conservative choice: it measures the variation a reader would see
    reproducing the pipeline from scratch, not just the variation from re-rolling
    the weights.
    """
    rows = []
    for seed in seeds:
        c = Config(**{**cfg.to_dict()})
        c = _with_seed(cfg, seed)
        if verbose:
            print(f"[seeds] seed {seed}")
        result = run_single(c, name=f"seed_{seed}", architecture=architecture, verbose=False)
        rows.append({"seed": seed, "architecture": architecture, **result.metrics})
    df = pd.DataFrame(rows)

    summary = []
    for metric in ("spearman", "kendall_tau", "relative_l2", "mae", "coverage",
                   "mean_width", "aurc", "error_auroc"):
        if metric not in df.columns:
            continue
        vals = df[metric].to_numpy(dtype=np.float64)
        summary.append(
            {
                "metric": metric,
                "mean": float(np.nanmean(vals)),
                "sd": float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1
                else float("nan"),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "range": float(np.nanmax(vals) - np.nanmin(vals)),
                "noise_scale": noise_scale(vals),
                "n_runs": int(np.isfinite(vals).sum()),
            }
        )
    _write(df, "seed_runs.csv")
    return _write(pd.DataFrame(summary), "seed_variance.csv")


def _with_seed(cfg: Config, seed: int) -> Config:
    from ..config import load_config

    c = load_config(None, None)
    _copy_into(cfg, c)
    c.run.seed = seed
    return c


def _copy_into(src: Config, dst: Config) -> None:
    for section in ("data", "model", "optim", "eval", "run"):
        s, d = getattr(src, section), getattr(dst, section)
        for key, value in vars(s).items():
            setattr(d, key, value)


def apply_noise_verdicts(
    table: pd.DataFrame, variance: pd.DataFrame, baseline_row: str, metric: str = "spearman",
    key: str = "method"
) -> pd.DataFrame:
    """Attach the run-to-run verdict to every row of a comparison table."""
    scales = dict(zip(variance["metric"], variance["noise_scale"], strict=True))
    scale = scales.get(metric, float("nan"))
    base = float(table.loc[table[key] == baseline_row, metric].iloc[0])
    out = table.copy()
    out[f"{metric}_delta"] = out[metric] - base
    out[f"{metric}_noise_scale"] = scale
    out[f"{metric}_ratio_to_noise"] = np.abs(out[f"{metric}_delta"]) / scale
    out[f"{metric}_verdict"] = [verdict(d, scale) for d in out[f"{metric}_delta"]]
    return out
