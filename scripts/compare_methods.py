"""The headline experiment: every method, one dataset, one training loop.

    python scripts/compare_methods.py
    python scripts/compare_methods.py --set optim.epochs=30 data.num_sequences=2000

Trains the six neural arms, fits the DTW and kinematic baselines (with the DTW
configuration selected on *validation*), then writes:

* ``results/tables/method_comparison.csv``  -- every metric for every method
* ``results/tables/statistical_tests.csv``  -- paired tests against the reference
* ``results/tables/attribution_fidelity.csv`` -- explanations scored against truth
* ``results/tables/monotonicity.csv``       -- violation rates per method
* ``results/tables/uncertainty.csv``        -- coverage, width, risk-coverage
* ``results/tables/per_action.csv``         -- per-class breakdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from saqa.baselines.dtw import per_joint_dtw_deviation  # noqa: E402
from saqa.config import load_config  # noqa: E402
from saqa.data.generator import reference_sequence  # noqa: E402
from saqa.metrics.attribution import fidelity_summary  # noqa: E402
from saqa.metrics.regression import per_group_summary  # noqa: E402
from saqa.pipelines import (  # noqa: E402
    attribution_completeness,
    attribution_fidelity,
    ensure_dirs,
    generator_config,
    make_splits,
    method_comparison,
    monotonicity_check,
    ordinal_consistency,
    statistical_tests,
)
from saqa.pipelines.experiments import _write  # noqa: E402


def dtw_attribution_fidelity(test, cfg) -> dict[str, float]:
    """Score the reference approach's own explanation against the same truth.

    The DTW baseline's per-joint deviation along the optimal path is the best
    explanation that formulation can produce, so the attribution comparison is
    against a real alternative rather than against nothing.
    """
    gcfg = generator_config(cfg)
    rows = []
    for sample in test.samples:
        ref = reference_sequence(sample.action, gcfg)
        rows.append(per_joint_dtw_deviation(sample.positions, ref))
    return fidelity_summary(np.stack(rows), np.asarray(test.joint_truth), "joint")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--skip-attribution", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_config(args.config, args.overrides)
    splits = make_splits(cfg)
    print(f"dataset: {splits.sizes()}  split={cfg.data.split}")

    arms = tuple(args.arms) if args.arms else None
    table, runs, baselines = (
        method_comparison(cfg, splits, arms) if arms
        else method_comparison(cfg, splits)
    )
    print("\n=== method comparison ===")
    show = ["method", "family", "spearman", "kendall_tau", "relative_l2", "mae",
            "coverage", "mean_width", "aurc", "error_auroc", "params"]
    print(table[[c for c in show if c in table.columns]].to_string(index=False))

    predictions = {name: r.pred["score"] for name, r in runs.items()}
    for name in ("kinematic_gbr", "dtw_reference", "framewise_reference",
                 "untrained_stgcn"):
        predictions[name] = baselines[name]["score"]
    tests = statistical_tests(splits.test, predictions, "dtw_reference",
                              cfg.eval.bootstrap, cfg.run.seed)
    print("\n=== paired tests vs the reference DTW approach ===")
    print(tests[["family", "name_a", "mean_a", "mean_b", "difference", "ci_lower",
                 "ci_upper", "p_adjusted", "significant"]].to_string(index=False))

    # --- monotonicity, one row per neural method ---------------------------
    mono_rows = []
    for name, result in runs.items():
        report, _, _ = monotonicity_check(result.model, cfg)
        mono_rows.append({"method": name, **report,
                          **ordinal_consistency(result.model, splits.test)})
    mono = _write(pd.DataFrame(mono_rows), "monotonicity.csv")
    print("\n=== monotonicity ===")
    print(mono[["method", "violation_rate", "ladders_with_any_violation",
                "max_increase", "rank_inconsistency"]].to_string(index=False))

    # --- uncertainty --------------------------------------------------------
    unc_cols = ["method", "coverage", "nominal", "coverage_gap", "mean_width",
                "median_width", "crossing_rate", "width_ratio_wrong_right",
                "aurc", "aurc_oracle", "e_aurc", "error_auroc"]
    _write(table[[c for c in unc_cols if c in table.columns]], "uncertainty.csv")

    # --- per-action breakdown ----------------------------------------------
    per_action = []
    for name, score in predictions.items():
        row = {"method": name}
        row.update(per_group_summary(splits.test.quality, score, splits.test.actions,
                                     "spearman"))
        per_action.append(row)
    pa = _write(pd.DataFrame(per_action), "per_action.csv")
    print("\n=== Spearman per action class ===")
    print(pa.to_string(index=False))

    if not args.skip_attribution:
        limit = min(cfg.eval.attribution_limit, len(splits.test))
        attr_data = splits.test.subset(np.arange(limit))
        print(f"\nattribution fidelity on {limit} of {len(splits.test)} test sequences")
        rows = []
        for name, result in runs.items():
            for method, scores in attribution_fidelity(
                result.model, attr_data, cfg, architecture=name
            ).items():
                rows.append({"model": name, "attribution": method, **scores})
        rows.append({"model": "dtw_reference", "attribution": "per_joint_dtw_deviation",
                     **{f"joint_{k}": v for k, v in
                        dtw_attribution_fidelity(attr_data, cfg).items()}})
        fid = _write(pd.DataFrame(rows), "attribution_fidelity.csv")
        print("\n=== attribution fidelity (joint) ===")
        cols = ["model", "attribution", "joint_precision", "joint_recall", "joint_iou",
                "joint_rank_corr", "joint_top1_hit", "frame_localisation_error"]
        print(fid[[c for c in cols if c in fid.columns]].to_string(index=False))

        best = runs.get("saqa_stgcn")
        if best is not None:
            comp = attribution_completeness(best.model, attr_data, cfg)
            _write(pd.DataFrame([comp]), "ig_completeness.csv")
            print(f"\nIG completeness residual: mean {comp['completeness_mean_abs']:.2e}, "
                  f"max {comp['completeness_max_abs']:.2e} at {comp['ig_steps']:.0f} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
