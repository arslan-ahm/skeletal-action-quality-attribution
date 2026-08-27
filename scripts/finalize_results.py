"""Place every headline difference against the run-to-run noise scale.

    python scripts/finalize_results.py

Reads ``results/tables/method_comparison.csv`` and
``results/tables/seed_variance.csv`` and writes
``results/tables/method_verdicts.csv``: for each method, its Spearman gap to the
reference DTW baseline expressed as a multiple of ``sqrt(2) * sd`` from the seed
study, and the resulting verdict.

This is deliberately a separate step run *after* the matrix. A difference between
two single runs is not evidence until it has been divided by the noise, and
keeping the division in its own script makes it impossible to read the raw deltas
first and the verdicts second.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from saqa.metrics.stats import verdict  # noqa: E402
from saqa.pipelines.experiments import _write  # noqa: E402

TABLES = ROOT / "results" / "tables"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="dtw_reference")
    parser.add_argument("--metrics", nargs="*",
                        default=["spearman", "kendall_tau", "relative_l2", "mae",
                                 "coverage", "mean_width", "error_auroc"])
    args = parser.parse_args(argv)

    methods = TABLES / "method_comparison.csv"
    seeds = TABLES / "seed_variance.csv"
    if not methods.exists():
        print("method_comparison.csv is missing; run scripts/compare_methods.py")
        return 1
    table = pd.read_csv(methods)
    scales = {}
    if seeds.exists():
        var = pd.read_csv(seeds)
        scales = dict(zip(var["metric"], var["noise_scale"], strict=True))
    else:
        print("warning: seed_variance.csv is missing; verdicts will read 'unknown'")

    if args.reference not in set(table["method"]):
        print(f"reference {args.reference!r} is not in the comparison table")
        return 1

    rows = []
    for metric in args.metrics:
        if metric not in table.columns:
            continue
        base = float(table.loc[table["method"] == args.reference, metric].iloc[0])
        scale = scales.get(metric, float("nan"))
        for _, row in table.iterrows():
            if row["method"] == args.reference:
                continue
            delta = float(row[metric]) - base
            rows.append(
                {
                    "metric": metric,
                    "method": row["method"],
                    "value": float(row[metric]),
                    "reference": args.reference,
                    "reference_value": base,
                    "delta": delta,
                    "noise_scale": scale,
                    "ratio_to_noise": abs(delta) / scale if scale and scale == scale else
                    float("nan"),
                    "verdict": verdict(delta, scale),
                }
            )
    out = _write(pd.DataFrame(rows), "method_verdicts.csv")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
