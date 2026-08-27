"""Parameters, MACs and measured CPU latency. No training required.

    python scripts/benchmark_efficiency.py
    python scripts/benchmark_efficiency.py --frames 64 --repeats 50

Writes ``results/tables/efficiency.csv`` and ``results/tables/cost_vs_length.csv``.
The second is the one that matters against the reference approach: DTW is
quadratic in sequence length and the graph model is linear, and the table fits
the empirical exponent rather than asserting it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from saqa.pipelines import dtw_cost_curve, efficiency_benchmark, ensure_dirs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=10,
                        help="must be >= 8; below that the first-call overhead dominates")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-curve", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    pd.set_option("display.width", 200)

    df = efficiency_benchmark(
        num_frames=args.frames, warmup=args.warmup, repeats=args.repeats,
        threads=args.threads,
    )
    cols = ["architecture", "params", "macs", "latency_bs1_ms", "iqr_bs1_ms",
            "latency_bs8_ms", "macs_per_ms_bs1", "params_reduction",
            "macs_reduction", "latency_reduction"]
    print("\n=== architecture cost (ratios against stgcn_reference_large) ===")
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

    if not args.skip_curve:
        curve = dtw_cost_curve(threads=args.threads)
        print("\n=== cost against sequence length: DTW vs the graph model ===")
        print(curve.to_string(index=False))
        print(f"\nfitted log-log exponent: DTW {curve['dtw_exponent'].iloc[0]:.2f}, "
              f"model {curve['model_exponent'].iloc[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
