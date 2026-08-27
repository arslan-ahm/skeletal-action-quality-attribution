"""Component ablations, split-regime comparison and the seed-variance study.

    python scripts/run_ablations.py                # everything
    python scripts/run_ablations.py --only seeds   # just the seed study

Each ablation changes exactly one config key relative to the default, and they
all share one generated dataset, so a difference cannot come from a different
draw of sequences. Ablation deltas are placed against the seed-variance noise
scale, not reported raw -- a delta smaller than ``sqrt(2) * sd`` is not evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from saqa.config import load_config  # noqa: E402
from saqa.pipelines import (  # noqa: E402
    apply_noise_verdicts,
    ensure_dirs,
    run_ablations,
    seed_study,
    split_comparison,
)
from saqa.pipelines.experiments import _write  # noqa: E402

STAGES = ("seeds", "ablations", "splits")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    parser.add_argument("--only", nargs="*", choices=STAGES, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 7, 1337])
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_config(args.config, args.overrides)
    stages = tuple(args.only) if args.only else STAGES
    variance = None

    if "seeds" in stages:
        # Deliberately first. Every ablation delta below is meaningless without
        # knowing how far two runs of the *same* config drift apart.
        variance = seed_study(cfg, tuple(args.seeds))
        print("\n=== seed variance (same config, different seeds) ===")
        print(variance.to_string(index=False))

    if "ablations" in stages:
        table = run_ablations(cfg)
        print("\n=== ablations ===")
        cols = ["variant", "change", "spearman", "kendall_tau", "relative_l2", "mae",
                "coverage", "mean_width", "params", "train_seconds"]
        print(table[[c for c in cols if c in table.columns]].to_string(index=False))
        if variance is None and Path("results/tables/seed_variance.csv").exists():
            variance = pd.read_csv("results/tables/seed_variance.csv")
        if variance is not None:
            judged = apply_noise_verdicts(table, variance, "full", "spearman", "variant")
            _write(judged, "ablation_components.csv")
            print("\n=== ablation deltas against the run-to-run noise scale ===")
            print(judged[["variant", "spearman", "spearman_delta",
                          "spearman_ratio_to_noise", "spearman_verdict"]]
                  .to_string(index=False))

    if "splits" in stages:
        table = split_comparison(cfg)
        print("\n=== split regimes (the gap is the leakage estimate) ===")
        cols = ["split", "spearman", "kendall_tau", "relative_l2", "mae", "coverage",
                "n_train", "n_test"]
        print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
