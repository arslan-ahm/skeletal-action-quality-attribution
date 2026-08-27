"""Train one configuration end to end and print its metrics.

    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/saqa_stgcn.yaml --set optim.epochs=30

Writes ``results/runs/<name>/{config.yaml,history.jsonl,per_item.csv,summary.json}``
and, unless ``--no-analysis`` is given, also reports the monotonicity-violation
rate, the ordinal rank-inconsistency rate and the attribution fidelity of the
trained model -- the three things the project claims, checked on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saqa.config import load_config  # noqa: E402
from saqa.pipelines import (  # noqa: E402
    attribution_completeness,
    attribution_fidelity,
    monotonicity_check,
    ordinal_consistency,
    run_single,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None,
                        help="section.key=value overrides")
    parser.add_argument("--no-analysis", action="store_true",
                        help="skip monotonicity and attribution analysis")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    result = run_single(cfg, verbose=not args.quiet)

    print(f"\n=== {result.name} ===")
    for key in ("spearman", "kendall_tau", "relative_l2", "mae", "coverage",
                "mean_width", "aurc", "error_auroc", "params", "train_seconds"):
        if key in result.metrics:
            print(f"  {key:<16s} {result.metrics[key]:.4f}")

    if not args.no_analysis:
        report, _, _ = monotonicity_check(result.model, cfg)
        print(f"  monotonicity violation rate  {report['violation_rate']:.4f} "
              f"over {report['n_ladders']:.0f} ladders")
        print(f"  ordinal rank inconsistency   "
              f"{ordinal_consistency(result.model, result.splits.test)['rank_inconsistency']:.4f}")
        fidelity = attribution_fidelity(
            result.model, result.splits.test, cfg,
            architecture=cfg.model.architecture,
        )
        for method, scores in fidelity.items():
            if "joint_iou" in scores:
                print(f"  attribution[{method:<34s}] joint IoU {scores['joint_iou']:.4f} "
                      f"top1 {scores.get('joint_top1_hit', float('nan')):.4f}")
        out = Path(cfg.run.out_dir) / result.name / "analysis.json"
        out.write_text(
            json.dumps(
                {"monotonicity": report,
                 "completeness": attribution_completeness(
                     result.model, result.splits.test, cfg),
                 "attribution": fidelity},
                indent=2, default=float),
            encoding="utf-8", newline="\n",
        )
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
