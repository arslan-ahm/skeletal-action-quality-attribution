"""How many labelled sequences does each method need?

    python scripts/data_efficiency.py
    python scripts/data_efficiency.py --budgets 100 250 500 0

Reference-free quality assessment is labelling-expensive in reality -- every
training sequence needs a human score -- so the label-budget curve is the
practically important efficiency axis, not the parameter count. ``0`` means all
available training sequences. Writes ``results/tables/data_efficiency.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saqa.config import load_config  # noqa: E402
from saqa.pipelines import data_efficiency, ensure_dirs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    parser.add_argument("--budgets", nargs="*", type=int, default=[100, 250, 500, 0])
    parser.add_argument("--methods", nargs="*",
                        default=["saqa_stgcn", "tcn", "frame_average", "kinematic_gbr"])
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_config(args.config, args.overrides)
    table = data_efficiency(cfg, tuple(args.budgets), tuple(args.methods))
    print("\n=== Spearman against labelled-sequence budget ===")
    print(table.pivot(index="n_train", columns="method", values="spearman")
          .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
