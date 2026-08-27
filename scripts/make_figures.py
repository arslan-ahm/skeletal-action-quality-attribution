"""Redraw every figure from artefacts already on disk.

    python scripts/make_figures.py

Figures read the committed CSVs rather than live objects, so a figure and the
number quoted next to it in the documentation cannot drift apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from saqa.config import load_config  # noqa: E402
from saqa.data.generator import generate_sample  # noqa: E402
from saqa.pipelines import ensure_dirs, generator_config  # noqa: E402
from saqa.viz import all_figures, plot_skeleton_frames  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--sample", type=int, default=3,
                        help="generator index for the skeleton panel")
    args = parser.parse_args(argv)

    ensure_dirs()
    written = all_figures()

    cfg = load_config(args.config)
    gcfg = generator_config(cfg)
    # Pick a sequence that actually carries a defect; a clean one makes a dull
    # figure and, worse, has no attribution ground truth to show against.
    idx = args.sample
    for candidate in range(args.sample, args.sample + 60):
        if generate_sample(candidate, gcfg).degradations:
            idx = candidate
            break
    sample = generate_sample(idx, gcfg)
    written.append(plot_skeleton_frames(
        sample, frames=tuple(np.linspace(0, cfg.data.num_frames - 1, 4).astype(int))
    ))

    for path in written:
        print(f"  wrote {path}")
    if not written:
        print("no figures written: run the experiments first (make all)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
