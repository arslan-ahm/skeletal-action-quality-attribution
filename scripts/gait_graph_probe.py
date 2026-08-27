"""Targeted follow-up: does the graph help on the one class where it should?

    python scripts/gait_graph_probe.py

The headline comparison showed the graph models scoring far higher than the
graph-free ones on ``gait`` specifically -- the only action class whose quality
signal lives in *inter-limb coordination* (the legs swing in anti-phase, the arms
counter-swing). That is precisely the structure a skeleton graph encodes and a
flattened temporal model does not.

With one training run per arm, that observation is an n of 1. This script runs
several seeds of the graph model and of the graph-free temporal CNN and compares
their **per-action** Spearman distributions, so the gait gap can be placed
against the run-to-run spread instead of being asserted from a single pair of
runs. The temporal CNN is cheap (tens of seconds per run), which is what makes
this affordable at all.

Writes ``results/tables/gait_graph_probe.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from saqa.config import load_config  # noqa: E402
from saqa.metrics.regression import per_group_summary  # noqa: E402
from saqa.metrics.stats import noise_scale, verdict  # noqa: E402
from saqa.pipelines import ensure_dirs, run_single  # noqa: E402
from saqa.pipelines.experiments import _with_seed, _write  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 7, 1337])
    parser.add_argument("--arms", nargs="*", default=["saqa_stgcn", "tcn"])
    args = parser.parse_args(argv)

    ensure_dirs()
    base = load_config(args.config, args.overrides)
    rows = []
    for arch in args.arms:
        for seed in args.seeds:
            cfg = _with_seed(base, seed)
            print(f"[probe] {arch} seed {seed}", flush=True)
            result = run_single(cfg, name=f"probe_{arch}_{seed}", architecture=arch,
                                save=False, verbose=False)
            per_action = per_group_summary(
                result.splits.test.quality, result.pred["score"],
                result.splits.test.actions, "spearman",
            )
            row = {"architecture": arch, "seed": seed,
                   "overall_spearman": result.metrics["spearman"]}
            row.update({k: v for k, v in per_action.items() if not k.endswith("__n")})
            rows.append(row)
    runs = _write(pd.DataFrame(rows), "gait_graph_probe_runs.csv")

    metrics = [c for c in runs.columns if c not in ("architecture", "seed")]
    summary = []
    for metric in metrics:
        by_arch = {a: sub[metric].to_numpy(dtype=float)
                   for a, sub in runs.groupby("architecture")}
        if len(by_arch) != 2:
            continue
        (a_name, a_vals), (b_name, b_vals) = sorted(by_arch.items())
        pooled = np.concatenate([a_vals - np.nanmean(a_vals),
                                 b_vals - np.nanmean(b_vals)])
        scale = noise_scale(pooled)
        delta = float(np.nanmean(a_vals) - np.nanmean(b_vals))
        summary.append(
            {
                "metric": metric,
                f"{a_name}_mean": float(np.nanmean(a_vals)),
                f"{b_name}_mean": float(np.nanmean(b_vals)),
                "delta": delta,
                "pooled_noise_scale": scale,
                "ratio_to_noise": abs(delta) / scale if scale and scale == scale
                else float("nan"),
                "verdict": verdict(delta, scale),
                "n_seeds": int(len(a_vals)),
            }
        )
    out = _write(pd.DataFrame(summary), "gait_graph_probe.csv")
    print(runs.round(4).to_string(index=False))
    print()
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
