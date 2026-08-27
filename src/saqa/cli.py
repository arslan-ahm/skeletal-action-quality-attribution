"""Console entry point: ``saqa <command>``.

A thin dispatcher over the same pipeline functions the scripts use, so the
installed command and the repository scripts cannot diverge.

    saqa data      --config configs/base.yaml     # generate and summarise a dataset
    saqa train     --config configs/smoke.yaml    # train one configuration
    saqa bench                                    # cost benchmark, no training
    saqa compare   --set optim.epochs=10          # the headline comparison
    saqa tables                                   # print the committed tables
"""

from __future__ import annotations

import argparse
import sys


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None,
                        help="section.key=value overrides")


def cmd_data(args) -> int:
    from .config import load_config
    from .data.dataset import label_stats, split_indices
    from .pipelines import make_splits

    cfg = load_config(args.config, args.overrides)
    splits = make_splits(cfg)
    print(f"split={cfg.data.split}  sizes={splits.sizes()}")
    for name, subset in (("train", splits.train), ("val", splits.val),
                         ("test", splits.test)):
        stats = label_stats(subset)
        print(f"  {name:<6s} n={stats['n']:.0f} quality mean={stats['mean']:.4f} "
              f"sd={stats['sd']:.4f} clean={stats['frac_clean']:.3f}")
    import collections

    print("  actions:", dict(collections.Counter(splits.full.actions.tolist())))
    top = collections.Counter(splits.full.combinations.tolist()).most_common(8)
    print("  top defect combinations:", top)
    for mode in ("random", "subject", "combination"):
        sizes = {k: len(v) for k, v in split_indices(splits.full, mode).items()}
        print(f"  split[{mode:<11s}] {sizes}")
    return 0


def cmd_train(args) -> int:
    from .config import load_config
    from .pipelines import run_single

    cfg = load_config(args.config, args.overrides)
    result = run_single(cfg, verbose=True)
    for key in ("spearman", "kendall_tau", "relative_l2", "mae", "coverage",
                "mean_width", "error_auroc", "params", "train_seconds"):
        if key in result.metrics:
            print(f"  {key:<16s} {result.metrics[key]:.4f}")
    return 0


def cmd_bench(args) -> int:
    from .pipelines import dtw_cost_curve, efficiency_benchmark, ensure_dirs

    ensure_dirs()
    print(efficiency_benchmark(num_frames=args.frames).to_string(index=False))
    print()
    print(dtw_cost_curve().to_string(index=False))
    return 0


def cmd_compare(args) -> int:
    from .config import load_config
    from .pipelines import ensure_dirs, make_splits, method_comparison

    ensure_dirs()
    cfg = load_config(args.config, args.overrides)
    table, _, _ = method_comparison(cfg, make_splits(cfg))
    print(table.to_string(index=False))
    return 0


def cmd_tables(args) -> int:
    from .report import ALL, render_all

    if args.only:
        for name in args.only:
            print(f"### {name}\n\n{ALL[name]()}\n")
    else:
        print(render_all())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saqa", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="generate a dataset and summarise it")
    _add_common(p)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("train", help="train one configuration end to end")
    _add_common(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("bench", help="parameters, MACs, latency, DTW cost curve")
    p.add_argument("--frames", type=int, default=48)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("compare", help="the headline method comparison")
    _add_common(p)
    p.set_defaults(func=cmd_compare)

    from .report import ALL

    p = sub.add_parser("tables", help="print the committed results as Markdown")
    p.add_argument("--only", nargs="*", choices=sorted(ALL), default=None)
    p.set_defaults(func=cmd_tables)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
