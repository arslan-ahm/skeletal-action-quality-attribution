"""Data assembly and the single-run training/evaluation pipeline.

Everything a script does goes through here, so that the tests can reach it and so
that no experiment can quietly acquire a different schedule than another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data.dataset import SequenceDataset, build_dataset, split_indices
from ..data.generator import GeneratorConfig
from ..engine.trainer import predict, train_model
from ..metrics.regression import per_group_summary, regression_summary
from ..metrics.uncertainty import error_detection_auroc, excess_aurc, interval_summary
from ..models.registry import build_model
from ..utils.seed import seed_everything


def generator_config(cfg: Config) -> GeneratorConfig:
    """Project the experiment config onto the generator's own config."""
    d = cfg.data
    return GeneratorConfig(
        num_frames=d.num_frames,
        num_subjects=d.num_subjects,
        max_degradations=d.max_degradations,
        severity_min=d.severity_min,
        severity_max=d.severity_max,
        noise_std=d.noise_std,
        jitter_std=d.jitter_std,
        dropout_prob=d.dropout_prob,
        dropout_max_frames=d.dropout_max_frames,
        clean_fraction=d.clean_fraction,
    )


#: Architectures whose channel budget, stride pattern and separability are part
#: of their identity, and must not be overridden by the generic ``model.*``
#: config section. Ablations of those switches operate on ``saqa_stgcn``.
SHAPE_OWNING_ARCHITECTURES: frozenset[str] = frozenset(
    {"stgcn_dense", "stgcn_reference", "stgcn_reference_large", "saqa_stgcn_tiny"}
)


@dataclass
class Splits:
    """The three splits plus the index arrays that produced them."""

    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    indices: dict[str, np.ndarray]
    full: SequenceDataset

    def sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in
                (("train", self.train), ("val", self.val), ("test", self.test))}


def make_splits(cfg: Config, split: str | None = None) -> Splits:
    """Build the dataset once and split it under the requested regime.

    ``data.train_limit`` truncates the *training* split only, and it truncates a
    seeded permutation rather than the head of the array, so the data-efficiency
    curve's smaller budgets are random subsets rather than whichever sequences
    happened to be generated first (which would correlate with subject id, since
    subjects cycle with the index).
    """
    gcfg = generator_config(cfg)
    data = build_dataset(cfg.data.num_sequences, gcfg)
    idx = split_indices(
        data,
        mode=split or cfg.data.split,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.run.seed,
        test_fraction=cfg.data.test_fraction,
    )
    train_idx = idx["train"]
    if cfg.data.train_limit and cfg.data.train_limit < train_idx.size:
        rng = np.random.default_rng(cfg.run.seed + 991)
        train_idx = np.sort(rng.permutation(train_idx)[: cfg.data.train_limit])
        idx = {**idx, "train": train_idx}
    return Splits(
        train=data.subset(train_idx),
        val=data.subset(idx["val"]),
        test=data.subset(idx["test"]),
        indices=idx,
        full=data,
    )


@dataclass
class RunResult:
    """One trained model, its predictions and its metrics.

    Attributes:
        name: Run name, used for the output directory.
        model: The trained model, with the selected checkpoint loaded.
        pred: ``{"score", "lower", "upper"}`` on the test split.
        metrics: Flat dict of scalars written to ``summary.json``.
        history: Per-epoch records.
        splits: The splits it was trained on.
        wall_seconds: Training wall-clock.
    """

    name: str
    model: torch.nn.Module
    pred: dict[str, np.ndarray]
    metrics: dict[str, float]
    history: list[dict[str, float]]
    splits: Splits
    wall_seconds: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


def evaluate_predictions(
    data: SequenceDataset,
    score: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    level: float = 0.90,
    prefix: str = "",
) -> dict[str, float]:
    """Every score-level metric for one set of predictions.

    Confidence for the risk-coverage curve is ``-interval_width`` when an interval
    is available. A model with no uncertainty head gets no risk-coverage numbers
    at all rather than a fabricated confidence -- ranking by the predicted score
    itself would silently measure "does the model abstain on low scores", which is
    a different and much easier question.
    """
    truth = np.asarray(data.quality, dtype=np.float64)
    out: dict[str, float] = {f"{prefix}{k}": v for k, v in
                             regression_summary(truth, score).items()}
    err = np.abs(np.asarray(score, dtype=np.float64) - truth)
    if lower is not None and upper is not None:
        out.update({f"{prefix}{k}": v for k, v in
                    interval_summary(truth, lower, upper, score, nominal=level).items()})
        confidence = -(np.asarray(upper) - np.asarray(lower))
        out.update({f"{prefix}{k}": v for k, v in excess_aurc(err, confidence).items()})
        out[f"{prefix}error_auroc"] = error_detection_auroc(err, confidence)
    return out


def evaluate_groups(data: SequenceDataset, score: np.ndarray) -> dict[str, float]:
    """Per-action-class and per-defect-combination breakdowns."""
    truth = np.asarray(data.quality, dtype=np.float64)
    out: dict[str, float] = {}
    for key, groups in (("action", data.actions), ("combination", data.combinations)):
        for metric in ("spearman", "mae"):
            for name, value in per_group_summary(truth, score, groups, metric).items():
                out[f"{key}__{metric}__{name}" if not name.endswith("__n")
                    else f"{key}__n__{name[:-3]}"] = value
    return out


def run_single(
    cfg: Config,
    name: str | None = None,
    architecture: str | None = None,
    splits: Splits | None = None,
    save: bool = True,
    verbose: bool = True,
    **trunk_kwargs,
) -> RunResult:
    """Train and evaluate one neural configuration end to end.

    Args:
        cfg: Resolved config.
        name: Run name; defaults to ``cfg.run.name``.
        architecture: Overrides ``cfg.model.architecture``.
        splits: Reuse pre-built splits (so a comparison shares exactly one
            dataset across methods instead of regenerating it per method).
        save: Write ``config.yaml``, ``history.jsonl``, ``per_item.csv`` and
            ``summary.json`` into ``results/runs/<name>/``.
        verbose: Per-epoch printing.
        **trunk_kwargs: Forwarded to the trunk builder, for ablations.

    Returns:
        A :class:`RunResult`.
    """
    run_name = name or cfg.run.name
    seed_everything(cfg.run.seed, cfg.run.torch_threads)
    sp = splits if splits is not None else make_splits(cfg)

    arch = architecture or cfg.model.architecture
    kwargs = dict(trunk_kwargs)
    kwargs.setdefault("temporal_kernel", cfg.model.temporal_kernel)
    kwargs.setdefault("partitions", cfg.model.partitions)
    kwargs.setdefault("edge_importance", cfg.model.edge_importance)
    kwargs.setdefault("dropout", cfg.model.dropout)
    kwargs.setdefault("norm", cfg.model.norm)
    if arch not in SHAPE_OWNING_ARCHITECTURES:
        # These keys define what the named variants *are*. Letting a config's
        # generic model.* section reach them silently collapsed `stgcn_dense`
        # and `stgcn_reference` into `saqa_stgcn` -- three identical arms in the
        # comparison table, with identical attribution scores, which is how the
        # bug was caught. Asserted by
        # tests/test_pipelines.py::test_named_variants_keep_their_own_shape.
        kwargs.setdefault("separable", cfg.model.separable)
        if cfg.model.channels:
            kwargs.setdefault("channels", tuple(cfg.model.channels))
        if cfg.model.strides:
            kwargs.setdefault("strides", tuple(cfg.model.strides))

    if arch in ("tcn", "lstm", "frame_average"):
        # Graph-free trunks accept none of the graph switches. Passing them would
        # be an error rather than a no-op, which is the right behaviour: an
        # ablation config that silently did nothing would be worse than a crash.
        for key in ("partitions", "separable", "edge_importance", "channels", "strides"):
            kwargs.pop(key, None)
        if arch == "lstm":
            kwargs.pop("temporal_kernel", None)
        if arch == "frame_average":
            kwargs.pop("temporal_kernel", None)
            kwargs.pop("norm", None)

    model = build_model(
        arch,
        head=cfg.model.head,
        uncertainty=cfg.model.uncertainty,
        num_bins=cfg.model.num_bins,
        uncertainty_weight=cfg.model.uncertainty_weight,
        **kwargs,
    )

    out_dir = Path(cfg.run.out_dir) / run_name
    log_path = out_dir / "history.jsonl" if save else None
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(out_dir / "config.yaml")

    state = train_model(model, sp.train, sp.val, cfg, log_path=log_path, verbose=verbose)
    pred = predict(model, sp.test.coords, cfg.data.batch_size)

    metrics = evaluate_predictions(
        sp.test, pred["score"], pred.get("lower"), pred.get("upper"),
        level=cfg.eval.interval_level,
    )
    metrics.update(
        {
            "best_epoch": float(state.best_epoch),
            "best_val_spearman": float(state.best_val_spearman),
            "train_seconds": float(state.wall_seconds),
            "n_train": float(len(sp.train)),
            "n_val": float(len(sp.val)),
            "n_test": float(len(sp.test)),
            "params": float(sum(p.numel() for p in model.parameters())),
        }
    )
    result = RunResult(run_name, model, pred, metrics, state.history, sp, state.wall_seconds)
    if save:
        _write_run(out_dir, result, cfg)
    return result


def _write_run(out_dir: Path, result: RunResult, cfg: Config) -> None:
    """Persist per-item predictions and the metric summary."""
    import pandas as pd

    test = result.splits.test
    rows = {
        "index": [s.index for s in test.samples],
        "action": list(test.actions),
        "subject": list(test.subjects),
        "combination": list(test.combinations),
        "quality": np.asarray(test.quality, dtype=np.float64),
        "score": result.pred["score"],
        "abs_error": np.abs(result.pred["score"] - np.asarray(test.quality)),
    }
    if "lower" in result.pred:
        rows["lower"] = result.pred["lower"]
        rows["upper"] = result.pred["upper"]
        rows["width"] = result.pred["upper"] - result.pred["lower"]
    pd.DataFrame(rows).to_csv(out_dir / "per_item.csv", index=False, lineterminator="\n")

    summary = {**result.metrics, **evaluate_groups(test, result.pred["score"])}
    (out_dir / "summary.json").write_text(
        json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in summary.items()}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    del cfg
