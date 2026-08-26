"""One training loop for every neural architecture and head.

Six architectures and three heads share this loop, and that is a deliberate
credibility decision rather than a convenience. If each architecture had its own
script, a difference in results could come from a difference in the schedule, the
batch order, the checkpoint-selection rule or the augmentation -- and there would
be no way to tell from the outside. Here ``model.architecture`` is the only thing
that differs between the comparison runs.

Two schedule properties worth naming:

* **Fixed gradient steps per epoch.** An epoch is ``ceil(n_train / batch)``
  steps, so the data-efficiency sweep (which varies ``n_train`` by 8x) varies the
  *label count* and, unavoidably, the number of steps. Rather than hide that, the
  sweep reports both the label count and the step count, and a fixed-step variant
  is available via ``steps_per_epoch``.
* **Checkpoint selection on validation Spearman, not validation loss.** The loss
  of an ordinal head and the loss of a regression head are not comparable
  quantities, so selecting on loss would apply a different rule to each arm.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..data.dataset import SequenceDataset
from ..metrics.regression import spearman


@dataclass
class TrainState:
    """What a completed training run leaves behind.

    Attributes:
        history: One dict per epoch, written to ``history.jsonl``.
        best_epoch: Epoch of the selected checkpoint.
        best_val_spearman: Its validation Spearman.
        best_state: The selected weights (kept in memory, optionally saved).
        wall_seconds: Total training wall-clock time.
    """

    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_spearman: float = float("-inf")
    best_state: dict[str, torch.Tensor] = field(default_factory=dict)
    wall_seconds: float = 0.0


def set_threads(n: int = 2) -> None:
    """Cap torch's thread pool. Called by every entry point.

    This machine runs several projects at once; a torch default of "all cores"
    makes every concurrent run slower than the sum of its parts and makes the
    latency benchmark unrepeatable.
    """
    torch.set_num_threads(max(1, int(n)))


def iterate_batches(
    coords: np.ndarray, quality: np.ndarray, batch_size: int, rng: np.random.Generator,
    shuffle: bool = True
):
    """Yield ``(x, y)`` tensor batches for one epoch."""
    n = coords.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield (
            torch.from_numpy(np.ascontiguousarray(coords[idx])),
            torch.from_numpy(np.ascontiguousarray(quality[idx])),
        )


def lr_at(epoch: int, cfg: Config) -> float:
    """Linear warm-up then cosine decay to ``lr * min_lr_factor``."""
    total, warm = cfg.optim.epochs, cfg.optim.warmup_epochs
    base, floor = cfg.optim.lr, cfg.optim.lr * cfg.optim.min_lr_factor
    if warm > 0 and epoch < warm:
        return base * (epoch + 1) / warm
    if total <= warm:
        return base
    t = (epoch - warm) / max(1, total - warm)
    return floor + 0.5 * (base - floor) * (1.0 + np.cos(np.pi * min(t, 1.0)))


@torch.no_grad()
def predict(
    model: nn.Module, coords: np.ndarray, batch_size: int = 64
) -> dict[str, np.ndarray]:
    """Batched inference.

    Returns:
        ``{"score", "lower", "upper"}`` arrays; the interval keys are absent when
        the model has no uncertainty head, rather than filled with the point
        estimate -- a zero-width interval would score perfect sharpness and 0%
        coverage, which is not what "no uncertainty head" means.
    """
    model.eval()
    scores, lowers, uppers = [], [], []
    for start in range(0, coords.shape[0], batch_size):
        x = torch.from_numpy(np.ascontiguousarray(coords[start : start + batch_size]))
        out = model(x)
        scores.append(out.score.cpu().numpy())
        if out.quantiles is not None or out.sigma is not None:
            lo, hi = out.interval()
            lowers.append(lo.cpu().numpy())
            uppers.append(hi.cpu().numpy())
    result = {"score": np.concatenate(scores)}
    if lowers:
        result["lower"] = np.concatenate(lowers)
        result["upper"] = np.concatenate(uppers)
    return result


def train_model(
    model: nn.Module,
    train: SequenceDataset,
    val: SequenceDataset,
    cfg: Config,
    log_path: Path | None = None,
    verbose: bool = True,
) -> TrainState:
    """Train one model and return its :class:`TrainState`.

    Args:
        model: The model.
        train: Training split.
        val: Validation split, used only for checkpoint selection.
        cfg: Resolved config.
        log_path: If given, ``history.jsonl`` is appended here as training goes,
            so a killed run still leaves a partial record.
        verbose: Print a line per epoch.

    Returns:
        The state, with ``best_state`` loaded back into ``model`` before return --
        so the caller always evaluates the selected checkpoint, never the last
        one. Forgetting this is a classic silent inflation of late-epoch
        overfitting.
    """
    set_threads(cfg.run.torch_threads)
    torch.manual_seed(cfg.run.seed)
    rng = np.random.default_rng(cfg.run.seed)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    state = TrainState()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8", newline="\n")

    t0 = time.perf_counter()
    for epoch in range(cfg.optim.epochs):
        lr = lr_at(epoch, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        losses: list[dict[str, float]] = []
        for x, y in iterate_batches(train.coords, train.quality, cfg.data.batch_size, rng):
            opt.zero_grad(set_to_none=True)
            loss, parts = model.loss(x, y)
            loss.backward()
            if cfg.optim.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()
            losses.append(parts)

        val_pred = predict(model, val.coords, cfg.data.batch_size)
        rho = spearman(val.quality, val_pred["score"])
        record = {
            "epoch": epoch,
            "lr": lr,
            "val_spearman": rho,
            "val_mae": float(np.mean(np.abs(val_pred["score"] - val.quality))),
            **{f"train_{k}": float(np.mean([p[k] for p in losses])) for k in losses[0]},
        }
        state.history.append(record)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(record) + "\n")
        if np.isfinite(rho) and rho > state.best_val_spearman:
            state.best_val_spearman = float(rho)
            state.best_epoch = epoch
            state.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(
                f"  epoch {epoch:3d} lr {lr:.2e} loss {record['train_total']:.4f} "
                f"val_rho {rho:+.4f}"
            )

    state.wall_seconds = time.perf_counter() - t0
    if state.best_state:
        model.load_state_dict(state.best_state)
    return state
