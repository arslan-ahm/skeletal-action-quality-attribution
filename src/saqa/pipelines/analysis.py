"""Attribution fidelity, monotonicity and uncertainty analyses of a trained model.

These are the three claims the project is actually about, so each is run against
a control that could refute it:

* attribution against **random** attribution and against the *same method applied
  to an untrained model* -- the parameter-randomisation sanity check of
  Adebayo et al. (2018);
* monotonicity against the *label*, on ladders where nothing but severity varies;
* intervals against their nominal coverage, with width conditioned on error.
"""

from __future__ import annotations

import numpy as np
import torch

from ..attribution.methods import (
    completeness_error,
    edge_importance_attribution,
    input_gradient,
    integrated_gradients,
    occlusion,
    to_frame_vector,
    to_joint_vector,
)
from ..config import Config
from ..data.dataset import SequenceDataset, paired_degradation_sequences
from ..engine.trainer import predict
from ..metrics.attribution import fidelity_summary, per_sequence_fidelity
from ..metrics.monotonicity import monotonicity_report, rank_inconsistency
from ..models.registry import build_model
from .core import generator_config


def compute_attribution(
    model: torch.nn.Module,
    coords: np.ndarray,
    method: str,
    cfg: Config,
    batch_size: int = 16,
) -> np.ndarray:
    """``(N, T, V)`` attribution for a whole split, in batches.

    Args:
        model: The trained model, put in eval mode here.
        coords: ``(N, C, T, V)`` inputs.
        method: ``"integrated_gradients"``, ``"occlusion"`` or ``"gradient"``.
        cfg: For ``eval.ig_steps``.
        batch_size: Sequences per batch.

    Raises:
        ValueError: on an unknown method.
    """
    model.eval()
    chunks = []
    for start in range(0, coords.shape[0], batch_size):
        x = torch.from_numpy(np.ascontiguousarray(coords[start : start + batch_size]))
        if method == "integrated_gradients":
            chunks.append(integrated_gradients(model, x, steps=cfg.eval.ig_steps))
        elif method == "occlusion":
            chunks.append(occlusion(model, x))
        elif method == "gradient":
            chunks.append(input_gradient(model, x))
        else:
            raise ValueError(f"Unknown attribution method {method!r}")
    return np.concatenate(chunks, axis=0)


def attribution_fidelity(
    model: torch.nn.Module,
    data: SequenceDataset,
    cfg: Config,
    include_controls: bool = True,
    architecture: str = "saqa_stgcn",
) -> dict[str, dict[str, float]]:
    """Fidelity of every attribution method, plus the two controls.

    Returns:
        ``{method_name: {metric: value}}``. Method names carry a ``joint_`` or
        ``frame_`` prefix inside the metric keys. The controls are
        ``random`` and ``<method>_untrained``.
    """
    out: dict[str, dict[str, float]] = {}
    joint_truth = np.asarray(data.joint_truth, dtype=np.float64)
    frame_truth = np.asarray(data.frame_truth, dtype=np.float64)

    for method in cfg.eval.attribution_methods:
        attr = compute_attribution(model, data.coords, method, cfg)
        out[method] = _score_both(attr, joint_truth, frame_truth)

    if include_controls:
        rng = np.random.default_rng(cfg.run.seed + 4242)
        rand = rng.random(joint_truth.shape)
        rand_frames = rng.random(frame_truth.shape)
        out["random"] = {
            **{f"joint_{k}": v for k, v in
               fidelity_summary(rand, joint_truth, "joint").items()},
            **{f"frame_{k}": v for k, v in
               fidelity_summary(rand_frames, frame_truth, "frame").items()},
        }
        untrained = build_model(
            architecture,
            head=cfg.model.head,
            uncertainty=cfg.model.uncertainty,
            num_bins=cfg.model.num_bins,
        )
        torch.manual_seed(cfg.run.seed + 17)
        attr_u = compute_attribution(untrained, data.coords, "integrated_gradients", cfg)
        out["integrated_gradients_untrained"] = _score_both(attr_u, joint_truth, frame_truth)

    edge = edge_importance_attribution(model)
    if edge.sum() > 0:
        tiled = np.tile(edge[None, :], (joint_truth.shape[0], 1))
        out["edge_importance_global"] = {
            f"joint_{k}": v for k, v in fidelity_summary(tiled, joint_truth, "joint").items()
        }
    return out


def _score_both(attr: np.ndarray, joint_truth: np.ndarray,
                frame_truth: np.ndarray) -> dict[str, float]:
    return {
        **{f"joint_{k}": v for k, v in
           fidelity_summary(to_joint_vector(attr), joint_truth, "joint").items()},
        **{f"frame_{k}": v for k, v in
           fidelity_summary(to_frame_vector(attr), frame_truth, "frame").items()},
    }


def per_sequence_attribution_scores(
    model: torch.nn.Module, data: SequenceDataset, cfg: Config, method: str, metric: str = "iou"
) -> np.ndarray:
    """``(N,)`` per-sequence joint-attribution fidelity, for paired tests."""
    attr = compute_attribution(model, data.coords, method, cfg)
    return per_sequence_fidelity(to_joint_vector(attr), np.asarray(data.joint_truth), metric)


def monotonicity_check(
    model: torch.nn.Module, cfg: Config, num_ladders: int | None = None
) -> tuple[dict[str, float], list[dict], list[np.ndarray]]:
    """Run the severity ladders through the model and score the violations.

    Returns:
        ``(report, ladders, predictions)`` so a caller can plot the ladders.
    """
    n = num_ladders if num_ladders is not None else cfg.eval.monotonicity_ladders
    ladders = paired_degradation_sequences(n, generator_config(cfg), seed=cfg.run.seed + 31)
    preds = [predict(model, lad["coords"], cfg.data.batch_size)["score"] for lad in ladders]
    return monotonicity_report(ladders, preds), ladders, preds


def ordinal_consistency(model: torch.nn.Module, data: SequenceDataset,
                        batch_size: int = 64) -> dict[str, float]:
    """Rank-inconsistency rate of the ordinal head on real inputs.

    Zero by construction for the shared-latent head. Reported anyway, because a
    guarantee that is never checked is a comment.
    """
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "logits"):
        return {"rank_inconsistency": float("nan"), "n": 0.0}
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, data.coords.shape[0], batch_size):
            x = torch.from_numpy(np.ascontiguousarray(data.coords[start : start + batch_size]))
            feats = model.trunk(x)
            probs.append(torch.sigmoid(head.logits(feats)).cpu().numpy())
    p = np.concatenate(probs, axis=0)
    return {"rank_inconsistency": rank_inconsistency(p), "n": float(p.shape[0])}


def attribution_completeness(model: torch.nn.Module, data: SequenceDataset,
                             cfg: Config, num: int = 32) -> dict[str, float]:
    """Integrated-gradients completeness residual on a sample of sequences."""
    err = completeness_error(model, data.coords[:num], steps=cfg.eval.ig_steps)
    return {
        "completeness_mean_abs": float(np.mean(err)),
        "completeness_max_abs": float(np.max(err)),
        "ig_steps": float(cfg.eval.ig_steps),
        "n": float(len(err)),
    }
