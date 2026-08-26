"""Attribution fidelity against exact ground truth.

Most attribution work stops at a heat map. This module does the thing that is
usually skipped: it scores the explanation against a *known* cause. Because the
generator applied the defects, "which joints and which frames were responsible
for the quality loss" is not a matter of interpretation
(:func:`saqa.data.generator.attribution_ground_truth`).

Four families, and each answers a different objection:

**Ranking (top-k precision/recall, IoU).** Does the explanation put the right
joints at the top? Reported against ``k`` chosen as the number of ground-truth
joints carrying 80% of the mass, so ``k`` adapts to how localised the true cause
is rather than being fixed at a flattering value.

**Distributional agreement (rank correlation, mass overlap).** Uses the whole
vector, not just the top. Attribution methods with the right shape but the wrong
peak get credit here and lose it above; the pair is informative.

**Random and sanity baselines.** Adebayo et al. (2018) showed that saliency maps
can look convincing while being independent of the model's parameters. So every
number is reported next to (a) a uniform-random attribution and (b) the same
attribution computed from an *untrained* model with the same architecture. An
attribution method that does not beat both is not explaining anything, and this
project reports that outcome where it occurs rather than omitting the control.

**Temporal localisation.** The phase-domain analogue: the distance between the
centre of mass of the attributed frames and the centre of mass of the true
frames, in units of the sequence length.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _normalise(x: np.ndarray) -> np.ndarray:
    """Non-negative, sum-1 normalisation. All-zero input stays all-zero."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    total = a.sum()
    return a / total if total > 0 else a


def adaptive_k(truth: np.ndarray, mass: float = 0.8, min_k: int = 1) -> int:
    """Number of top ground-truth entries needed to reach ``mass`` of the total.

    Fixing ``k`` would let the reported precision be tuned; deriving it from how
    concentrated the *truth* is makes the metric answer "did you find the cause",
    not "did you guess three joints".
    """
    t = _normalise(truth)
    if t.sum() == 0:
        return min_k
    ordered = np.sort(t)[::-1]
    cum = np.cumsum(ordered)
    k = int(np.searchsorted(cum, mass) + 1)
    return max(min_k, min(k, t.size))


def topk_scores(
    attribution: np.ndarray, truth: np.ndarray, k: int | None = None, mass: float = 0.8
) -> dict[str, float]:
    """Top-k precision, recall, F1, IoU and captured ground-truth mass.

    Args:
        attribution: ``(V,)`` or ``(T,)`` predicted importance, any sign.
        truth: Same shape, non-negative ground truth.
        k: Fixed ``k``, or ``None`` to derive it from ``truth`` via
            :func:`adaptive_k`.
        mass: Passed to :func:`adaptive_k`.

    Returns:
        A dict of scalars. Every entry is ``NaN`` when the ground truth is
        all-zero (a clean sequence has no cause, so precision is undefined rather
        than 1.0).
    """
    a = _normalise(attribution)
    t = _normalise(truth)
    if t.sum() == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                "iou": float("nan"), "mass_captured": float("nan"), "k": float("nan")}
    kk = adaptive_k(t, mass) if k is None else int(k)
    pred_set = set(np.argsort(-a)[:kk].tolist())
    true_set = set(np.argsort(-t)[:kk].tolist())
    inter = pred_set & true_set
    precision = len(inter) / kk
    recall = len(inter) / len(true_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = len(inter) / len(pred_set | true_set)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "mass_captured": float(t[list(pred_set)].sum()),
        "k": float(kk),
    }


def distribution_scores(attribution: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Whole-vector agreement: rank correlation and overlap of the mass."""
    a = _normalise(attribution)
    t = _normalise(truth)
    if t.sum() == 0 or np.ptp(t) == 0:
        return {"rank_corr": float("nan"), "overlap": float("nan"),
                "top1_hit": float("nan")}
    rank_corr = (
        float(stats.spearmanr(a, t).statistic) if np.ptp(a) > 0 else float("nan")
    )
    return {
        "rank_corr": rank_corr,
        # Histogram-intersection: sum of elementwise minima of two unit-mass
        # vectors, which is 1 for identical distributions and 0 for disjoint.
        "overlap": float(np.minimum(a, t).sum()),
        "top1_hit": float(int(np.argmax(a) == np.argmax(t))),
    }


def temporal_localisation_error(attribution: np.ndarray, truth: np.ndarray) -> float:
    """Distance between attribution and truth temporal centres of mass.

    In units of sequence length, so 0.1 means "off by a tenth of the clip".
    ``NaN`` when the ground truth is empty.
    """
    a = _normalise(attribution)
    t = _normalise(truth)
    if t.sum() == 0 or a.sum() == 0:
        return float("nan")
    idx = np.arange(t.size, dtype=np.float64) / max(t.size - 1, 1)
    return float(abs((a * idx).sum() - (t * idx).sum()))


def fidelity_summary(
    attributions: np.ndarray,
    truths: np.ndarray,
    kind: str = "joint",
    mass: float = 0.8,
) -> dict[str, float]:
    """Mean fidelity over a set of sequences, with contributing counts.

    Args:
        attributions: ``(N, D)`` predicted importance.
        truths: ``(N, D)`` ground truth.
        kind: ``"joint"`` or ``"frame"``; ``"frame"`` adds the localisation error.
        mass: Ground-truth mass fraction that sets ``k``.

    Returns:
        A dict of means, each accompanied by ``n_<metric>`` -- the number of
        sequences that contributed. Clean sequences contribute to nothing and
        are excluded rather than counted as perfect.

    Raises:
        ValueError: on a shape mismatch.
    """
    a = np.asarray(attributions, dtype=np.float64)
    t = np.asarray(truths, dtype=np.float64)
    if a.shape != t.shape:
        raise ValueError(f"attributions and truths must match: {a.shape} vs {t.shape}")

    rows: list[dict[str, float]] = []
    for i in range(a.shape[0]):
        row = {**topk_scores(a[i], t[i], mass=mass), **distribution_scores(a[i], t[i])}
        if kind == "frame":
            row["localisation_error"] = temporal_localisation_error(a[i], t[i])
        rows.append(row)

    out: dict[str, float] = {}
    keys = rows[0].keys() if rows else []
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[key] = float(finite.mean()) if finite.size else float("nan")
        out[f"n_{key}"] = float(finite.size)
    return out


def per_sequence_fidelity(
    attributions: np.ndarray, truths: np.ndarray, metric: str = "iou", mass: float = 0.8
) -> np.ndarray:
    """``(N,)`` per-sequence fidelity, for paired statistical tests.

    ``NaN`` for clean sequences, which the paired tests then drop.
    """
    a = np.asarray(attributions, dtype=np.float64)
    t = np.asarray(truths, dtype=np.float64)
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        scores = {**topk_scores(a[i], t[i], mass=mass), **distribution_scores(a[i], t[i])}
        out[i] = scores.get(metric, float("nan"))
    return out


def random_attribution(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Uniform-random attribution of the given shape -- the null control."""
    return np.random.default_rng(seed).random(shape)
