"""Monotonicity: does adding a degradation ever *raise* the predicted score?

The claim being tested is not that the ordinal head guarantees monotonicity in
the input -- it does not, and :mod:`saqa.models.heads` says so. The claim is that
this is a property worth *measuring*, because a quality-assessment system that
sometimes rewards a worse execution is unusable regardless of its rank
correlation.

Two quantities:

**Pairwise violation rate.** Over every ordered pair on a severity ladder, the
fraction where the prediction increased while the label decreased. This is the
directly interpretable number: "in X% of comparisons, making the movement worse
made the score go up".

**Rank inconsistency of the ordinal head.** A separate, structural quantity: the
fraction of items where the predicted survival function
``P(y>1), P(y>2), ...`` is not non-increasing. The shared-latent CORAL head has
this at exactly 0 by construction; the independent-logit variant does not, and
the difference is what "structural guarantee" means here.
"""

from __future__ import annotations

import numpy as np


def ladder_violations(
    predictions: np.ndarray, labels: np.ndarray, tolerance: float = 0.0
) -> dict[str, float]:
    """Violation statistics for one severity ladder.

    Args:
        predictions: ``(S,)`` model scores, in ladder order (ascending severity).
        labels: ``(S,)`` exact labels, non-increasing along the ladder.
        tolerance: A prediction increase below this is not counted. Zero by
            default: any increase is a violation. A tolerance is offered because
            a 1e-6 increase is not a coaching failure, but the shipped tables use
            0 so the number cannot be tuned into looking good.

    Returns:
        ``violation_rate`` over all ordered pairs, ``adjacent_violation_rate``
        over consecutive steps only, ``max_increase`` (the largest wrong-way
        movement), and the pair count.

    Raises:
        ValueError: if the labels are not non-increasing, which would mean the
            ladder was built wrong and the test would be meaningless.
    """
    p = np.asarray(predictions, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError(f"predictions and labels must match: {p.shape} vs {y.shape}")
    if p.size < 2:
        raise ValueError("a ladder needs at least two rungs")
    if np.any(np.diff(y) > 1e-9):
        raise ValueError("ladder labels must be non-increasing")

    total = bad = 0
    worst = 0.0
    for i in range(p.size):
        for j in range(i + 1, p.size):
            if y[j] >= y[i] - 1e-12:  # no actual worsening between these rungs
                continue
            total += 1
            delta = p[j] - p[i]
            if delta > tolerance:
                bad += 1
                worst = max(worst, delta)
    adj_total = adj_bad = 0
    for i in range(p.size - 1):
        if y[i + 1] < y[i] - 1e-12:
            adj_total += 1
            if p[i + 1] - p[i] > tolerance:
                adj_bad += 1
    return {
        "violation_rate": bad / total if total else float("nan"),
        "adjacent_violation_rate": adj_bad / adj_total if adj_total else float("nan"),
        "max_increase": worst,
        "n_pairs": float(total),
    }


def monotonicity_report(
    ladders: list[dict], predictions: list[np.ndarray], tolerance: float = 0.0
) -> dict[str, float]:
    """Aggregate violation statistics over many ladders.

    Args:
        ladders: As produced by
            :func:`saqa.data.dataset.paired_degradation_sequences`.
        predictions: One ``(S,)`` prediction array per ladder, same order.
        tolerance: Passed through.

    Returns:
        Overall rates plus a per-defect-kind breakdown, and
        ``ladders_with_any_violation`` -- the fraction of *executions* that
        exhibit at least one wrong-way comparison, which is the number a user of
        the system would actually feel.
    """
    if len(ladders) != len(predictions):
        raise ValueError(f"{len(ladders)} ladders vs {len(predictions)} predictions")
    per_kind: dict[str, list[float]] = {}
    all_rates, any_flags, worst = [], [], 0.0
    for lad, pred in zip(ladders, predictions, strict=True):
        stats = ladder_violations(pred, np.asarray(lad["quality"]), tolerance)
        all_rates.append(stats["violation_rate"])
        any_flags.append(stats["violation_rate"] > 0)
        worst = max(worst, stats["max_increase"])
        per_kind.setdefault(str(lad["kind"]), []).append(stats["violation_rate"])

    out = {
        "violation_rate": float(np.nanmean(all_rates)) if all_rates else float("nan"),
        "ladders_with_any_violation": float(np.mean(any_flags)) if any_flags else float("nan"),
        "max_increase": worst,
        "n_ladders": float(len(ladders)),
    }
    for kind, rates in sorted(per_kind.items()):
        out[f"violation_rate__{kind}"] = float(np.nanmean(rates))
        out[f"n__{kind}"] = float(len(rates))
    return out


def rank_inconsistency(cumulative_probs: np.ndarray) -> float:
    """Fraction of items whose ordinal survival function is not non-increasing.

    Args:
        cumulative_probs: ``(N, K)`` values of ``P(y > k)``.

    Returns:
        The fraction of rows with any ``P(y>k+1) > P(y>k)``. Exactly 0 for a
        shared-latent CORAL head with ordered thresholds -- which is asserted in
        the tests, not assumed here.
    """
    p = np.asarray(cumulative_probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"expected (N, K), got {p.shape}")
    if p.shape[1] < 2:
        return 0.0
    return float(np.mean(np.any(np.diff(p, axis=1) > 1e-9, axis=1)))
