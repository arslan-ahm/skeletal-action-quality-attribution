"""Interval calibration, sharpness, and abstention.

Three questions, three answers, and they are not interchangeable:

**Is the interval honest?** Empirical coverage against the nominal level. An
interval that covers 99% of a set at a nominal 90% is not "safe", it is
uninformative -- so coverage is always read next to width.

**Is it sharp?** Mean interval width, plus the width conditional on being
correct vs incorrect. A useful interval is *narrow when the model is right and
wide when it is wrong*; a constant-width interval can hit nominal coverage
exactly while carrying no information at all, and the coverage number alone
cannot tell the two apart.

**When should the system decline to grade?** The risk-coverage curve
(Geifman & El-Yaniv, 2017 formulation): sort by confidence, and plot the error
of the retained fraction. AURC summarises it; the *oracle* AURC (sorting by true
error) is the achievable floor, and the excess over the oracle -- E-AURC -- is
the part that is about the confidence signal rather than about the model's
accuracy.
"""

from __future__ import annotations

import numpy as np


def coverage(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of labels inside ``[lower, upper]``, inclusive."""
    t = np.asarray(truth, dtype=np.float64).ravel()
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    if not (t.shape == lo.shape == hi.shape):
        raise ValueError("truth, lower and upper must have the same shape")
    m = np.isfinite(t) & np.isfinite(lo) & np.isfinite(hi)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((t[m] >= lo[m]) & (t[m] <= hi[m])))


def mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width. Negative widths (crossed quantiles) are kept, not
    clipped -- silently clipping them would hide the very failure that the
    non-crossing parameterisation exists to prevent."""
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    w = hi - lo
    w = w[np.isfinite(w)]
    return float(w.mean()) if w.size else float("nan")


def crossing_rate(lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of intervals with ``upper < lower``. Should be exactly 0."""
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    m = np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean(hi[m] < lo[m])) if m.sum() else float("nan")


def interval_summary(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    prediction: np.ndarray,
    nominal: float = 0.90,
    error_threshold: float = 0.10,
) -> dict[str, float]:
    """Coverage, width, and the width split by whether the point estimate was
    accurate.

    Args:
        truth: ``(N,)`` labels.
        lower: ``(N,)`` interval lower bounds.
        upper: ``(N,)`` interval upper bounds.
        prediction: ``(N,)`` point estimates.
        nominal: Target coverage.
        error_threshold: Absolute error above which an item counts as "wrong",
            for the conditional-width split. 0.10 is roughly one ordinal band of
            this project's score scale.

    Returns:
        A dict of scalars. ``width_ratio_wrong_right`` above 1 means the interval
        widens on the items the model gets wrong, which is the property that
        makes it usable for abstention.
    """
    t = np.asarray(truth, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    err = np.abs(p - t)
    wrong = err > error_threshold
    width = hi - lo
    w_wrong = float(width[wrong].mean()) if wrong.any() else float("nan")
    w_right = float(width[~wrong].mean()) if (~wrong).any() else float("nan")
    return {
        "coverage": coverage(t, lo, hi),
        "nominal": float(nominal),
        "coverage_gap": coverage(t, lo, hi) - float(nominal),
        "mean_width": mean_width(lo, hi),
        "median_width": float(np.median(width[np.isfinite(width)]))
        if np.isfinite(width).any()
        else float("nan"),
        "crossing_rate": crossing_rate(lo, hi),
        "width_wrong": w_wrong,
        "width_right": w_right,
        "width_ratio_wrong_right": w_wrong / w_right
        if np.isfinite(w_wrong) and np.isfinite(w_right) and w_right > 0
        else float("nan"),
        "n_wrong": float(wrong.sum()),
        "n": float(t.size),
    }


def risk_coverage_curve(
    error: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Risk (mean error of the retained set) against coverage.

    Args:
        error: ``(N,)`` per-item error, higher is worse.
        confidence: ``(N,)`` per-item confidence, higher means "keep".

    Returns:
        ``(coverages, risks)``, both ``(N,)``, coverage ascending from ``1/N``
        to 1. Ties in confidence are broken by the input order, which is
        deterministic but arbitrary; with many ties the curve should be read as
        one of several equally valid orderings.
    """
    e = np.asarray(error, dtype=np.float64).ravel()
    c = np.asarray(confidence, dtype=np.float64).ravel()
    if e.shape != c.shape:
        raise ValueError(f"error and confidence must match: {e.shape} vs {c.shape}")
    m = np.isfinite(e) & np.isfinite(c)
    e, c = e[m], c[m]
    if e.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(-c, kind="stable")
    e_sorted = e[order]
    risks = np.cumsum(e_sorted) / np.arange(1, e.size + 1)
    covs = np.arange(1, e.size + 1) / e.size
    return covs, risks


def aurc(error: np.ndarray, confidence: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    covs, risks = risk_coverage_curve(error, confidence)
    if covs.size == 0:
        return float("nan")
    return float(np.trapezoid(risks, covs) / (covs[-1] - covs[0]) if covs.size > 1
                 else risks[0])


def excess_aurc(error: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    """AURC, the oracle AURC, and their difference.

    The oracle sorts by the true error, so its AURC is the floor achievable by
    *any* confidence signal on this model's errors. Reporting AURC alone
    confounds "the confidence signal is good" with "the model is accurate";
    E-AURC isolates the former.
    """
    a = aurc(error, confidence)
    e = np.asarray(error, dtype=np.float64).ravel()
    oracle = aurc(e, -e)
    return {"aurc": a, "aurc_oracle": oracle, "e_aurc": a - oracle}


def error_detection_auroc(error: np.ndarray, confidence: np.ndarray,
                          threshold: float = 0.10) -> float:
    """AUROC of ``-confidence`` as a detector of "error above threshold".

    Computed by the rank-sum identity rather than by sweeping thresholds, so it
    is exact and handles ties by mid-ranks. Returns ``NaN`` when one class is
    empty, since AUROC is undefined there -- reporting 0.5 would claim a
    measurement that was not made.
    """
    e = np.asarray(error, dtype=np.float64).ravel()
    c = np.asarray(confidence, dtype=np.float64).ravel()
    m = np.isfinite(e) & np.isfinite(c)
    e, c = e[m], c[m]
    pos = e > threshold
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(-c)  # low confidence -> high score for "is an error"
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
