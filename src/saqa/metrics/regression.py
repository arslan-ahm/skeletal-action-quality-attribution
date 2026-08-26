"""Action-quality regression metrics.

Spearman rank correlation is the standard metric in the AQA literature (Parmar &
Morris, 2019; Xu et al., 2022) and it is reported first, but on its own it is
misleading in two ways this module guards against:

* It is **scale- and offset-blind**. A model that predicts ``0.5 * q + 0.2``
  scores a perfect 1.0. So relative L2 error is reported alongside it.
* It is **tie-sensitive**. The quality label here has an exact ceiling at 1.0 and
  a floor at 0.05, and ties at the ceiling inflate or deflate rho depending on
  how the implementation handles them. Kendall tau-b, which corrects for ties
  explicitly, is reported as the cross-check, and the tie fraction is reported so
  a reader can see how much correction is in play.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _clean(truth: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(truth, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    if t.shape != p.shape:
        raise ValueError(f"truth and pred must match: {t.shape} vs {p.shape}")
    m = np.isfinite(t) & np.isfinite(p)
    return t[m], p[m]


def spearman(truth: np.ndarray, pred: np.ndarray) -> float:
    """Spearman rank correlation.

    Returns ``NaN`` -- not 0.0 -- when either input is constant, because the
    correlation is genuinely undefined there and 0.0 would be read as "no
    relationship measured", which is a different claim.
    """
    t, p = _clean(truth, pred)
    if t.size < 3 or np.ptp(t) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(stats.spearmanr(t, p).statistic)


def kendall_tau(truth: np.ndarray, pred: np.ndarray) -> float:
    """Kendall tau-b (tie-corrected). ``NaN`` when undefined."""
    t, p = _clean(truth, pred)
    if t.size < 3 or np.ptp(t) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(stats.kendalltau(t, p, variant="b").statistic)


def pearson(truth: np.ndarray, pred: np.ndarray) -> float:
    """Pearson correlation. ``NaN`` when undefined."""
    t, p = _clean(truth, pred)
    if t.size < 3 or np.ptp(t) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(stats.pearsonr(t, p).statistic)


def relative_l2(truth: np.ndarray, pred: np.ndarray) -> float:
    """Relative L2 error ``||pred - truth|| / ||truth||``.

    The AQA convention (Parmar & Morris, 2019) normalises by the *range* of the
    score; here the label range is a known constant, so the norm of the truth
    vector is used, which is the stricter version and does not flatter a model
    that predicts the mean.
    """
    t, p = _clean(truth, pred)
    denom = float(np.linalg.norm(t))
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm(p - t) / denom)


def per_item_absolute_error(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """``(N,)`` absolute errors -- the per-sequence unit for paired tests."""
    t = np.asarray(truth, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    return np.abs(p - t)


def tie_fraction(values: np.ndarray) -> float:
    """Fraction of items sharing a value with at least one other item."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    _, counts = np.unique(v, return_counts=True)
    return float((counts[counts > 1]).sum() / v.size)


def regression_summary(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """All point metrics in one dict, plus the counts that produced them."""
    t, p = _clean(truth, pred)
    return {
        "spearman": spearman(t, p),
        "kendall_tau": kendall_tau(t, p),
        "pearson": pearson(t, p),
        "relative_l2": relative_l2(t, p),
        "mae": float(np.mean(np.abs(p - t))) if t.size else float("nan"),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))) if t.size else float("nan"),
        "tie_fraction_truth": tie_fraction(t),
        "n": float(t.size),
    }


def per_group_summary(
    truth: np.ndarray, pred: np.ndarray, groups: np.ndarray, metric: str = "spearman"
) -> dict[str, float]:
    """One metric per group (action class, defect combination, subject).

    Groups with fewer than three items give ``NaN`` rather than a rank
    correlation computed from two points, and the contributing count is returned
    alongside each value so a reader can discount a thin cell.
    """
    fn = {"spearman": spearman, "kendall_tau": kendall_tau, "relative_l2": relative_l2,
          "mae": lambda a, b: float(np.mean(np.abs(a - b)))}[metric]
    g = np.asarray(groups).ravel()
    out: dict[str, float] = {}
    for name in sorted(set(g.tolist())):
        m = g == name
        out[str(name)] = fn(np.asarray(truth)[m], np.asarray(pred)[m]) if m.sum() >= 3 else float(
            "nan"
        )
        out[f"{name}__n"] = float(m.sum())
    return out
