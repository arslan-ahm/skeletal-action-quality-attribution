"""Statistical machinery for comparing runs.

Reporting that one method reached Spearman 0.84 and another 0.81 is not a
result. Two separate questions have to be kept apart, and this module exists
because conflating them is the most common failure in small-scale ML reporting:

**Question 1 -- is the difference consistent across test sequences?** Answered by
a paired test on per-sequence values (:func:`compare`). It conditions on *one
trained model per method* and is a statement about two sets of weights.

**Question 2 -- is the difference bigger than the run-to-run noise?** Answered by
training the same configuration under several seeds and comparing the difference
against ``sqrt(2) * sd`` (:func:`noise_scale`, :func:`verdict`). The sampling
unit here is the *training run*. No number of test sequences substitutes for it.

Rank metrics need one extra piece of care: Spearman correlation is a statistic of
the whole set, not a per-item value, so it cannot be tested with a paired
per-item test at all. :func:`bootstrap_metric_difference` resamples *sequences*
and recomputes the metric on each resample, which is the correct paired
bootstrap for a set-level statistic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    lower: float
    upper: float
    level: float = 0.95
    n: int = 0

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class Comparison:
    """The outcome of comparing two methods on the same sequences."""

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    difference: Interval
    p_value: float
    effect_size: float
    n: int
    p_adjusted: float | None = None

    @property
    def significant(self) -> bool:
        p = self.p_value if self.p_adjusted is None else self.p_adjusted
        return bool(np.isfinite(p) and p < 0.05)

    def to_dict(self) -> dict[str, object]:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "difference": self.difference.estimate,
            "ci_lower": self.difference.lower,
            "ci_upper": self.difference.upper,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "effect_size": self.effect_size,
            "n": self.n,
            "significant": self.significant,
        }


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
) -> Interval:
    """Percentile bootstrap interval for a summary of ``values``.

    ``NaN`` entries are dropped and the surviving count is recorded, because an
    undefined per-item value must not be silently replaced by a convenient one.
    """
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    reduce = np.mean if statistic == "mean" else np.median
    if data.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), level, 0)
    point = float(reduce(data))
    if data.size < 2:
        return Interval(point, point, point, level, int(data.size))

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, data.size, size=(n_resamples, data.size))
    reps = reduce(data[picks], axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha])
    return Interval(point, float(lo), float(hi), level, int(data.size))


def paired_bootstrap_difference(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 2000, level: float = 0.95, seed: int = 0
) -> Interval:
    """Bootstrap interval for the mean paired difference ``a - b``.

    Resamples *sequence indices*, preserving the pairing.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Paired arrays must match: {x.shape} vs {y.shape}")
    valid = np.isfinite(x) & np.isfinite(y)
    return bootstrap_ci((x - y)[valid], n_resamples, level, seed)


def compare(
    a: np.ndarray,
    b: np.ndarray,
    name_a: str = "a",
    name_b: str = "b",
    n_resamples: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Paired comparison of two methods on the same sequences.

    Returns a :class:`Comparison` whose ``p_value`` is ``NaN`` when every paired
    difference is exactly zero (the signed-rank test is undefined there).
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    diff = x - y
    if diff.size == 0 or np.allclose(diff, 0.0):
        p = float("nan")
    else:
        p = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return Comparison(
        name_a=name_a,
        name_b=name_b,
        mean_a=float(x.mean()) if x.size else float("nan"),
        mean_b=float(y.mean()) if y.size else float("nan"),
        difference=paired_bootstrap_difference(x, y, n_resamples, seed=seed),
        p_value=p,
        effect_size=float(diff.mean() / sd) if sd > 0 else 0.0,
        n=int(diff.size),
    )


def holm_bonferroni(comparisons: list[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """Holm-Bonferroni step-down correction, applied in place.

    ``NaN`` p-values are excluded from the family size, since an undefined test
    carries no evidence in either direction.
    """
    testable = [c for c in comparisons if np.isfinite(c.p_value)]
    m = len(testable)
    if m == 0:
        return comparisons
    order = sorted(range(m), key=lambda i: testable[i].p_value)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * testable[idx].p_value)
        running = max(running, adj)
        testable[idx].p_adjusted = running
    del alpha
    return comparisons


def bootstrap_metric_difference(
    truth: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired bootstrap interval for the difference in a *set-level* metric.

    Spearman correlation, Kendall tau and AUROC are functions of a whole set;
    there is no per-item value to feed a Wilcoxon test. The correct paired test
    resamples sequence indices and recomputes the metric on each resample for
    both methods, which is what this does.

    Args:
        truth: ``(N,)`` labels.
        pred_a: ``(N,)`` predictions of method A.
        pred_b: ``(N,)`` predictions of method B, same order.
        metric: ``(truth, pred) -> float``.
        n_resamples: Replicates.
        level: Coverage.
        seed: RNG seed.

    Returns:
        Interval on ``metric(A) - metric(B)``. A resample on which the metric is
        undefined for either method (e.g. a constant resample for a rank
        correlation) is dropped and the surviving count is reported.
    """
    t = np.asarray(truth, dtype=np.float64)
    a = np.asarray(pred_a, dtype=np.float64)
    b = np.asarray(pred_b, dtype=np.float64)
    if not (t.shape == a.shape == b.shape):
        raise ValueError(f"Shapes must match: {t.shape}, {a.shape}, {b.shape}")

    point = metric(t, a) - metric(t, b)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_resamples):
        idx = rng.integers(0, t.size, size=t.size)
        d = metric(t[idx], a[idx]) - metric(t[idx], b[idx])
        if np.isfinite(d):
            reps.append(d)
    if not reps:
        return Interval(point, float("nan"), float("nan"), level, 0)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha])
    return Interval(float(point), float(lo), float(hi), level, len(reps))


def noise_scale(values: np.ndarray) -> float:
    """Run-to-run scale of a *difference*, from repeated-seed values.

    Two independent runs differ with standard deviation ``sqrt(2) * sd``, so a
    gap smaller than that is not distinguishable from having reseeded.

    Args:
        values: The metric from >= 2 runs of the *same* configuration.

    Returns:
        ``sqrt(2) * sd``, or ``NaN`` with fewer than two finite values.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(np.sqrt(2.0) * v.std(ddof=1))


def verdict(difference: float, scale: float) -> str:
    """Classify a claimed improvement against the run-to-run noise scale.

    Returns one of ``"robust"`` (>= 3x), ``"survives"`` (>= 2x),
    ``"suggestive"`` (>= 1x), ``"inside noise"`` (< 1x) or ``"unknown"`` when
    the scale could not be estimated. The thresholds are a reporting
    convention, stated so that a reader can apply their own.
    """
    if not np.isfinite(scale) or scale <= 0:
        return "unknown"
    ratio = abs(difference) / scale
    if ratio >= 3.0:
        return "robust"
    if ratio >= 2.0:
        return "survives"
    if ratio >= 1.0:
        return "suggestive"
    return "inside noise"
