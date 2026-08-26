"""The reference approach: DTW alignment to a reference performance.

This is what the repository being upgraded does -- extract 3D pose from two
videos, align them with dynamic time warping (Sakoe & Chiba, 1978), and report a
similarity score. It is implemented here properly and given every advantage,
because a baseline that is described but not run, or run badly, proves nothing.

What "properly" means concretely:

* **Three distance functions**, not one: raw position L2, *per-joint normalised*
  L2 (each joint's distance divided by its own spread in the reference, so a
  wrist does not dominate a pelvis), and velocity cosine distance (which is
  invariant to posture offsets and sensitive to timing and smoothness).
* **A Sakoe-Chiba band** option, since an unbounded warp can absorb the very
  tempo defect the score should be penalising -- an unbanded DTW is not a
  stronger baseline, it is a blinder one, and both are reported.
* **A calibration stage.** A DTW distance is not a quality score. Comparing a raw
  distance against a calibrated model on a score metric would be a rigged fight,
  so the distance is mapped to the quality scale by isotonic regression fitted on
  the *training* split -- the best monotone map available, which is exactly what
  a rank metric rewards. Spearman is invariant to it; relative L2 is not, and
  without it the baseline's L2 would be meaningless.

The structural limitation is not fixable by any of that, and it is the point of
the project: this baseline **requires a reference performance**. The reference is
supplied here from the generator. In free-form assessment it does not exist.
"""

from __future__ import annotations

import numpy as np

from ..data.skeleton import NUM_JOINTS

DISTANCES: tuple[str, ...] = ("l2", "per_joint_norm", "velocity_cosine")


def _flatten(seq: np.ndarray) -> np.ndarray:
    """``(T, V, 3)`` or ``(3, T, V)`` -> ``(T, V, 3)``."""
    a = np.asarray(seq, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"expected a 3-D sequence, got {a.shape}")
    if a.shape[0] == 3 and a.shape[2] == NUM_JOINTS:
        return a.transpose(1, 2, 0)
    return a


def frame_cost_matrix(
    query: np.ndarray, reference: np.ndarray, distance: str = "per_joint_norm"
) -> np.ndarray:
    """``(Tq, Tr)`` frame-to-frame cost matrix.

    Args:
        query: The learner's sequence.
        reference: The reference execution.
        distance: One of :data:`DISTANCES`.

    Returns:
        The cost matrix.

    Raises:
        ValueError: on an unknown distance.
    """
    q = _flatten(query)
    r = _flatten(reference)
    if distance == "l2":
        diff = q[:, None, :, :] - r[None, :, :, :]
        return np.linalg.norm(diff, axis=-1).mean(axis=-1)
    if distance == "per_joint_norm":
        # Divide each joint's contribution by that joint's own spread in the
        # reference. Without this the distance is dominated by the extremities,
        # which move most in every action -- so a pelvis-level stability defect
        # would be invisible to the baseline for reasons of scale rather than of
        # information.
        spread = r.std(axis=0).mean(axis=-1)  # (V,)
        spread = np.maximum(spread, 1e-3)
        diff = q[:, None, :, :] - r[None, :, :, :]
        per_joint = np.linalg.norm(diff, axis=-1) / spread[None, None, :]
        return per_joint.mean(axis=-1)
    if distance == "velocity_cosine":
        qv = np.diff(q, axis=0, prepend=q[:1])
        rv = np.diff(r, axis=0, prepend=r[:1])
        qn = qv / np.maximum(np.linalg.norm(qv, axis=-1, keepdims=True), 1e-8)
        rn = rv / np.maximum(np.linalg.norm(rv, axis=-1, keepdims=True), 1e-8)
        cos = np.einsum("tvc,svc->tsv", qn, rn)
        return (1.0 - cos).mean(axis=-1)
    raise ValueError(f"Unknown distance {distance!r}; known: {DISTANCES}")


def dtw_distance(
    cost: np.ndarray, band: float | None = 0.15
) -> tuple[float, np.ndarray]:
    """DTW distance and the accumulated cost matrix.

    Standard dynamic program with the three-step transition set
    ``{(-1,-1), (-1,0), (0,-1)}`` and length normalisation by the path length,
    which makes distances comparable across sequences of different length.

    Args:
        cost: ``(Tq, Tr)`` frame cost matrix.
        band: Sakoe-Chiba band width as a fraction of the longer sequence, or
            ``None`` for an unconstrained warp.

    Returns:
        ``(normalised_distance, accumulated)``. Cells outside the band are
        ``inf`` in ``accumulated``.
    """
    c = np.asarray(cost, dtype=np.float64)
    n, m = c.shape
    inf = np.inf
    acc = np.full((n + 1, m + 1), inf)
    acc[0, 0] = 0.0
    steps = np.zeros((n + 1, m + 1), dtype=np.int64)
    width = int(np.ceil(band * max(n, m))) if band is not None else max(n, m)

    for i in range(1, n + 1):
        centre = (i - 1) * m / n
        lo = max(1, int(np.floor(centre - width)) + 1)
        hi = min(m, int(np.ceil(centre + width)) + 1)
        for j in range(lo, hi + 1):
            best, best_steps = inf, 0
            for di, dj in ((1, 1), (1, 0), (0, 1)):
                prev = acc[i - di, j - dj]
                if prev < best:
                    best, best_steps = prev, steps[i - di, j - dj]
            if best == inf:
                continue
            acc[i, j] = best + c[i - 1, j - 1]
            steps[i, j] = best_steps + 1
    if not np.isfinite(acc[n, m]):
        # A band too narrow to admit any path: report NaN rather than a number
        # that would be silently compared against the other configurations.
        return float("nan"), acc
    return float(acc[n, m] / max(steps[n, m], 1)), acc


def framewise_distance(query: np.ndarray, reference: np.ndarray,
                       distance: str = "per_joint_norm") -> float:
    """No-alignment control: the mean diagonal cost, resampling to equal length.

    The frame-wise variant of the reference approach, included because it is what
    a lot of implementations actually do and it isolates how much the DTW
    alignment itself contributes.
    """
    q, r = _flatten(query), _flatten(reference)
    if q.shape[0] != r.shape[0]:
        idx = np.linspace(0, r.shape[0] - 1, q.shape[0])
        r = np.stack([
            np.stack([np.interp(idx, np.arange(r.shape[0]), r[:, v, c])
                      for c in range(3)], axis=-1)
            for v in range(r.shape[1])
        ], axis=1)
    return float(np.mean(np.diagonal(frame_cost_matrix(q, r, distance))))


def dtw_similarity(
    query: np.ndarray,
    reference: np.ndarray,
    distance: str = "per_joint_norm",
    band: float | None = 0.15,
) -> float:
    """The reference repo's output: one DTW distance, lower is more similar."""
    d, _ = dtw_distance(frame_cost_matrix(query, reference, distance), band)
    return d


def per_joint_dtw_deviation(
    query: np.ndarray, reference: np.ndarray, band: float | None = 0.15
) -> np.ndarray:
    """``(V,)`` per-joint deviation along the DTW-optimal path.

    The most any reference-based method can offer as an explanation: align once
    on the whole body, then report which joints deviate most along that path. It
    is included so the attribution comparison is against the reference approach's
    *best* explanation rather than against nothing -- and its weakness is
    structural: the alignment is chosen to minimise whole-body cost, so a defect
    that DTW absorbed into the warp leaves no per-joint trace.
    """
    q, r = _flatten(query), _flatten(reference)
    cost = frame_cost_matrix(q, r, "per_joint_norm")
    _, acc = dtw_distance(cost, band)
    n, m = cost.shape
    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (acc[i - 1, j - 1], i - 1, j - 1),
            (acc[i - 1, j], i - 1, j),
            (acc[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda t: t[0])
    spread = np.maximum(r.std(axis=0).mean(axis=-1), 1e-3)
    dev = np.zeros(q.shape[1], dtype=np.float64)
    for qi, ri in path:
        dev += np.linalg.norm(q[qi] - r[ri], axis=-1) / spread
    return dev / max(len(path), 1)
