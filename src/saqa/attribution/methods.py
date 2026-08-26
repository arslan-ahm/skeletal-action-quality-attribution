"""Attribution methods: integrated gradients, occlusion, plain gradient.

All three produce a ``(T, V)`` importance map per sequence, which is then reduced
to per-joint and per-phase vectors. The sign convention throughout is **"how much
did this joint at this time cost quality"**: a positive attribution means the
model's score would have been *higher* without whatever is happening there. That
is the direction a coach needs, and getting it backwards produces a plausible-
looking map that names the joints doing the work correctly rather than the joints
doing it wrong.

Three methods rather than one, because they fail differently:

* **Integrated gradients** (Sundararajan et al., 2017) satisfies completeness --
  the attributions sum to ``f(x) - f(baseline)`` -- which is checkable, and it is
  checked in the tests. It needs a baseline, and the choice of baseline is the
  method's weak point (see :func:`integrated_gradients`).
* **Occlusion of joint groups** needs no gradient and no baseline choice beyond
  "what does removal mean", so it is the honest cross-check. It is also the
  expensive one: one forward pass per group.
* **Plain input-gradient** is the cheap control. If it matches integrated
  gradients, the extra 24 forward-backward passes bought nothing, and that is
  worth knowing.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..data.skeleton import JOINT_GROUPS, NUM_JOINTS


def _as_batch(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if t.ndim == 3:
        t = t[None]
    if t.ndim != 4:
        raise ValueError(f"expected (N, C, T, V) or (C, T, V), got {tuple(t.shape)}")
    return t


def _score_fn(model: nn.Module):
    fn = getattr(model, "score_only", None)
    return fn if callable(fn) else model


def temporal_baseline(x: torch.Tensor) -> torch.Tensor:
    """Baseline that replaces each joint's trajectory by its temporal mean.

    The obvious baseline -- all zeros -- is a body collapsed to the origin, which
    is not a *movement* at all; integrated gradients then explains "why is this a
    skeleton" rather than "why is this a poor execution", and the resulting map is
    dominated by whichever joints are furthest from the pelvis. Freezing each
    joint at its own mean position keeps the pose and removes the *motion*, which
    is the quantity quality is a function of. This choice materially changes the
    fidelity numbers and is one of the things the results section reports.
    """
    return x.mean(dim=2, keepdim=True).expand_as(x).clone()


def integrated_gradients(
    model: nn.Module,
    x: np.ndarray | torch.Tensor,
    steps: int = 24,
    baseline: str = "temporal_mean",
) -> np.ndarray:
    """Integrated gradients of the *negated* score with respect to the input.

    The score is negated so that a positive attribution means "cost quality",
    matching the convention above.

    Args:
        model: A :class:`~saqa.models.quality.QualityModel` (or anything with
            ``score_only``).
        x: ``(C, T, V)`` or ``(N, C, T, V)`` input.
        steps: Riemann steps along the straight path. The midpoint rule is used
            (``(i + 0.5) / steps``) rather than the left endpoint, because the
            left rule systematically under-integrates a convex path and breaks
            the completeness check by several percent at small ``steps``.
        baseline: ``"temporal_mean"`` or ``"zeros"``.

    Returns:
        ``(N, T, V)`` attribution, summed over coordinate channels.

    Raises:
        ValueError: on an unknown baseline.
    """
    batch = _as_batch(x)
    if baseline == "temporal_mean":
        base = temporal_baseline(batch)
    elif baseline == "zeros":
        base = torch.zeros_like(batch)
    else:
        raise ValueError(f"Unknown baseline {baseline!r}")

    score = _score_fn(model)
    total = torch.zeros_like(batch)
    diff = batch - base
    for i in range(steps):
        alpha = (i + 0.5) / steps
        point = (base + alpha * diff).detach().requires_grad_(True)
        out = -score(point).sum()
        (grad,) = torch.autograd.grad(out, point)
        total = total + grad
    attribution = (diff * total / steps).sum(dim=1)
    return attribution.detach().cpu().numpy()


def completeness_error(
    model: nn.Module, x: np.ndarray | torch.Tensor, steps: int = 24,
    baseline: str = "temporal_mean"
) -> np.ndarray:
    """``|sum(IG) - (f(baseline) - f(x))|`` per sequence.

    Integrated gradients of ``-f`` sums to ``f(baseline) - f(x)``. The residual
    is pure numerical integration error, so it is a direct check that ``steps``
    is large enough -- and it is reported rather than assumed.
    """
    batch = _as_batch(x)
    base = temporal_baseline(batch) if baseline == "temporal_mean" else torch.zeros_like(batch)
    score = _score_fn(model)
    with torch.no_grad():
        gap = (score(base) - score(batch)).cpu().numpy()
    attr = integrated_gradients(model, batch, steps, baseline)
    return np.abs(attr.reshape(attr.shape[0], -1).sum(axis=1) - gap)


def input_gradient(model: nn.Module, x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Gradient x input, the single-pass control. ``(N, T, V)``."""
    batch = _as_batch(x).detach().requires_grad_(True)
    out = -_score_fn(model)(batch).sum()
    (grad,) = torch.autograd.grad(out, batch)
    return (grad * batch).sum(dim=1).detach().cpu().numpy()


def occlusion(
    model: nn.Module,
    x: np.ndarray | torch.Tensor,
    groups: dict[str, tuple[int, ...]] | None = None,
    window: int = 16,
    stride: int = 8,
) -> np.ndarray:
    """Occlusion attribution over (joint group x time window) tiles.

    A tile is occluded by replacing those joints' coordinates *within that window*
    by their temporal mean -- the same "remove the motion, keep the pose" edit as
    the integrated-gradients baseline, so the two methods are answering the same
    counterfactual question and a disagreement between them is about the method,
    not about the intervention.

    The score *increase* under occlusion is the tile's attribution: if hiding a
    joint's motion makes the execution look better, that motion was costing
    quality.

    Args:
        model: The model.
        x: ``(C, T, V)`` or ``(N, C, T, V)``.
        groups: Joint groups; defaults to
            :data:`saqa.data.skeleton.JOINT_GROUPS`. Groups rather than single
            joints because the skeleton is a chain -- occluding one joint while
            its neighbours stay put creates an anatomically impossible pose far
            outside the training distribution, and the model's response to that
            is not attribution, it is extrapolation.
        window: Temporal window length in frames.
        stride: Window stride. Overlapping windows are averaged.

    Returns:
        ``(N, T, V)`` attribution, constant within each tile.
    """
    batch = _as_batch(x)
    n, c, t, v = batch.shape
    groups = groups or JOINT_GROUPS
    score = _score_fn(model)
    with torch.no_grad():
        base_score = score(batch)
    means = batch.mean(dim=2, keepdim=True)

    attr = np.zeros((n, t, v), dtype=np.float64)
    count = np.zeros((n, t, v), dtype=np.float64)
    starts = list(range(0, max(1, t - window + 1), stride)) or [0]
    if starts[-1] + window < t:
        starts.append(t - window)
    for joints in groups.values():
        j_idx = list(joints)
        for s in starts:
            e = min(t, s + window)
            probe = batch.clone()
            probe[:, :, s:e, j_idx] = means[:, :, :, j_idx].expand(n, c, e - s, len(j_idx))
            with torch.no_grad():
                delta = (score(probe) - base_score).cpu().numpy()
            attr[:, s:e, :][:, :, j_idx] += delta[:, None, None]
            count[:, s:e, :][:, :, j_idx] += 1.0
    return attr / np.maximum(count, 1.0)


def to_joint_vector(attribution: np.ndarray) -> np.ndarray:
    """Reduce a ``(N, T, V)`` map to ``(N, V)`` per-joint importance.

    Positive parts only. A joint with large positive attribution in one phase and
    large negative in another has a real, localised cost, and summing signed
    values would cancel it to zero -- which is how a genuinely informative
    explanation gets thrown away by an averaging step.
    """
    a = np.asarray(attribution, dtype=np.float64)
    if a.ndim == 2:
        a = a[None]
    return np.clip(a, 0.0, None).sum(axis=1)


def to_frame_vector(attribution: np.ndarray) -> np.ndarray:
    """Reduce a ``(N, T, V)`` map to ``(N, T)`` per-frame importance."""
    a = np.asarray(attribution, dtype=np.float64)
    if a.ndim == 2:
        a = a[None]
    return np.clip(a, 0.0, None).sum(axis=2)


def edge_importance_attribution(model: nn.Module) -> np.ndarray:
    """The model's *own* view of joint importance, from the learned edge masks.

    Averages ``|A * edge_weight|`` over partitions and blocks and sums the
    incoming mass per joint. This is a *global*, input-independent attribution --
    it cannot explain one sequence -- so it is reported separately and never
    scored against per-sequence ground truth, which would be a category error.

    Returns:
        ``(V,)`` importance, or an all-zero vector if no block has edge weights.
    """
    total = np.zeros(NUM_JOINTS, dtype=np.float64)
    found = False
    for module in model.modules():
        weight = getattr(module, "edge_weight", None)
        adj = getattr(module, "A", None)
        if weight is None or adj is None:
            continue
        found = True
        eff = (adj * weight).abs().detach().cpu().numpy()
        total += eff.sum(axis=(0, 1))
    return total if found else np.zeros(NUM_JOINTS, dtype=np.float64)
