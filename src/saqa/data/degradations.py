"""Parameterised, ground-truth-labelled movement degradations.

This module is where the project's methodological claim lives. A degradation is
a **named, severity-parameterised edit to a joint-angle trajectory**, and the
quality label is an explicit function of the applied severities:

.. math::

    q = 1 - \\mathrm{clip}\\left(\\sum_d w_{k(d)}\\, s_d,\\; 0,\\; 0.95\\right)

with :math:`s_d \\in [0, 1]` the severity of degradation :math:`d` and
:math:`w_k` a per-kind cost. Three consequences, and they are the reason the
repository exists:

1. **The score is exact.** No annotator, no inter-rater noise, no ceiling.
2. **The label is monotone in every severity by construction**, so a paired
   sequence that differs only by *added* degradation has a strictly lower label.
   That is what makes the monotonicity-violation rate of a *model* measurable
   (:mod:`saqa.metrics.monotonicity`).
3. **Attribution ground truth is exact**, because "which joints did this defect
   move, and when" is answered by running forward kinematics twice -- once with
   the defect and once without -- rather than by declaring it.

The honest limitation, stated here and in the README: :math:`w_k` is a modelling
choice, not a measured human judgement. This validates the *mechanism* (can a
model recover a known score function and a known cause?), not agreement with a
real judge. A synthetic label cannot establish the latter and this project does
not claim it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actions import ACTION_SPECS, DOF

#: Quality cost per unit severity, by kind. Ordered so that a *compensatory*
#: pattern (a defect that recruits a second joint to hide itself) is the most
#: expensive, which is the coaching reality: a squat hidden behind lumbar flexion
#: is worse than a shallow squat, not better.
KIND_WEIGHTS: dict[str, float] = {
    "rom": 0.40,
    "asymmetry": 0.35,
    "tempo": 0.25,
    "jerk": 0.20,
    "instability": 0.30,
    "compensation": 0.45,
}

DEGRADATION_KINDS: tuple[str, ...] = tuple(KIND_WEIGHTS)

#: Maximum total quality loss. Keeping the label off 0 avoids a floor pile-up
#: that would make rank correlation look artificially good on the worst decile.
MAX_LOSS = 0.95


@dataclass
class Degradation:
    """One applied defect.

    Attributes:
        kind: One of :data:`DEGRADATION_KINDS`.
        severity: In ``[0, 1]``. Zero is a no-op and is never emitted.
        dofs: The DOF names this defect edits.
        phase_window: ``(start, end)`` within-repetition phase covered by the
            defect, in ``[0, 1]``. ``(0, 1)`` means the whole repetition.
        detail: Free-form extras for reporting (e.g. the compensating DOF).
    """

    kind: str
    severity: float
    dofs: tuple[str, ...]
    phase_window: tuple[float, float] = (0.0, 1.0)
    detail: dict[str, float | str] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        """This defect's contribution to the quality loss, before clipping."""
        return KIND_WEIGHTS[self.kind] * float(self.severity)

    @property
    def joints(self) -> tuple[int, ...]:
        """Joints *declared* by the edited DOFs (the strict ground-truth view).

        This is the narrow answer to "where is the defect": the joints whose
        Euler angles were literally changed. It is deliberately different from
        the *displacement* ground truth in
        :func:`saqa.data.generator.attribution_ground_truth`, which includes
        every joint the edit actually moved through the kinematic chain and
        floor contact. Both are reported, because a reader should be able to see
        the difference between "the joint I edited" and "the joints that moved".
        """
        return tuple(sorted({DOF[d][0] for d in self.dofs}))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "severity": round(float(self.severity), 4),
            "dofs": ",".join(self.dofs),
            "phase_start": round(self.phase_window[0], 3),
            "phase_end": round(self.phase_window[1], 3),
            "cost": round(self.cost, 4),
            **{f"detail_{k}": v for k, v in self.detail.items()},
        }


def quality_from_degradations(degradations: list[Degradation]) -> float:
    """The exact quality label in ``[1 - MAX_LOSS, 1]``.

    Args:
        degradations: Applied defects; an empty list gives 1.0.

    Returns:
        The label. Linear in each severity, hence monotone non-increasing in
        each -- deliberately, so that the model-side monotonicity test has an
        unambiguous reference.
    """
    loss = sum(d.cost for d in degradations)
    return float(1.0 - min(loss, MAX_LOSS))


def _phase_mask(phase: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Smooth ``[0, 1]`` weight selecting frames inside a phase window.

    A hard rectangular mask would inject a step discontinuity into the angle
    trajectory and therefore a spike into acceleration, which would make every
    windowed range-of-motion defect *also* a jerk defect and confound the two.
    The mask is raised-cosine tapered over 15% of the window for that reason.
    """
    start, end = window
    if end <= start:
        return np.zeros_like(phase)
    inside = ((phase >= start) & (phase <= end)).astype(np.float64)
    taper = max(0.15 * (end - start), 1e-6)
    ramp_in = np.clip((phase - start) / taper, 0.0, 1.0)
    ramp_out = np.clip((end - phase) / taper, 0.0, 1.0)
    x = np.minimum(ramp_in, ramp_out)
    smoothstep = x * x * (3.0 - 2.0 * x)
    return inside * smoothstep


def apply_amplitude_degradation(
    traj: dict[str, np.ndarray],
    deg: Degradation,
    phase: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Apply one amplitude-domain defect to the DOF trajectories, in place.

    Handles ``rom``, ``asymmetry``, ``compensation`` and ``jerk``. ``tempo`` and
    ``instability`` act on time and on the root respectively and are handled by
    the generator.

    Args:
        traj: DOF name -> ``(T,)`` radians. Mutated and returned.
        deg: The defect.
        phase: ``(T,)`` within-repetition phase.
        rng: Randomness for the jerk noise.

    Returns:
        ``traj``.
    """
    s = float(deg.severity)
    mask = _phase_mask(phase, deg.phase_window)

    if deg.kind in ("rom", "asymmetry"):
        # Shrink the DOF toward its own temporal mean: the joint still moves,
        # just less. Shrinking toward zero instead would change the posture's
        # offset as well as its range, which is a different defect.
        for name in deg.dofs:
            if name not in traj:
                continue
            x = traj[name]
            factor = 1.0 - 0.85 * s * mask
            traj[name] = x.mean() + (x - x.mean()) * factor
    elif deg.kind == "compensation":
        # The primary DOF loses range and a second DOF *gains* range to hide it.
        primary, secondary = deg.dofs[0], deg.dofs[1]
        if primary in traj:
            x = traj[primary]
            traj[primary] = x.mean() + (x - x.mean()) * (1.0 - 0.80 * s * mask)
        base = traj.get(secondary)
        if base is None:
            base = np.zeros_like(phase)
        amp = float(deg.detail.get("compensation_gain", 0.55))
        drive = traj.get(primary, base)
        drive = drive - drive.mean()
        scale = np.abs(drive).max()
        shape = drive / scale if scale > 1e-9 else np.zeros_like(drive)
        traj[secondary] = base + amp * s * mask * shape
    elif deg.kind == "jerk":
        for name in deg.dofs:
            if name not in traj:
                continue
            # Band-limited: a sum of a few high harmonics, not white noise. Real
            # tremor and control failure are oscillatory; white noise would be
            # removed by any smoothing and would not survive the sensor model as
            # a distinguishable defect.
            t = np.linspace(0.0, 1.0, phase.size, endpoint=False)
            wave = np.zeros_like(t)
            for _ in range(3):
                freq = rng.uniform(6.0, 16.0)
                wave += np.sin(2.0 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
            span = float(np.ptp(traj[name])) or 0.5
            traj[name] = traj[name] + 0.11 * s * span * mask * wave / 3.0
    return traj


def tempo_warp(num_frames: int, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Fractional frame indices implementing a timing/tempo error.

    A tempo defect is a *monotone reparameterisation of time*: the athlete
    rushes one part of the movement and drags another. It is implemented as
    ``u + a * sin(2*pi*u)`` clipped to remain monotone, so no frame order is
    reversed -- a non-monotone warp would be a different (and physically
    impossible) defect.

    Args:
        num_frames: ``T``.
        severity: In ``[0, 1]``; 0 returns the identity index.
        rng: Randomness for the sign of the rush.

    Returns:
        ``(T,)`` fractional indices in ``[0, T-1]``, strictly increasing.
    """
    u = np.linspace(0.0, 1.0, num_frames)
    a = 0.30 * float(severity) * (1.0 if rng.random() < 0.5 else -1.0)
    warped = u + a * np.sin(2.0 * np.pi * u) / (2.0 * np.pi)
    warped = np.clip(warped, 0.0, 1.0)
    warped = np.maximum.accumulate(warped)
    return warped * (num_frames - 1)


def resample_trajectories(
    traj: dict[str, np.ndarray], index: np.ndarray
) -> dict[str, np.ndarray]:
    """Linearly resample every DOF at fractional frame ``index``."""
    src = np.arange(next(iter(traj.values())).size, dtype=np.float64)
    return {k: np.interp(index, src, v) for k, v in traj.items()}


def postural_sway(
    num_frames: int, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Low-frequency root translation modelling loss of postural stability.

    Args:
        num_frames: ``T``.
        severity: In ``[0, 1]``.
        rng: Randomness.

    Returns:
        ``(T, 3)`` metres of pelvis translation. Vertical sway is a third of the
        horizontal, because balance loss is mostly a base-of-support problem.
    """
    t = np.linspace(0.0, 1.0, num_frames, endpoint=False)
    out = np.zeros((num_frames, 3), dtype=np.float64)
    for axis, gain in ((0, 1.0), (1, 0.33), (2, 1.0)):
        wave = np.zeros_like(t)
        for _ in range(2):
            freq = rng.uniform(0.8, 2.6)
            wave += np.sin(2.0 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        out[:, axis] = 0.055 * float(severity) * gain * wave / 2.0
    return out


def sample_degradations(
    action: str,
    rng: np.random.Generator,
    max_count: int = 3,
    severity_range: tuple[float, float] = (0.15, 1.0),
    allowed_kinds: tuple[str, ...] | None = None,
) -> list[Degradation]:
    """Draw a random, non-degenerate set of defects for one execution.

    Args:
        action: Action class, used to pick DOFs that the action actually moves.
        rng: Randomness.
        max_count: Upper bound on simultaneous defects. The count is drawn from
            ``0..max_count`` with a bias toward 1-2, because a set in which most
            sequences carry three defects would have almost no label spread near
            the top of the scale.
        severity_range: Bounds on drawn severity. The floor is above 0 so that
            every emitted defect is genuinely present -- a severity-0 defect
            would be an unlabelled no-op in the attribution ground truth.
        allowed_kinds: Restrict to these kinds (used by the held-out
            degradation-combination split and by ablations).

    Returns:
        A list of :class:`Degradation`, possibly empty. At most one defect per
        kind, so the combination identity is a set of kinds.
    """
    spec = ACTION_SPECS[action]
    kinds = list(allowed_kinds or DEGRADATION_KINDS)
    weights = np.array([0.10, 0.36, 0.34, 0.20], dtype=np.float64)[: max_count + 1]
    weights = weights / weights.sum()
    count = int(rng.choice(np.arange(len(weights)), p=weights))
    count = min(count, len(kinds))
    if count == 0:
        return []

    chosen = rng.choice(len(kinds), size=count, replace=False)
    out: list[Degradation] = []
    for ci in chosen:
        kind = kinds[int(ci)]
        sev = float(rng.uniform(*severity_range))
        window = (0.0, 1.0)
        if rng.random() < 0.5 and kind in ("rom", "jerk", "compensation"):
            start = float(rng.uniform(0.0, 0.55))
            window = (start, min(1.0, start + float(rng.uniform(0.3, 0.5))))

        if kind == "rom":
            # Bilateral by construction: a range-of-motion limitation reduces
            # the DOF *and its mirror* equally. Restricting one side only is the
            # separate ``asymmetry`` kind, and if ``rom`` were also one-sided the
            # two kinds would be mechanically identical while carrying different
            # quality costs -- an unlearnable label, and a silent one.
            dof = str(rng.choice(spec.primary_dofs))
            dofs = (dof,) if dof not in spec.mirror else (dof, spec.mirror[dof])
            out.append(Degradation(kind, sev, dofs, window))
        elif kind == "asymmetry":
            sided = [d for d in spec.primary_dofs if d in spec.mirror]
            if not sided:
                continue
            dof = str(rng.choice(sided))
            out.append(Degradation(kind, sev, (dof,), (0.0, 1.0),
                                   {"mirror_of": spec.mirror[dof]}))
        elif kind == "compensation":
            primary = str(rng.choice(spec.primary_dofs))
            pool = [d for d in ("spine_flex", "spine_lat", "spine_rot", "neck_flex")
                    if d != primary]
            secondary = str(rng.choice(pool))
            out.append(Degradation(kind, sev, (primary, secondary), window,
                                   {"compensation_gain": 0.55}))
        elif kind == "jerk":
            dofs = tuple(str(d) for d in rng.choice(
                spec.primary_dofs, size=min(2, len(spec.primary_dofs)), replace=False))
            out.append(Degradation(kind, sev, dofs, window))
        elif kind == "tempo":
            out.append(Degradation(kind, sev, (), (0.0, 1.0)))
        elif kind == "instability":
            out.append(Degradation(kind, sev, (), (0.0, 1.0)))
    return out


def combination_key(degradations: list[Degradation]) -> str:
    """Canonical name for the *set of kinds* present, for held-out splits.

    Severity is continuous, so a "held-out degradation combination" can only
    mean a held-out set of kinds. ``"clean"`` for no defects.
    """
    kinds = sorted({d.kind for d in degradations})
    return "+".join(kinds) if kinds else "clean"
