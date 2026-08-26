"""Parametric joint-angle trajectories for five distinct action classes.

Each action is a function ``phase -> angles``, where ``phase`` runs over ``[0, R]``
for ``R`` repetitions and ``angles`` has shape ``(T, V, 3)``. The trajectories
are deliberately smooth and low-order: they are a *scaffold* on which
degradations are applied, and the interesting variance in this project comes
from the degradations and the sensor model, not from the elaborateness of the
clean motion.

The five classes were chosen to be kinematically dissimilar so that a
per-action-class breakdown is informative:

============  ==========================================================
squat         bilateral, sagittal, whole-body, one dominant DOF group
overhead_press  bilateral, upper body only, legs near-static
lunge         *intrinsically asymmetric* -- the left-right symmetry feature
              is uninformative here by construction, which is exactly why it
              is in the set
gait          *cyclic and anti-phase* -- the only class where inter-limb
              timing carries the quality signal
throw         *ballistic and single-sided*, with a sharp velocity peak that
              punishes any model that only looks at pose extrema
============  ==========================================================

DOF naming follows the "rotation belongs to the child" convention of
:mod:`saqa.data.kinematics`: ``l_thigh`` is ``angles[:, L_KNEE]`` (it steers the
hip->knee segment, i.e. hip flexion) and ``l_shank`` is ``angles[:, L_ANKLE]``
(knee flexion).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skeleton import NUM_JOINTS, joint_index

#: Human-readable segment name -> (joint index, Euler axis). The axis is 0 for
#: sagittal flexion, 1 for long-axis rotation, 2 for frontal abduction.
DOF: dict[str, tuple[int, int]] = {
    "spine_flex": (joint_index("spine"), 0),
    "spine_rot": (joint_index("thorax"), 1),
    "spine_lat": (joint_index("spine"), 2),
    "neck_flex": (joint_index("neck"), 0),
    "l_upperarm_flex": (joint_index("l_elbow"), 0),
    "l_upperarm_abd": (joint_index("l_elbow"), 2),
    "l_forearm_flex": (joint_index("l_wrist"), 0),
    "r_upperarm_flex": (joint_index("r_elbow"), 0),
    "r_upperarm_abd": (joint_index("r_elbow"), 2),
    "r_forearm_flex": (joint_index("r_wrist"), 0),
    "l_thigh_flex": (joint_index("l_knee"), 0),
    "l_thigh_abd": (joint_index("l_knee"), 2),
    "l_shank_flex": (joint_index("l_ankle"), 0),
    "r_thigh_flex": (joint_index("r_knee"), 0),
    "r_thigh_abd": (joint_index("r_knee"), 2),
    "r_shank_flex": (joint_index("r_ankle"), 0),
}

DOF_NAMES: tuple[str, ...] = tuple(DOF)

ACTION_CLASSES: tuple[str, ...] = ("squat", "overhead_press", "lunge", "gait", "throw")


@dataclass(frozen=True)
class ActionSpec:
    """Static description of one action class.

    Attributes:
        name: Class name.
        repetitions: How many cycles fill the sequence.
        primary_dofs: The DOFs that carry the action's quality signal. Range-of-
            motion and asymmetry degradations are drawn from this set, so a
            degradation is always applied somewhere the action actually moves.
            Degrading a static DOF would be invisible and would poison the
            attribution ground truth with unlabelled no-ops.
        cyclic: Whether the trajectory returns to its start (gait does; a throw
            does not). Timing degradations behave differently in the two cases.
    """

    name: str
    repetitions: int
    primary_dofs: tuple[str, ...]
    cyclic: bool = False
    mirror: dict[str, str] = field(default_factory=dict)


def _sym(pairs: tuple[tuple[str, str], ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a, b in pairs:
        out[a] = b
        out[b] = a
    return out


_LEG_SYM = _sym((("l_thigh_flex", "r_thigh_flex"), ("l_shank_flex", "r_shank_flex"),
                 ("l_thigh_abd", "r_thigh_abd")))
_ARM_SYM = _sym((("l_upperarm_flex", "r_upperarm_flex"), ("l_forearm_flex", "r_forearm_flex"),
                 ("l_upperarm_abd", "r_upperarm_abd")))

ACTION_SPECS: dict[str, ActionSpec] = {
    "squat": ActionSpec(
        "squat",
        repetitions=2,
        primary_dofs=("l_thigh_flex", "r_thigh_flex", "l_shank_flex", "r_shank_flex",
                      "spine_flex"),
        cyclic=True,
        mirror=_LEG_SYM,
    ),
    "overhead_press": ActionSpec(
        "overhead_press",
        repetitions=2,
        primary_dofs=("l_upperarm_abd", "r_upperarm_abd", "l_forearm_flex",
                      "r_forearm_flex", "spine_flex"),
        cyclic=True,
        mirror=_ARM_SYM,
    ),
    "lunge": ActionSpec(
        "lunge",
        repetitions=2,
        primary_dofs=("l_thigh_flex", "r_thigh_flex", "l_shank_flex", "spine_flex"),
        cyclic=True,
        mirror=_LEG_SYM,
    ),
    "gait": ActionSpec(
        "gait",
        repetitions=3,
        primary_dofs=("l_thigh_flex", "r_thigh_flex", "l_shank_flex", "r_shank_flex",
                      "l_upperarm_flex", "r_upperarm_flex"),
        cyclic=True,
        mirror={**_LEG_SYM, **_ARM_SYM},
    ),
    "throw": ActionSpec(
        "throw",
        repetitions=1,
        primary_dofs=("r_upperarm_flex", "r_upperarm_abd", "r_forearm_flex", "spine_rot",
                      "spine_flex"),
        cyclic=False,
        mirror=_ARM_SYM,
    ),
}


def _bell(p: np.ndarray, centre: float = 0.5, width: float = 0.22) -> np.ndarray:
    """Smooth unimodal bump on ``[0, 1]``, peaking at ``centre``."""
    return np.exp(-0.5 * ((p - centre) / width) ** 2)


def _updown(p: np.ndarray) -> np.ndarray:
    """Raised cosine: 0 at the ends, 1 in the middle. One rep of a lift."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * p))


def dof_trajectories(action: str, num_frames: int, style: float = 0.0) -> dict[str, np.ndarray]:
    """Clean per-DOF angle trajectories for one action, in radians.

    Args:
        action: One of :data:`ACTION_CLASSES`.
        num_frames: Sequence length ``T``.
        style: Inter-subject style factor in roughly ``[-1, 1]``. It scales
            amplitudes and shifts timing slightly, so two clean executions of
            the same action by different subjects are not identical. Style is
            *not* a quality defect and does not enter the quality label -- which
            is precisely the confound a reference-free model has to survive.

    Returns:
        Mapping from DOF name to a ``(T,)`` trajectory. DOFs absent from the
        mapping are zero.

    Raises:
        ValueError: on an unknown action.
    """
    if action not in ACTION_SPECS:
        raise ValueError(f"Unknown action {action!r}; known: {ACTION_CLASSES}")
    spec = ACTION_SPECS[action]
    t = np.linspace(0.0, 1.0, num_frames, endpoint=False)
    reps = spec.repetitions
    # Phase within the current repetition, in [0, 1).
    p = (t * reps) % 1.0
    amp = 1.0 + 0.15 * style
    out: dict[str, np.ndarray] = {}

    if action == "squat":
        depth = _updown(p) * amp
        out["l_thigh_flex"] = -1.45 * depth
        out["r_thigh_flex"] = -1.45 * depth
        out["l_shank_flex"] = 1.60 * depth
        out["r_shank_flex"] = 1.60 * depth
        out["spine_flex"] = -0.32 * depth
        out["l_upperarm_flex"] = -0.55 * depth
        out["r_upperarm_flex"] = -0.55 * depth
    elif action == "overhead_press":
        lift = _updown(p) * amp
        out["l_upperarm_abd"] = 2.55 * lift
        out["r_upperarm_abd"] = -2.55 * lift
        out["l_forearm_flex"] = -1.75 * (1.0 - lift) - 0.10
        out["r_forearm_flex"] = -1.75 * (1.0 - lift) - 0.10
        out["spine_flex"] = 0.10 * lift
        out["l_thigh_flex"] = -0.05 * lift
        out["r_thigh_flex"] = -0.05 * lift
    elif action == "lunge":
        drop = _updown(p) * amp
        out["l_thigh_flex"] = -1.05 * drop
        out["l_shank_flex"] = 1.35 * drop
        out["r_thigh_flex"] = 0.45 * drop
        out["r_shank_flex"] = 1.55 * drop
        out["spine_flex"] = -0.12 * drop
        out["l_upperarm_flex"] = -0.25 * drop
        out["r_upperarm_flex"] = 0.25 * drop
    elif action == "gait":
        ph = 2.0 * np.pi * (t * reps)
        swing = amp * np.sin(ph)
        out["l_thigh_flex"] = -0.55 * swing
        out["r_thigh_flex"] = 0.55 * swing
        out["l_shank_flex"] = 1.00 * np.clip(np.sin(ph - 0.9), 0.0, None) * amp
        out["r_shank_flex"] = 1.00 * np.clip(np.sin(ph - 0.9 + np.pi), 0.0, None) * amp
        out["l_upperarm_flex"] = 0.40 * np.sin(ph + np.pi) * amp
        out["r_upperarm_flex"] = 0.40 * np.sin(ph) * amp
        out["spine_rot"] = 0.12 * np.sin(ph)
    else:  # throw
        wind = _bell(t, 0.30, 0.16)
        release = _bell(t, 0.62, 0.10)
        out["r_upperarm_flex"] = (1.65 * wind - 1.15 * release) * amp
        out["r_upperarm_abd"] = -1.20 * (wind + 0.4 * release) * amp
        out["r_forearm_flex"] = (-1.85 * wind + 0.65 * release) * amp
        out["spine_rot"] = (0.55 * wind - 0.45 * release) * amp
        out["spine_flex"] = -0.22 * release * amp
        out["l_upperarm_flex"] = 0.45 * wind * amp
        out["r_thigh_flex"] = -0.20 * release * amp
    return out


def assemble_angles(traj: dict[str, np.ndarray], num_frames: int) -> np.ndarray:
    """Scatter per-DOF trajectories into a ``(T, V, 3)`` Euler-angle array.

    Args:
        traj: DOF name -> ``(T,)`` trajectory.
        num_frames: ``T``.

    Returns:
        ``(T, NUM_JOINTS, 3)`` angles, zero where no DOF wrote.

    Raises:
        KeyError: on an unknown DOF name.
        ValueError: on a trajectory of the wrong length.
    """
    angles = np.zeros((num_frames, NUM_JOINTS, 3), dtype=np.float64)
    for name, values in traj.items():
        if name not in DOF:
            raise KeyError(f"Unknown DOF {name!r}")
        v = np.asarray(values, dtype=np.float64)
        if v.shape != (num_frames,):
            raise ValueError(f"DOF {name!r} must be ({num_frames},), got {v.shape}")
        j, axis = DOF[name]
        angles[:, j, axis] += v
    return angles


def phase_of(action: str, num_frames: int) -> np.ndarray:
    """Within-repetition phase in ``[0, 1)`` for each frame.

    The phase is what makes "per-phase" attribution meaningful: a degradation
    confined to the bottom of a squat lives at phase ~0.5 of every repetition,
    and localisation error is measured against that.
    """
    reps = ACTION_SPECS[action].repetitions
    t = np.linspace(0.0, 1.0, num_frames, endpoint=False)
    return (t * reps) % 1.0
