"""Handcrafted kinematic features into a gradient-boosted regressor.

This baseline is often the strongest thing in a skeleton-based paper and it is
usually reported as an afterthought or not at all. It is reported honestly here,
including where it beats the graph model.

The feature set is built from the quantities a biomechanist would name, and it is
deliberately *aligned with the generator's defect taxonomy* -- joint angle range
for range-of-motion, left-right differences for asymmetry, jerk for smoothness,
pelvis path length for stability, phase-timing for tempo. That makes it a strong
and slightly favoured baseline: it is being handed the vocabulary of the
degradations. If a learned model cannot beat features that were designed with
knowledge of the label function, that is the interesting finding, not a nuisance.

Features are computed **per action class in the same code path**, with the class
one-hot appended, so the regressor can learn a per-class mapping without needing
five separate models.
"""

from __future__ import annotations

import numpy as np

from ..data.kinematics import joint_angle
from ..data.skeleton import JOINT_NAMES, LEFT_RIGHT_PAIRS, NUM_JOINTS

#: Angle triples ``(a, b, c)`` measured at joint ``b``, with a readable name.
ANGLE_TRIPLES: tuple[tuple[str, int, int, int], ...] = (
    ("l_knee", 11, 12, 13),
    ("r_knee", 14, 15, 16),
    ("l_hip", 2, 11, 12),
    ("r_hip", 2, 14, 15),
    ("l_elbow", 5, 6, 7),
    ("r_elbow", 8, 9, 10),
    ("l_shoulder", 2, 5, 6),
    ("r_shoulder", 8, 2, 9),
    ("trunk", 0, 2, 4),
)


def _to_tvc(seq: np.ndarray) -> np.ndarray:
    a = np.asarray(seq, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"expected 3-D sequence, got {a.shape}")
    if a.shape[0] == 3 and a.shape[2] == NUM_JOINTS:
        return a.transpose(1, 2, 0)
    return a


def kinematic_features(seq: np.ndarray, action: str | None = None,
                       actions: tuple[str, ...] = ()) -> tuple[np.ndarray, list[str]]:
    """Feature vector and matching names for one sequence.

    Args:
        seq: ``(T, V, 3)`` or ``(3, T, V)`` canonicalised coordinates.
        action: Action class, one-hot encoded when ``actions`` is given.
        actions: The full ordered class list.

    Returns:
        ``(features, names)``. Non-finite values (e.g. a degenerate joint angle)
        are replaced by 0 *and counted* in the ``n_nonfinite`` feature, so the
        regressor is told that something was undefined rather than being handed a
        silent zero.
    """
    x = _to_tvc(seq)
    t = x.shape[0]
    vel = np.diff(x, axis=0, prepend=x[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    jerk = np.diff(acc, axis=0, prepend=acc[:1])

    feats: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        feats.append(float(value))
        names.append(name)

    # --- range of motion, per named joint angle -----------------------------
    angles: dict[str, np.ndarray] = {}
    for name, a, b, c in ANGLE_TRIPLES:
        ang = joint_angle(x, a, b, c)
        angles[name] = ang
        finite = ang[np.isfinite(ang)]
        add(f"angle_range__{name}", np.ptp(finite) if finite.size else 0.0)
        add(f"angle_min__{name}", finite.min() if finite.size else 0.0)
        add(f"angle_max__{name}", finite.max() if finite.size else 0.0)
        add(f"angle_std__{name}", finite.std() if finite.size else 0.0)

    # --- symmetry: left-right differences ----------------------------------
    for left, right in (("l_knee", "r_knee"), ("l_hip", "r_hip"),
                        ("l_elbow", "r_elbow"), ("l_shoulder", "r_shoulder")):
        la, ra = angles[left], angles[right]
        lf, rf = la[np.isfinite(la)], ra[np.isfinite(ra)]
        rl = np.ptp(lf) if lf.size else 0.0
        rr = np.ptp(rf) if rf.size else 0.0
        add(f"sym_range_diff__{left}_{right}", abs(rl - rr))
        add(f"sym_range_ratio__{left}_{right}", rl / rr if rr > 1e-6 else 0.0)
    for a, b in LEFT_RIGHT_PAIRS:
        # Mirror the right side in x before comparing, otherwise a perfectly
        # symmetric movement registers as maximally asymmetric.
        mirrored = x[:, b].copy()
        mirrored[:, 0] *= -1.0
        add(f"sym_pos_rms__{JOINT_NAMES[a]}", float(np.sqrt(((x[:, a] - mirrored) ** 2).mean())))

    # --- smoothness ---------------------------------------------------------
    add("jerk_rms", float(np.sqrt((jerk**2).mean())))
    add("jerk_rms_upper", float(np.sqrt((jerk[:, 5:11] ** 2).mean())))
    add("jerk_rms_lower", float(np.sqrt((jerk[:, 11:] ** 2).mean())))
    add("accel_rms", float(np.sqrt((acc**2).mean())))
    add("speed_mean", float(np.linalg.norm(vel, axis=-1).mean()))
    add("speed_max", float(np.linalg.norm(vel, axis=-1).max()))
    # Dimensionless jerk is the standard smoothness index: it removes the
    # amplitude and duration of the movement, so it separates "jerky" from
    # "fast", which raw jerk RMS does not.
    speed = np.linalg.norm(vel, axis=-1).mean(axis=-1)
    peak = max(float(speed.max()), 1e-8)
    add("dimensionless_jerk", float(np.sqrt((jerk**2).mean()) * t / peak))

    # --- postural stability -------------------------------------------------
    pelvis = x[:, 0]
    add("pelvis_path_length", float(np.linalg.norm(np.diff(pelvis, axis=0), axis=-1).sum()))
    add("pelvis_sway_x", float(pelvis[:, 0].std()))
    add("pelvis_sway_z", float(pelvis[:, 2].std()))
    add("pelvis_height_range", float(np.ptp(pelvis[:, 1])))
    com = x.mean(axis=1)
    add("com_sway", float(np.linalg.norm(com - com.mean(axis=0), axis=-1).mean()))

    # --- tempo / phase timing ----------------------------------------------
    # Where in the clip the movement's energy sits, and how spread out it is. A
    # rushed concentric phase moves the centroid; an uneven tempo widens it.
    energy = np.linalg.norm(vel, axis=-1).mean(axis=-1)
    total = max(float(energy.sum()), 1e-12)
    idx = np.arange(t, dtype=np.float64) / max(t - 1, 1)
    centroid = float((energy * idx).sum() / total)
    add("energy_centroid", centroid)
    add("energy_spread", float(np.sqrt(((idx - centroid) ** 2 * energy).sum() / total)))
    add("energy_peak_time", float(idx[int(np.argmax(energy))]))
    add("energy_kurtosis", float(((energy - energy.mean()) ** 4).mean()
                                 / max(energy.std() ** 4, 1e-12)))

    # --- per-joint displacement profile ------------------------------------
    disp = np.linalg.norm(x - x.mean(axis=0, keepdims=True), axis=-1).mean(axis=0)
    for v in range(NUM_JOINTS):
        add(f"disp__{JOINT_NAMES[v]}", float(disp[v]))

    arr = np.array(feats, dtype=np.float64)
    n_bad = int((~np.isfinite(arr)).sum())
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.append(arr, float(n_bad))
    names.append("n_nonfinite")

    if actions:
        onehot = np.array([1.0 if action == a else 0.0 for a in actions])
        arr = np.concatenate([arr, onehot])
        names += [f"action__{a}" for a in actions]
    return arr, names


def feature_matrix(
    sequences: np.ndarray, actions_per_item: list[str] | None = None,
    action_classes: tuple[str, ...] = ()
) -> tuple[np.ndarray, list[str]]:
    """Stack :func:`kinematic_features` over a batch.

    Args:
        sequences: ``(N, 3, T, V)`` or ``(N, T, V, 3)``.
        actions_per_item: Action class per sequence.
        action_classes: Ordered class list for the one-hot.

    Returns:
        ``(X, names)`` with ``X`` of shape ``(N, F)``.
    """
    rows, names = [], []
    for i in range(len(sequences)):
        act = actions_per_item[i] if actions_per_item is not None else None
        row, names = kinematic_features(sequences[i], act, action_classes)
        rows.append(row)
    return np.stack(rows), names
