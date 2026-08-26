"""Forward kinematics on the fixed skeleton, plus canonicalisation.

Everything downstream -- the action library, every degradation, the exact
attribution ground truth -- is a function of one array of Euler angles of shape
``(T, V, 3)``. That is the whole design: **the generative parameters are joint
angles, so a degradation is a change to a named angle trajectory, and the
resulting change in joint positions can be computed exactly by running forward
kinematics twice.** No labelling, no annotation, no approximation.

Conventions
-----------
* World frame: ``+Y`` up, ``+X`` to the subject's right, ``+Z`` forward (the
  direction the subject faces).
* Euler order is intrinsic ``XYZ``: ``R = Rx(ax) @ Ry(ay) @ Rz(az)``. ``X`` is
  the sagittal (flexion) axis, ``Z`` the frontal (abduction) axis, ``Y`` the
  long-axis (rotation) axis.
* The rotation stored at joint ``j`` rotates the **bone that ends at j**, and
  therefore also every descendant of ``j``. So ``angles[:, L_KNEE]`` steers the
  thigh (hip flexion) and ``angles[:, L_ANKLE]`` steers the shank (knee
  flexion). This is the standard "rotation belongs to the child" convention and
  it is why the DOF names in :mod:`saqa.data.actions` are segment names.
* After the kinematic pass the whole body is translated so the lower ankle sits
  on ``y = 0``. Without that, bending the knees would leave the pelvis pinned in
  space and a squat would look like a levitation. Grounding is what makes a
  reduced-knee-flexion degradation lower the *whole body's* trajectory, which is
  what it does in reality -- and it is also why attribution ground truth is
  measured, not declared (see :mod:`saqa.data.degradations`).
"""

from __future__ import annotations

import numpy as np

from .skeleton import NUM_JOINTS, PARENTS, bone_lengths

#: Unit rest direction of the bone ending at each joint, in the parent's frame.
REST_DIRECTIONS: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],  # 0 pelvis (root, no bone)
        [0.0, 1.0, 0.0],  # 1 spine: up
        [0.0, 1.0, 0.0],  # 2 thorax: up
        [0.0, 1.0, 0.0],  # 3 neck: up
        [0.0, 1.0, 0.0],  # 4 head: up
        [-1.0, 0.0, 0.0],  # 5 l_shoulder: left
        [0.0, -1.0, 0.0],  # 6 l_elbow: down (upper arm)
        [0.0, -1.0, 0.0],  # 7 l_wrist: down (forearm)
        [1.0, 0.0, 0.0],  # 8 r_shoulder: right
        [0.0, -1.0, 0.0],  # 9 r_elbow
        [0.0, -1.0, 0.0],  # 10 r_wrist
        [-1.0, 0.0, 0.0],  # 11 l_hip: left
        [0.0, -1.0, 0.0],  # 12 l_knee: down (thigh)
        [0.0, -1.0, 0.0],  # 13 l_ankle: down (shank)
        [1.0, 0.0, 0.0],  # 14 r_hip: right
        [0.0, -1.0, 0.0],  # 15 r_knee
        [0.0, -1.0, 0.0],  # 16 r_ankle
    ],
    dtype=np.float64,
)


def euler_to_matrix(angles: np.ndarray) -> np.ndarray:
    """Intrinsic XYZ Euler angles to rotation matrices.

    Args:
        angles: ``(..., 3)`` array of radians ``(ax, ay, az)``.

    Returns:
        ``(..., 3, 3)`` rotation matrices ``Rx(ax) @ Ry(ay) @ Rz(az)``.
    """
    a = np.asarray(angles, dtype=np.float64)
    if a.shape[-1] != 3:
        raise ValueError(f"angles must end in a 3-axis, got {a.shape}")
    cx, cy, cz = np.cos(a[..., 0]), np.cos(a[..., 1]), np.cos(a[..., 2])
    sx, sy, sz = np.sin(a[..., 0]), np.sin(a[..., 1]), np.sin(a[..., 2])

    r = np.empty(a.shape[:-1] + (3, 3), dtype=np.float64)
    r[..., 0, 0] = cy * cz
    r[..., 0, 1] = -cy * sz
    r[..., 0, 2] = sy
    r[..., 1, 0] = sx * sy * cz + cx * sz
    r[..., 1, 1] = -sx * sy * sz + cx * cz
    r[..., 1, 2] = -sx * cy
    r[..., 2, 0] = -cx * sy * cz + sx * sz
    r[..., 2, 1] = cx * sy * sz + sx * cz
    r[..., 2, 2] = cx * cy
    return r


def forward_kinematics(
    angles: np.ndarray,
    root_translation: np.ndarray | None = None,
    limb_scale: np.ndarray | None = None,
    ground: bool = True,
) -> np.ndarray:
    """Joint positions from joint angles.

    Args:
        angles: ``(T, V, 3)`` Euler angles in radians.
        root_translation: ``(T, 3)`` world translation of the pelvis, added
            before grounding. ``None`` means zero.
        limb_scale: ``(V,)`` multiplicative bone-length factors (inter-subject
            variation). ``None`` means the default 1.75 m adult.
        ground: Translate each frame so the lower ankle sits at ``y = 0``.

    Returns:
        ``(T, V, 3)`` positions in metres.

    Raises:
        ValueError: on a shape mismatch.
    """
    a = np.asarray(angles, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != NUM_JOINTS or a.shape[2] != 3:
        raise ValueError(f"angles must be (T, {NUM_JOINTS}, 3), got {a.shape}")
    t = a.shape[0]

    lengths = bone_lengths(limb_scale)
    offsets = REST_DIRECTIONS * lengths[:, None]  # (V, 3)
    rot_local = euler_to_matrix(a)  # (T, V, 3, 3)

    rot_global = np.empty_like(rot_local)
    pos = np.zeros((t, NUM_JOINTS, 3), dtype=np.float64)

    rot_global[:, 0] = rot_local[:, 0]
    for j in range(1, NUM_JOINTS):
        p = PARENTS[j]
        rot_global[:, j] = rot_global[:, p] @ rot_local[:, j]
        pos[:, j] = pos[:, p] + rot_global[:, j] @ offsets[j]

    if root_translation is not None:
        rt = np.asarray(root_translation, dtype=np.float64)
        if rt.shape != (t, 3):
            raise ValueError(f"root_translation must be ({t}, 3), got {rt.shape}")
        pos += rt[:, None, :]

    if ground:
        floor = np.minimum(pos[:, 13, 1], pos[:, 16, 1])
        pos[:, :, 1] -= floor[:, None]
    return pos


def canonicalise(positions: np.ndarray, scale_to_height: bool = True) -> np.ndarray:
    """Remove global translation and (optionally) subject size.

    Two nuisance factors have to go before a model sees the sequence, and *how*
    they go matters:

    * **Translation** is removed by subtracting the pelvis position averaged
      over the *whole sequence*, not per frame. Per-frame subtraction would pin
      the pelvis at the origin and destroy squat depth, hip drop, and every
      other whole-body signal -- the single most damaging normalisation choice
      available here.
    * **Size** is removed by dividing by the subject's standing height estimate
      (mean pelvis-to-head distance plus mean pelvis height). Inter-subject limb
      length is a nuisance the model should be invariant to; leaving it in makes
      the generator's subject factor leak into the score.

    Args:
        positions: ``(T, V, 3)`` or ``(N, T, V, 3)`` metres.
        scale_to_height: Divide by the height estimate.

    Returns:
        Array of the same shape, in units of body height (dimensionless) when
        ``scale_to_height`` is set.
    """
    p = np.asarray(positions, dtype=np.float64)
    squeeze = p.ndim == 3
    if squeeze:
        p = p[None]
    if p.ndim != 4:
        raise ValueError(f"positions must be (T,V,3) or (N,T,V,3), got {p.shape}")

    centre = p[:, :, 0, :].mean(axis=1, keepdims=True)[:, :, None, :]  # (N,1,1,3)
    out = p - centre
    if scale_to_height:
        torso = np.linalg.norm(p[:, :, 4, :] - p[:, :, 0, :], axis=-1).mean(axis=1)
        legs = p[:, :, 0, 1].mean(axis=1)
        height = np.maximum(torso + legs, 1e-6)
        out = out / height[:, None, None, None]
    return out[0] if squeeze else out


def joint_angle(positions: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Interior angle at joint ``b`` in the chain ``a-b-c``, in radians.

    Args:
        positions: ``(T, V, 3)``.
        a: First joint index.
        b: Vertex joint index.
        c: Third joint index.

    Returns:
        ``(T,)`` angles in ``[0, pi]``. Degenerate (zero-length) segments give
        ``NaN`` rather than a fabricated angle.
    """
    p = np.asarray(positions, dtype=np.float64)
    u = p[:, a] - p[:, b]
    v = p[:, c] - p[:, b]
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    denom = nu * nv
    cos = np.divide((u * v).sum(-1), denom, out=np.full(nu.shape, np.nan), where=denom > 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def subject_limb_scale(rng: np.random.Generator, spread: float = 0.10) -> np.ndarray:
    """Draw a plausible per-joint limb-length scaling for one subject.

    Limb lengths are correlated in real bodies -- long femurs come with long
    tibias -- so this draws one global stature factor plus mildly correlated
    segment factors, and keeps left and right *identical*. Random per-bone noise
    would create anatomically impossible and, worse, left-right asymmetric
    subjects, which would be indistinguishable from the asymmetry degradation.

    Args:
        rng: Source of randomness.
        spread: Standard deviation of the multiplicative factors.

    Returns:
        ``(V,)`` scale factors, all positive.
    """
    stature = 1.0 + rng.normal(0.0, spread)
    trunk = stature * (1.0 + rng.normal(0.0, spread * 0.5))
    arm = stature * (1.0 + rng.normal(0.0, spread * 0.6))
    leg = stature * (1.0 + rng.normal(0.0, spread * 0.6))

    scale = np.ones(NUM_JOINTS, dtype=np.float64)
    scale[[1, 2, 3, 4]] = trunk
    scale[[5, 6, 7, 8, 9, 10]] = arm
    scale[[11, 12, 13, 14, 15, 16]] = leg
    return np.clip(scale, 0.6, 1.5)
