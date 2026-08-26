"""Skeleton topology: joints, bones, adjacency partitions, limb scaling.

The topology is fixed and small (17 joints) on purpose. Action-quality
assessment is not a pose-estimation problem: the interesting variation lives in
*how* a limb moves, not in how many keypoints describe it. A 17-joint
kinematic-tree skeleton is enough to express every degradation this project
studies, keeps forward kinematics analytic, and keeps the graph small enough
that a hand-rolled adjacency-matrix message pass is faster than any library.

Joint order (index: name), chosen so that the kinematic tree is a valid
topological order -- every joint's parent has a lower index, which is what lets
forward kinematics run as a single left-to-right pass:

    0  pelvis      (root)
    1  spine
    2  thorax
    3  neck
    4  head
    5  l_shoulder   8  r_shoulder
    6  l_elbow      9  r_elbow
    7  l_wrist     10  r_wrist
    11 l_hip       14  r_hip
    12 l_knee      15  r_knee
    13 l_ankle     16  r_ankle
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "spine",
    "thorax",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_hip",
    "r_knee",
    "r_ankle",
)

NUM_JOINTS = len(JOINT_NAMES)

#: ``PARENTS[j]`` is the index of ``j``'s parent, or ``-1`` for the root.
PARENTS: tuple[int, ...] = (-1, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11, 12, 0, 14, 15)

#: Default bone length from each joint to its parent, in metres. The root's
#: entry is 0. Numbers are a ~1.75 m adult; inter-subject variation scales them.
BONE_LENGTHS: tuple[float, ...] = (
    0.00,  # pelvis (root)
    0.12,  # spine
    0.16,  # thorax
    0.10,  # neck
    0.12,  # head
    0.17,  # l_shoulder (from thorax, lateral)
    0.28,  # l_elbow
    0.26,  # l_wrist
    0.17,  # r_shoulder
    0.28,  # r_elbow
    0.26,  # r_wrist
    0.10,  # l_hip (from pelvis, lateral)
    0.42,  # l_knee
    0.40,  # l_ankle
    0.10,  # r_hip
    0.42,  # r_knee
    0.40,  # r_ankle
)

#: Symmetric joint pairs, used by the asymmetry degradation and the symmetry
#: kinematic feature.
LEFT_RIGHT_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 8),
    (6, 9),
    (7, 10),
    (11, 14),
    (12, 15),
    (13, 16),
)

#: Named joint groups used for occlusion attribution and reporting.
JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "trunk": (0, 1, 2, 3, 4),
    "left_arm": (5, 6, 7),
    "right_arm": (8, 9, 10),
    "left_leg": (11, 12, 13),
    "right_leg": (14, 15, 16),
}


def joint_index(name: str) -> int:
    """Index of a joint by name.

    Raises:
        KeyError: if ``name`` is not a joint of this skeleton.
    """
    try:
        return JOINT_NAMES.index(name)
    except ValueError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown joint {name!r}; known: {JOINT_NAMES}") from exc


def bones() -> tuple[tuple[int, int], ...]:
    """The ``(parent, child)`` edge list, excluding the root's virtual edge."""
    return tuple((PARENTS[j], j) for j in range(NUM_JOINTS) if PARENTS[j] >= 0)


def hop_distance_to_root() -> np.ndarray:
    """Number of bones between each joint and the pelvis.

    Used by the *spatial configuration* partitioning of Yan et al. (2018): an
    edge is centripetal if it moves toward the body centre and centrifugal if it
    moves away. The pelvis is the body centre here, which is the usual choice
    for a full-body skeleton (the original paper uses the skeleton's centre of
    gravity; for this topology the pelvis *is* that joint).
    """
    dist = np.zeros(NUM_JOINTS, dtype=np.int64)
    for j in range(NUM_JOINTS):
        p, d = PARENTS[j], 0
        while p >= 0:
            d += 1
            p = PARENTS[p]
        dist[j] = d
    return dist


def adjacency(partitions: str = "spatial", self_loops: bool = True) -> np.ndarray:
    """Normalised adjacency stack ``A`` of shape ``(K, V, V)``.

    Args:
        partitions: One of

            * ``"identity"`` -- ``K=1``, self-loops only. No message passing at
              all; the ablation that shows whether the graph matters.
            * ``"uniform"`` -- ``K=1``, self-loops plus all bones in one
              partition. Message passing with no direction information.
            * ``"spatial"`` -- ``K=3``: self, centripetal (child->parent, i.e.
              toward the pelvis) and centrifugal (parent->child). This is the
              partitioning of Yan et al. (2018) and the default.
        self_loops: Include the identity partition. Only meaningful for
            ``"spatial"``/``"uniform"``; ``"identity"`` is self-loops by
            definition.

    Returns:
        ``A`` with each partition row-normalised, i.e.
        ``A[k] = D_k^{-1} * Adj_k`` with ``D_k`` the row degree and zero rows
        left as zero. Row normalisation (rather than the symmetric
        ``D^{-1/2} A D^{-1/2}``) keeps each partition an *average* over
        neighbours, so a joint with three children does not shout louder than a
        joint with one -- which matters here because the shoulders and hips are
        high-degree and the wrists are leaves.

    Raises:
        ValueError: on an unknown partition scheme.
    """
    v = NUM_JOINTS
    eye = np.eye(v, dtype=np.float64)

    if partitions == "identity":
        return eye[None, :, :]

    inward = np.zeros((v, v), dtype=np.float64)
    outward = np.zeros((v, v), dtype=np.float64)
    dist = hop_distance_to_root()
    for p, c in bones():
        # A[i, j] carries information from j into i (see stgcn.SpatialGraphConv).
        if dist[p] < dist[c]:
            inward[p, c] = 1.0  # child -> parent: centripetal
            outward[c, p] = 1.0  # parent -> child: centrifugal

    if partitions == "uniform":
        merged = inward + outward + (eye if self_loops else 0.0)
        return _row_normalise(merged[None, :, :])
    if partitions == "spatial":
        parts = [inward, outward]
        if self_loops:
            parts.insert(0, eye)
        return _row_normalise(np.stack(parts, axis=0))
    raise ValueError(f"Unknown partition scheme {partitions!r}")


def _row_normalise(a: np.ndarray) -> np.ndarray:
    """Divide each row of each partition by its degree, leaving zero rows zero."""
    deg = a.sum(axis=-1, keepdims=True)
    return np.divide(a, deg, out=np.zeros_like(a), where=deg > 0)


def bone_lengths(scale: np.ndarray | None = None) -> np.ndarray:
    """Bone lengths, optionally scaled per joint by ``scale``.

    Args:
        scale: Multiplicative per-joint factors, shape ``(NUM_JOINTS,)``.

    Returns:
        Array of shape ``(NUM_JOINTS,)``.
    """
    lengths = np.asarray(BONE_LENGTHS, dtype=np.float64).copy()
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float64)
        if scale.shape != (NUM_JOINTS,):
            raise ValueError(f"scale must have shape ({NUM_JOINTS},), got {scale.shape}")
        lengths *= scale
    return lengths
