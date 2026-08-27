"""Optional loaders for real skeleton and action-quality datasets.

None of the committed results use these. They exist so the real path is
first-class for a reader who has the archives, and so the expected format is
documented in code rather than in prose. ``scripts/download_real.py`` prints the
retrieval steps.

The important part of this module is :data:`NTU_TO_SAQA`, the joint mapping.
NTU RGB+D has 25 joints and this project's tree has 17; the mapping is explicit
and lossy, and the joints it drops (hands, thumbs, feet tips) are named so a
reader can see exactly what information a real run would not have.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .kinematics import canonicalise
from .skeleton import JOINT_NAMES, NUM_JOINTS

#: NTU RGB+D joint index for each of this project's joints, in this project's
#: order. NTU indices are 0-based (the raw files are 1-based; the reader below
#: subtracts one). Dropped NTU joints: 7/11 (hands), 21-24 (hand tips, thumbs,
#: foot tips) -- this skeleton has no wrist-distal segments, so nothing that uses
#: them would be defined.
NTU_TO_SAQA: tuple[int, ...] = (
    0,   # pelvis      <- NTU base of spine
    1,   # spine       <- NTU middle of spine
    20,  # thorax      <- NTU spine shoulder
    2,   # neck        <- NTU neck
    3,   # head        <- NTU head
    4,   # l_shoulder
    5,   # l_elbow
    6,   # l_wrist
    8,   # r_shoulder
    9,   # r_elbow
    10,  # r_wrist
    12,  # l_hip
    13,  # l_knee
    14,  # l_ankle
    16,  # r_hip
    17,  # r_knee
    18,  # r_ankle
)


def ntu_to_saqa_topology(skeleton: np.ndarray) -> np.ndarray:
    """Map an NTU ``(T, 25, 3)`` skeleton onto this project's 17 joints.

    Args:
        skeleton: ``(T, 25, 3)`` or ``(N, T, 25, 3)`` coordinates in metres.

    Returns:
        The same leading shape with 17 joints.

    Raises:
        ValueError: if the joint axis is not 25.
    """
    a = np.asarray(skeleton, dtype=np.float64)
    if a.shape[-2] != 25:
        raise ValueError(f"expected 25 NTU joints, got {a.shape[-2]}")
    return a[..., list(NTU_TO_SAQA), :]


def read_ntu_skeleton(path: str | Path) -> np.ndarray:
    """Parse one NTU ``.skeleton`` file into ``(T, 25, 3)`` for the first body.

    Frames with no detected body are filled by linear interpolation from the
    neighbouring frames, and a file with *no* bodies at all raises rather than
    returning zeros -- an all-zero skeleton would silently train as a valid
    still-standing subject.

    Raises:
        ValueError: if the file contains no body in any frame.
    """
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    pos = 0

    def take() -> str:
        nonlocal pos
        value = lines[pos].strip()
        pos += 1
        return value

    num_frames = int(take())
    out = np.full((num_frames, 25, 3), np.nan)
    for f in range(num_frames):
        bodies = int(take())
        for b in range(bodies):
            take()  # body info line
            joints = int(take())
            for j in range(joints):
                parts = take().split()
                if b == 0 and j < 25:
                    out[f, j] = [float(parts[0]), float(parts[1]), float(parts[2])]
    valid = np.isfinite(out[:, 0, 0])
    if not valid.any():
        raise ValueError(f"{path} contains no detected body in any frame")
    idx = np.arange(num_frames)
    for j in range(25):
        for c in range(3):
            out[:, j, c] = np.interp(idx, idx[valid], out[valid, j, c])
    return out


def load_aqa_directory(
    root: str | Path, num_frames: int = 64, score_column: str = "score"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load an MTL-AQA / FineDiving style directory.

    Expects ``poses.npy`` of shape ``(N, T, V, 3)`` and ``scores.csv`` with a
    ``clip_id`` column and a score column. Sequences are resampled to
    ``num_frames`` and canonicalised the same way the synthetic data is, so the
    same trained weights apply.

    Args:
        root: Directory.
        num_frames: Target length.
        score_column: Column holding the judge score.

    Returns:
        ``(coords, scores, clip_ids)`` with ``coords`` of shape ``(N, 3, T, V)``
        and ``scores`` min-max scaled onto ``[0.05, 1]`` -- the same range the
        ordinal head is built for. The scaling constants are returned in the
        clip-id list's place only if a caller needs them; they are recoverable
        from the raw CSV.

    Raises:
        FileNotFoundError: if either file is missing.
        ValueError: on a joint count this topology cannot accept.
    """
    import pandas as pd

    root = Path(root)
    poses_path, scores_path = root / "poses.npy", root / "scores.csv"
    if not poses_path.exists() or not scores_path.exists():
        raise FileNotFoundError(
            f"expected {poses_path} and {scores_path}; "
            "run scripts/download_real.py for retrieval steps"
        )
    poses = np.load(poses_path)
    if poses.shape[-2] == 25:
        poses = ntu_to_saqa_topology(poses)
    if poses.shape[-2] != NUM_JOINTS:
        raise ValueError(
            f"expected {NUM_JOINTS} or 25 joints, got {poses.shape[-2]}; "
            f"this topology is {JOINT_NAMES}"
        )
    frame = pd.read_csv(scores_path)
    scores = frame[score_column].to_numpy(dtype=np.float64)

    resampled = np.empty((poses.shape[0], num_frames) + poses.shape[2:])
    src = np.arange(poses.shape[1])
    dst = np.linspace(0, poses.shape[1] - 1, num_frames)
    for i in range(poses.shape[0]):
        for v in range(poses.shape[2]):
            for c in range(3):
                resampled[i, :, v, c] = np.interp(dst, src, poses[i, :, v, c])

    coords = canonicalise(resampled).transpose(0, 3, 1, 2).astype(np.float32)
    lo, hi = scores.min(), scores.max()
    scaled = 0.05 + 0.95 * (scores - lo) / max(hi - lo, 1e-9)
    return coords, scaled.astype(np.float32), list(frame["clip_id"].astype(str))
