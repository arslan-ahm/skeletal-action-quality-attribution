"""The procedural 3D skeleton motion generator.

One integer in, one fully-labelled sequence out. Every sample is a pure function
of its index and the generator config, so a dataset is reproducible from a seed
and a *single sample* can be regenerated in isolation for debugging or for a
figure.

Pipeline, in order (the order matters and is asserted by tests):

1. draw a subject (limb lengths, style) and an action class;
2. build clean DOF trajectories;
3. apply amplitude defects (range of motion, asymmetry, compensation, jerk);
4. apply the tempo warp -- *after* the amplitude edits, because a tempo defect
   re-times whatever the athlete actually did, it does not re-time an idealised
   trajectory;
5. forward kinematics, with postural sway added to the root;
6. the sensor model: per-frame jitter, i.i.d. noise, joint dropout.

Steps 1-5 are the *movement*; step 6 is the *measurement*. The quality label
and the attribution ground truth are functions of steps 1-5 only. Sensor noise
is a nuisance the model has to survive, not a defect it should be penalising --
conflating the two is the single easiest way to build a benchmark that rewards
the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actions import ACTION_CLASSES, ACTION_SPECS, assemble_angles, dof_trajectories, phase_of
from .degradations import (
    Degradation,
    apply_amplitude_degradation,
    combination_key,
    postural_sway,
    quality_from_degradations,
    resample_trajectories,
    sample_degradations,
    tempo_warp,
)
from .kinematics import canonicalise, forward_kinematics, subject_limb_scale
from .skeleton import NUM_JOINTS


@dataclass
class GeneratorConfig:
    """Everything that defines the data distribution.

    Attributes:
        num_frames: Sequence length ``T``.
        num_subjects: Size of the subject pool. Subjects are held out entirely
            in the cross-subject split, so this bounds how many distinct held-out
            identities exist.
        max_degradations: Upper bound on simultaneous defects per sequence.
        severity_min / severity_max: Bounds on drawn severity.
        noise_std: I.i.d. Gaussian position noise, in body heights (the
            canonical unit), applied per joint per frame.
        jitter_std: Temporally correlated (AR(1)) marker jitter, same units.
        dropout_prob: Probability that a joint is dropped for a short run of
            frames, imitating occlusion. Dropped joints are filled by linear
            interpolation, which is what every real pose pipeline does.
        dropout_max_frames: Longest dropout run.
        clean_fraction: Fraction of sequences forced to have no defects at all,
            so the top of the quality scale is populated.
        allowed_kinds: Restrict which defect kinds may appear.
    """

    num_frames: int = 64
    num_subjects: int = 24
    max_degradations: int = 3
    severity_min: float = 0.15
    severity_max: float = 1.0
    noise_std: float = 0.006
    jitter_std: float = 0.004
    dropout_prob: float = 0.05
    dropout_max_frames: int = 5
    clean_fraction: float = 0.04
    allowed_kinds: tuple[str, ...] | None = None
    actions: tuple[str, ...] = ACTION_CLASSES


@dataclass
class Sample:
    """One synthesised execution with its exact labels.

    Attributes:
        index: The integer the sample is a pure function of.
        action: Action class name.
        subject: Subject id.
        positions: ``(T, V, 3)`` canonicalised, sensor-corrupted coordinates --
            what a model sees.
        clean_positions: ``(T, V, 3)`` canonicalised coordinates before the
            sensor model. Used only for diagnostics and for the DTW reference.
        quality: The exact label in ``[0.05, 1]``.
        degradations: The applied defects.
        joint_truth: ``(V,)`` non-negative ground-truth attribution mass per
            joint, summing to 1 when any defect is present and all-zero for a
            clean sequence.
        frame_truth: ``(T,)`` ground-truth attribution mass per frame.
        combination: Canonical name of the set of defect kinds.
    """

    index: int
    action: str
    subject: int
    positions: np.ndarray
    clean_positions: np.ndarray
    quality: float
    degradations: list[Degradation]
    joint_truth: np.ndarray
    frame_truth: np.ndarray
    combination: str
    meta: dict[str, float] = field(default_factory=dict)


def _subject_of(index: int, cfg: GeneratorConfig) -> int:
    return index % cfg.num_subjects


def _build_positions(
    action: str,
    cfg: GeneratorConfig,
    style: float,
    limb_scale: np.ndarray,
    degradations: list[Degradation],
    seed: int,
) -> np.ndarray:
    """Run steps 2-5 of the pipeline for a given subset of defects.

    Called once for the full defect set and once per leave-one-out subset, which
    is what makes the attribution ground truth exact. The RNG is re-seeded from
    ``seed`` on every call so that the *stochastic* parts of a defect (jerk
    phase, sway waveform, tempo direction) are identical across the
    leave-one-out calls -- otherwise the counterfactual would differ by noise as
    well as by the removed defect, and the ground truth would be garbage.
    """
    t = cfg.num_frames
    traj = dof_trajectories(action, t, style=style)
    phase = phase_of(action, t)

    for i, deg in enumerate(degradations):
        rng = np.random.default_rng(seed * 9973 + i * 101 + 7)
        if deg.kind in ("rom", "asymmetry", "compensation", "jerk"):
            traj = apply_amplitude_degradation(traj, deg, phase, rng)

    for i, deg in enumerate(degradations):
        if deg.kind == "tempo":
            rng = np.random.default_rng(seed * 9973 + i * 101 + 7)
            traj = resample_trajectories(traj, tempo_warp(t, deg.severity, rng))

    angles = assemble_angles(traj, t)
    root = np.zeros((t, 3), dtype=np.float64)
    if action == "gait":
        # Walking translates forward. Without this, gait is a treadmill and the
        # tempo defect loses its main observable consequence.
        root[:, 2] = np.linspace(0.0, 1.2 * ACTION_SPECS["gait"].repetitions, t)
    for i, deg in enumerate(degradations):
        if deg.kind == "instability":
            rng = np.random.default_rng(seed * 9973 + i * 101 + 7)
            root = root + postural_sway(t, deg.severity, rng)
    return forward_kinematics(angles, root_translation=root, limb_scale=limb_scale)


def attribution_ground_truth(
    action: str,
    cfg: GeneratorConfig,
    style: float,
    limb_scale: np.ndarray,
    degradations: list[Degradation],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-joint and per-frame attribution ground truth.

    For each defect ``d``, forward kinematics is run with the full defect set and
    again with ``d`` removed. The per-joint displacement between the two
    canonicalised sequences is *what that defect did to the movement*, and it is
    weighted by ``d``'s contribution to the quality loss, because a defect that
    costs 0.05 quality should not dominate the explanation of a defect that
    costs 0.4.

    Leave-one-out (rather than leave-one-in) is used deliberately: it measures
    each defect's marginal effect in the presence of the others, which is the
    quantity an attribution method applied to the full sequence could possibly
    recover. Leave-one-in would measure an effect in a context the model never
    saw.

    Args:
        action: Action class.
        cfg: Generator config.
        style: Subject style factor.
        limb_scale: Subject limb scaling.
        degradations: Applied defects.
        seed: Sample seed, so counterfactuals share the defects' randomness.

    Returns:
        ``(joint_truth, frame_truth)``, each non-negative and summing to 1 (or
        all-zero when there are no defects, because a clean sequence has no
        cause to attribute -- returning a uniform vector there would silently
        reward a model that always says "everything").
    """
    v, t = NUM_JOINTS, cfg.num_frames
    joint_truth = np.zeros(v, dtype=np.float64)
    frame_truth = np.zeros(t, dtype=np.float64)
    if not degradations:
        return joint_truth, frame_truth

    full = canonicalise(_build_positions(action, cfg, style, limb_scale, degradations, seed))
    for i in range(len(degradations)):
        subset = [d for j, d in enumerate(degradations) if j != i]
        without = canonicalise(_build_positions(action, cfg, style, limb_scale, subset, seed))
        disp = np.linalg.norm(full - without, axis=-1)  # (T, V)
        weight = degradations[i].cost
        joint_truth += weight * disp.mean(axis=0)
        frame_truth += weight * disp.mean(axis=1)

    for arr in (joint_truth, frame_truth):
        total = arr.sum()
        if total > 0:
            arr /= total
    return joint_truth, frame_truth


def _sensor_model(
    positions: np.ndarray, cfg: GeneratorConfig, rng: np.random.Generator
) -> np.ndarray:
    """Jitter, noise and occlusion dropout, applied to canonicalised coords."""
    t, v, _ = positions.shape
    out = positions.copy()

    if cfg.jitter_std > 0:
        # AR(1) with rho = 0.8: marker jitter is correlated in time, which is
        # what makes it hard to remove and easy to mistake for a jerk defect.
        eps = rng.normal(0.0, cfg.jitter_std, size=(t, v, 3))
        jitter = np.empty_like(eps)
        jitter[0] = eps[0]
        for i in range(1, t):
            jitter[i] = 0.8 * jitter[i - 1] + eps[i]
        out += jitter
    if cfg.noise_std > 0:
        out += rng.normal(0.0, cfg.noise_std, size=out.shape)

    if cfg.dropout_prob > 0:
        for j in range(v):
            if rng.random() >= cfg.dropout_prob:
                continue
            run = int(rng.integers(2, cfg.dropout_max_frames + 1))
            start = int(rng.integers(0, max(1, t - run)))
            idx = np.arange(start, min(t, start + run))
            keep = np.setdiff1d(np.arange(t), idx)
            if keep.size >= 2:
                for axis in range(3):
                    out[idx, j, axis] = np.interp(idx, keep, out[keep, j, axis])
    return out


def generate_sample(index: int, cfg: GeneratorConfig | None = None) -> Sample:
    """Synthesise one fully-labelled execution.

    Args:
        index: Sample index. The sample is a pure function of this and ``cfg``.
        cfg: Generator config; defaults to :class:`GeneratorConfig`.

    Returns:
        A :class:`Sample`.
    """
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(1_000_003 + index)
    subject = _subject_of(index, cfg)
    subj_rng = np.random.default_rng(555_555 + subject)
    limb_scale = subject_limb_scale(subj_rng)
    style = float(subj_rng.normal(0.0, 0.6))

    action = str(cfg.actions[int(rng.integers(0, len(cfg.actions)))])
    if rng.random() < cfg.clean_fraction:
        degradations: list[Degradation] = []
    else:
        degradations = sample_degradations(
            action,
            rng,
            max_count=cfg.max_degradations,
            severity_range=(cfg.severity_min, cfg.severity_max),
            allowed_kinds=cfg.allowed_kinds,
        )

    raw = _build_positions(action, cfg, style, limb_scale, degradations, index)
    clean = canonicalise(raw)
    joint_truth, frame_truth = attribution_ground_truth(
        action, cfg, style, limb_scale, degradations, index
    )
    observed = _sensor_model(clean, cfg, rng)

    return Sample(
        index=index,
        action=action,
        subject=subject,
        positions=observed.astype(np.float32),
        clean_positions=clean.astype(np.float32),
        quality=quality_from_degradations(degradations),
        degradations=degradations,
        joint_truth=joint_truth,
        frame_truth=frame_truth,
        combination=combination_key(degradations),
        meta={"style": style, "num_degradations": float(len(degradations))},
    )


def reference_sequence(action: str, cfg: GeneratorConfig | None = None) -> np.ndarray:
    """The canonical *defect-free, style-neutral* execution of an action.

    This is the "reference performance" the DTW baseline needs and the
    reference-free model does not. Providing it is what makes the baseline
    comparison fair: the baseline is given exactly the extra information its
    formulation assumes, generated from the same code path.

    Returns:
        ``(T, V, 3)`` canonicalised coordinates, noise-free.
    """
    cfg = cfg or GeneratorConfig()
    limb_scale = np.ones(NUM_JOINTS, dtype=np.float64)
    raw = _build_positions(action, cfg, 0.0, limb_scale, [], 0)
    return canonicalise(raw).astype(np.float32)
