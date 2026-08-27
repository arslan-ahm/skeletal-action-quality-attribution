"""Forward kinematics against closed forms, and the canonicalisation contract."""

from __future__ import annotations

import numpy as np
import pytest

from saqa.data.actions import assemble_angles, dof_trajectories
from saqa.data.kinematics import (
    REST_DIRECTIONS,
    canonicalise,
    euler_to_matrix,
    forward_kinematics,
    joint_angle,
    subject_limb_scale,
)
from saqa.data.skeleton import BONE_LENGTHS, LEFT_RIGHT_PAIRS, NUM_JOINTS, PARENTS


def test_euler_zero_is_identity():
    np.testing.assert_allclose(euler_to_matrix(np.zeros(3)), np.eye(3), atol=1e-12)


def test_euler_matrices_are_orthonormal():
    rng = np.random.default_rng(0)
    r = euler_to_matrix(rng.uniform(-np.pi, np.pi, size=(20, 3)))
    for m in r:
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(m) == pytest.approx(1.0)


def test_euler_x_rotation_matches_closed_form():
    """Rx(theta) maps (0,1,0) to (0, cos, sin)."""
    theta = 0.7
    r = euler_to_matrix(np.array([theta, 0.0, 0.0]))
    np.testing.assert_allclose(r @ np.array([0.0, 1.0, 0.0]),
                               [0.0, np.cos(theta), np.sin(theta)], atol=1e-12)


def test_euler_composition_order_is_xyz():
    a = np.array([0.3, -0.2, 0.5])
    expected = (
        euler_to_matrix(np.array([a[0], 0, 0]))
        @ euler_to_matrix(np.array([0, a[1], 0]))
        @ euler_to_matrix(np.array([0, 0, a[2]]))
    )
    np.testing.assert_allclose(euler_to_matrix(a), expected, atol=1e-12)


def test_euler_rejects_bad_shape():
    with pytest.raises(ValueError, match="3-axis"):
        euler_to_matrix(np.zeros((4, 2)))


def test_rest_directions_are_unit_or_zero():
    norms = np.linalg.norm(REST_DIRECTIONS, axis=-1)
    assert norms[0] == 0.0
    np.testing.assert_allclose(norms[1:], 1.0, atol=1e-12)


def test_bone_lengths_are_preserved_by_fk():
    """Rotations cannot stretch a bone. This is the single strongest invariant
    the kinematic chain has, so it is checked on a randomly-posed skeleton."""
    rng = np.random.default_rng(3)
    angles = rng.uniform(-1.0, 1.0, size=(5, NUM_JOINTS, 3))
    pos = forward_kinematics(angles, ground=False)
    for j in range(1, NUM_JOINTS):
        d = np.linalg.norm(pos[:, j] - pos[:, PARENTS[j]], axis=-1)
        np.testing.assert_allclose(d, BONE_LENGTHS[j], atol=1e-10)


def test_zero_angles_give_the_rest_pose():
    pos = forward_kinematics(np.zeros((1, NUM_JOINTS, 3)), ground=False)
    np.testing.assert_allclose(pos[0, 0], np.zeros(3), atol=1e-12)
    # head sits directly above the pelvis by the summed trunk bones
    expected = sum(BONE_LENGTHS[j] for j in (1, 2, 3, 4))
    assert pos[0, 4, 1] == pytest.approx(expected)


def test_grounding_puts_the_lower_ankle_on_the_floor():
    rng = np.random.default_rng(5)
    angles = rng.uniform(-0.4, 0.4, size=(6, NUM_JOINTS, 3))
    pos = forward_kinematics(angles, ground=True)
    floor = np.minimum(pos[:, 13, 1], pos[:, 16, 1])
    np.testing.assert_allclose(floor, 0.0, atol=1e-12)


def test_grounding_only_translates_vertically():
    angles = np.random.default_rng(6).uniform(-0.3, 0.3, size=(4, NUM_JOINTS, 3))
    a = forward_kinematics(angles, ground=False)
    b = forward_kinematics(angles, ground=True)
    delta = b - a
    np.testing.assert_allclose(delta[:, :, [0, 2]], 0.0, atol=1e-12)
    # the same shift for every joint in a frame
    np.testing.assert_allclose(
        delta[:, :, 1], np.broadcast_to(delta[:, :1, 1], delta[:, :, 1].shape), atol=1e-12
    )


def test_root_translation_is_applied():
    t = np.tile(np.array([1.0, 0.0, -2.0]), (3, 1))
    a = forward_kinematics(np.zeros((3, NUM_JOINTS, 3)), ground=False)
    b = forward_kinematics(np.zeros((3, NUM_JOINTS, 3)), root_translation=t, ground=False)
    np.testing.assert_allclose(b - a, np.broadcast_to(t[:, None, :], (b - a).shape),
                               atol=1e-12)


def test_fk_rejects_wrong_angle_shape():
    with pytest.raises(ValueError, match="angles must be"):
        forward_kinematics(np.zeros((4, 3)))


def test_fk_rejects_wrong_translation_shape():
    with pytest.raises(ValueError, match="root_translation"):
        forward_kinematics(np.zeros((4, NUM_JOINTS, 3)), root_translation=np.zeros((3, 3)))


def test_limb_scale_scales_bones():
    scale = np.full(NUM_JOINTS, 1.5)
    pos = forward_kinematics(np.zeros((1, NUM_JOINTS, 3)), limb_scale=scale, ground=False)
    assert pos[0, 4, 1] == pytest.approx(1.5 * sum(BONE_LENGTHS[j] for j in (1, 2, 3, 4)))


def test_joint_angle_of_a_straight_leg_is_pi():
    pos = forward_kinematics(np.zeros((1, NUM_JOINTS, 3)), ground=False)
    assert joint_angle(pos, 11, 12, 13)[0] == pytest.approx(np.pi, abs=1e-9)


def test_joint_angle_bends_with_flexion():
    """Squat knee angle must be far from straight at the bottom of the rep."""
    traj = dof_trajectories("squat", 32)
    pos = forward_kinematics(assemble_angles(traj, 32))
    knee = joint_angle(pos, 11, 12, 13)
    assert np.degrees(knee).min() < 110


def test_joint_angle_is_nan_for_degenerate_segments():
    pos = np.zeros((2, NUM_JOINTS, 3))
    assert np.isnan(joint_angle(pos, 11, 12, 13)).all()


def test_canonicalise_removes_translation():
    angles = np.zeros((8, NUM_JOINTS, 3))
    t = np.tile(np.array([3.0, 0.0, 5.0]), (8, 1))
    a = canonicalise(forward_kinematics(angles, ground=False))
    b = canonicalise(forward_kinematics(angles, root_translation=t, ground=False))
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_canonicalise_preserves_within_sequence_pelvis_motion():
    """Subtracting the *mean* pelvis, not the per-frame pelvis, is the whole
    point: squat depth must survive canonicalisation."""
    traj = dof_trajectories("squat", 32)
    pos = forward_kinematics(assemble_angles(traj, 32))
    canon = canonicalise(pos, scale_to_height=False)
    assert np.ptp(canon[:, 0, 1]) == pytest.approx(np.ptp(pos[:, 0, 1]), rel=1e-9)
    assert np.ptp(canon[:, 0, 1]) > 0.2


def test_canonicalise_scales_out_subject_size():
    traj = dof_trajectories("squat", 24)
    angles = assemble_angles(traj, 24)
    small = canonicalise(forward_kinematics(angles, limb_scale=np.full(NUM_JOINTS, 0.85)))
    big = canonicalise(forward_kinematics(angles, limb_scale=np.full(NUM_JOINTS, 1.20)))
    assert np.abs(small - big).max() < 0.02


def test_canonicalise_handles_batches():
    batch = np.random.default_rng(1).normal(size=(3, 10, NUM_JOINTS, 3))
    assert canonicalise(batch).shape == batch.shape


def test_canonicalise_rejects_bad_rank():
    with pytest.raises(ValueError, match="positions must be"):
        canonicalise(np.zeros((5, 3)))


def test_subject_limb_scale_is_left_right_symmetric():
    """Random per-bone noise would be indistinguishable from the asymmetry
    degradation, so subjects must be anatomically symmetric."""
    for seed in range(8):
        scale = subject_limb_scale(np.random.default_rng(seed))
        for left, right in LEFT_RIGHT_PAIRS:
            assert scale[left] == pytest.approx(scale[right])


def test_subject_limb_scale_is_positive_and_bounded():
    for seed in range(20):
        scale = subject_limb_scale(np.random.default_rng(seed))
        assert (scale > 0.5).all() and (scale < 1.6).all()


def test_subject_limb_scale_is_deterministic_per_seed():
    a = subject_limb_scale(np.random.default_rng(42))
    b = subject_limb_scale(np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
