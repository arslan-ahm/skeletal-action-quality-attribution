"""Actions, degradations and the generator: the label and truth contracts."""

from __future__ import annotations

import numpy as np
import pytest

from saqa.data.actions import (
    ACTION_CLASSES,
    ACTION_SPECS,
    DOF,
    assemble_angles,
    dof_trajectories,
    phase_of,
)
from saqa.data.degradations import (
    DEGRADATION_KINDS,
    KIND_WEIGHTS,
    MAX_LOSS,
    Degradation,
    apply_amplitude_degradation,
    combination_key,
    postural_sway,
    quality_from_degradations,
    resample_trajectories,
    sample_degradations,
    tempo_warp,
)
from saqa.data.generator import (
    attribution_ground_truth,
    generate_sample,
    reference_sequence,
)
from saqa.data.kinematics import forward_kinematics, subject_limb_scale
from saqa.data.skeleton import NUM_JOINTS

# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("action", ACTION_CLASSES)
def test_trajectories_have_the_right_length(action):
    for name, traj in dof_trajectories(action, 32).items():
        assert name in DOF
        assert traj.shape == (32,)


@pytest.mark.parametrize("action", ACTION_CLASSES)
def test_trajectories_are_finite(action):
    for traj in dof_trajectories(action, 32).values():
        assert np.isfinite(traj).all()


@pytest.mark.parametrize("action", ACTION_CLASSES)
def test_primary_dofs_actually_move(action):
    """A degradation is drawn from ``primary_dofs``. If one of them were static
    the defect would be an invisible no-op with a real quality cost, which would
    poison both the label and the attribution ground truth."""
    traj = dof_trajectories(action, 48)
    for dof in ACTION_SPECS[action].primary_dofs:
        assert dof in traj, f"{action}: primary DOF {dof} has no trajectory"
        assert np.ptp(traj[dof]) > 1e-3, f"{action}: primary DOF {dof} does not move"


@pytest.mark.parametrize("action", ACTION_CLASSES)
def test_actions_are_kinematically_distinct(action):
    """Each class must produce a materially different motion; otherwise the
    per-class breakdown measures nothing."""
    others = [a for a in ACTION_CLASSES if a != action]
    mine = forward_kinematics(assemble_angles(dof_trajectories(action, 32), 32))
    for other in others:
        theirs = forward_kinematics(assemble_angles(dof_trajectories(other, 32), 32))
        assert np.abs(mine - theirs).mean() > 1e-3


def test_style_changes_amplitude_but_not_class():
    a = dof_trajectories("squat", 32, style=0.0)
    b = dof_trajectories("squat", 32, style=1.0)
    assert np.ptp(b["l_thigh_flex"]) > np.ptp(a["l_thigh_flex"])


def test_unknown_action_rejected():
    with pytest.raises(ValueError, match="Unknown action"):
        dof_trajectories("backflip", 16)


def test_assemble_angles_scatters_correctly():
    traj = {"l_thigh_flex": np.arange(8, dtype=float)}
    angles = assemble_angles(traj, 8)
    j, axis = DOF["l_thigh_flex"]
    np.testing.assert_array_equal(angles[:, j, axis], np.arange(8))
    assert angles.sum() == np.arange(8).sum()


def test_assemble_angles_rejects_unknown_dof():
    with pytest.raises(KeyError, match="Unknown DOF"):
        assemble_angles({"tail_wag": np.zeros(4)}, 4)


def test_assemble_angles_rejects_wrong_length():
    with pytest.raises(ValueError, match="must be"):
        assemble_angles({"spine_flex": np.zeros(3)}, 4)


@pytest.mark.parametrize("action", ACTION_CLASSES)
def test_phase_is_in_range(action):
    p = phase_of(action, 40)
    assert p.shape == (40,)
    assert ((p >= 0) & (p < 1)).all()


def test_gait_is_left_right_antiphase():
    """The only class whose quality signal lives in inter-limb timing."""
    traj = dof_trajectories("gait", 64)
    corr = np.corrcoef(traj["l_thigh_flex"], traj["r_thigh_flex"])[0, 1]
    assert corr < -0.9


# --------------------------------------------------------------------------
# degradations
# --------------------------------------------------------------------------


def test_quality_of_no_defects_is_one():
    assert quality_from_degradations([]) == 1.0


def test_quality_is_linear_in_severity():
    d = Degradation("rom", 0.5, ("l_thigh_flex",))
    assert quality_from_degradations([d]) == pytest.approx(1.0 - 0.5 * KIND_WEIGHTS["rom"])


@pytest.mark.parametrize("kind", DEGRADATION_KINDS)
def test_quality_is_monotone_non_increasing_in_severity(kind):
    prev = 2.0
    for sev in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = quality_from_degradations([Degradation(kind, sev, ("spine_flex",))])
        assert q <= prev + 1e-12
        prev = q


def test_quality_is_clipped_at_the_floor():
    many = [Degradation(k, 1.0, ("spine_flex",)) for k in DEGRADATION_KINDS]
    assert quality_from_degradations(many) == pytest.approx(1.0 - MAX_LOSS)


def test_quality_never_leaves_its_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        degs = [Degradation(str(rng.choice(DEGRADATION_KINDS)), float(rng.random()),
                            ("spine_flex",))
                for _ in range(int(rng.integers(0, 4)))]
        q = quality_from_degradations(degs)
        assert 1.0 - MAX_LOSS - 1e-12 <= q <= 1.0 + 1e-12


def test_degradation_cost_matches_weight():
    d = Degradation("compensation", 0.4, ("l_thigh_flex", "spine_flex"))
    assert d.cost == pytest.approx(0.4 * KIND_WEIGHTS["compensation"])


def test_declared_joints_come_from_the_edited_dofs():
    d = Degradation("rom", 0.5, ("l_thigh_flex", "r_thigh_flex"))
    assert d.joints == (DOF["l_thigh_flex"][0], DOF["r_thigh_flex"][0])


def test_degradation_to_dict_is_flat_and_serialisable():
    row = Degradation("jerk", 0.3, ("spine_flex",), (0.1, 0.6)).to_dict()
    assert row["kind"] == "jerk" and row["phase_start"] == 0.1
    assert all(not isinstance(v, dict | list) for v in row.values())


def test_rom_reduces_range_but_keeps_the_mean():
    traj = dof_trajectories("squat", 48)
    phase = phase_of("squat", 48)
    before = traj["l_thigh_flex"].copy()
    after = apply_amplitude_degradation(
        dict(traj), Degradation("rom", 1.0, ("l_thigh_flex",)), phase,
        np.random.default_rng(0),
    )["l_thigh_flex"]
    assert np.ptp(after) < np.ptp(before)
    assert after.mean() == pytest.approx(before.mean(), abs=1e-9)


def test_rom_severity_zero_is_a_no_op():
    traj = dof_trajectories("squat", 48)
    phase = phase_of("squat", 48)
    after = apply_amplitude_degradation(
        dict(traj), Degradation("rom", 0.0, ("l_thigh_flex",)), phase,
        np.random.default_rng(0),
    )
    np.testing.assert_allclose(after["l_thigh_flex"], traj["l_thigh_flex"], atol=1e-12)


def test_asymmetry_touches_only_one_side():
    traj = dof_trajectories("squat", 48)
    phase = phase_of("squat", 48)
    after = apply_amplitude_degradation(
        dict(traj), Degradation("asymmetry", 0.8, ("l_thigh_flex",)), phase,
        np.random.default_rng(0),
    )
    assert np.ptp(after["l_thigh_flex"]) < np.ptp(traj["l_thigh_flex"])
    np.testing.assert_allclose(after["r_thigh_flex"], traj["r_thigh_flex"], atol=1e-12)


def test_compensation_reduces_one_dof_and_grows_another():
    traj = dof_trajectories("squat", 48)
    phase = phase_of("squat", 48)
    deg = Degradation("compensation", 1.0, ("l_thigh_flex", "spine_lat"), (0.0, 1.0),
                      {"compensation_gain": 0.55})
    after = apply_amplitude_degradation(dict(traj), deg, phase, np.random.default_rng(0))
    assert np.ptp(after["l_thigh_flex"]) < np.ptp(traj["l_thigh_flex"])
    assert np.ptp(after["spine_lat"]) > np.ptp(traj.get("spine_lat", np.zeros(48)))


def test_jerk_raises_acceleration_not_range():
    traj = dof_trajectories("squat", 48)
    phase = phase_of("squat", 48)
    after = apply_amplitude_degradation(
        dict(traj), Degradation("jerk", 1.0, ("l_thigh_flex",)), phase,
        np.random.default_rng(1),
    )["l_thigh_flex"]
    before = traj["l_thigh_flex"]
    assert np.abs(np.diff(after, 2)).mean() > 2 * np.abs(np.diff(before, 2)).mean()
    assert np.ptp(after) == pytest.approx(np.ptp(before), rel=0.25)


def test_phase_window_confines_the_edit():
    traj = dof_trajectories("squat", 64)
    phase = phase_of("squat", 64)
    deg = Degradation("rom", 1.0, ("l_thigh_flex",), (0.6, 0.9))
    after = apply_amplitude_degradation(dict(traj), deg, phase, np.random.default_rng(0))
    untouched = phase < 0.55
    np.testing.assert_allclose(
        after["l_thigh_flex"][untouched], traj["l_thigh_flex"][untouched], atol=1e-9
    )


def test_tempo_warp_is_monotone():
    for sev in (0.0, 0.4, 1.0):
        idx = tempo_warp(64, sev, np.random.default_rng(2))
        assert (np.diff(idx) >= -1e-9).all()
        assert idx[0] == pytest.approx(0.0)
        assert idx[-1] == pytest.approx(63.0)


def test_tempo_warp_zero_severity_is_the_identity():
    np.testing.assert_allclose(tempo_warp(32, 0.0, np.random.default_rng(0)),
                               np.arange(32), atol=1e-9)


def test_resample_preserves_endpoints():
    traj = {"a": np.linspace(0.0, 1.0, 32)}
    out = resample_trajectories(traj, tempo_warp(32, 0.5, np.random.default_rng(0)))
    assert out["a"][0] == pytest.approx(0.0)
    assert out["a"][-1] == pytest.approx(1.0)


def test_postural_sway_scales_with_severity():
    a = np.abs(postural_sway(48, 0.2, np.random.default_rng(3))).mean()
    b = np.abs(postural_sway(48, 1.0, np.random.default_rng(3))).mean()
    assert b > 3 * a


def test_postural_sway_is_zero_at_zero_severity():
    np.testing.assert_allclose(postural_sway(24, 0.0, np.random.default_rng(0)), 0.0)


@pytest.mark.parametrize("action", ACTION_CLASSES)
def test_sampled_degradations_are_well_formed(action):
    rng = np.random.default_rng(7)
    for _ in range(20):
        degs = sample_degradations(action, rng)
        assert len(degs) <= 3
        kinds = [d.kind for d in degs]
        assert len(kinds) == len(set(kinds)), "at most one defect per kind"
        for d in degs:
            assert 0.0 < d.severity <= 1.0
            assert d.kind in DEGRADATION_KINDS
            assert 0.0 <= d.phase_window[0] < d.phase_window[1] <= 1.0


def test_rom_is_bilateral_and_asymmetry_is_not():
    """If both were one-sided they would be mechanically identical while
    carrying different quality costs -- an unlearnable label."""
    rng = np.random.default_rng(11)
    seen_rom = seen_asym = 0
    for _ in range(200):
        for d in sample_degradations("squat", rng, allowed_kinds=("rom", "asymmetry")):
            if d.kind == "rom" and any(x in ACTION_SPECS["squat"].mirror for x in d.dofs):
                assert len(d.dofs) == 2
                seen_rom += 1
            if d.kind == "asymmetry":
                assert len(d.dofs) == 1
                seen_asym += 1
    assert seen_rom > 0 and seen_asym > 0


def test_allowed_kinds_is_respected():
    rng = np.random.default_rng(4)
    for _ in range(30):
        for d in sample_degradations("gait", rng, allowed_kinds=("jerk",)):
            assert d.kind == "jerk"


def test_combination_key_is_order_invariant():
    a = [Degradation("jerk", 0.5, ()), Degradation("rom", 0.5, ("spine_flex",))]
    assert combination_key(a) == combination_key(list(reversed(a))) == "jerk+rom"


def test_combination_key_of_clean():
    assert combination_key([]) == "clean"


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------


def test_sample_is_a_pure_function_of_its_index(tiny_cfg):
    a, b = generate_sample(5, tiny_cfg), generate_sample(5, tiny_cfg)
    np.testing.assert_array_equal(a.positions, b.positions)
    assert a.quality == b.quality
    assert a.combination == b.combination


def test_different_indices_give_different_samples(tiny_cfg):
    a, b = generate_sample(5, tiny_cfg), generate_sample(6, tiny_cfg)
    assert not np.array_equal(a.positions, b.positions)


def test_sample_shapes(tiny_cfg):
    s = generate_sample(0, tiny_cfg)
    assert s.positions.shape == (tiny_cfg.num_frames, NUM_JOINTS, 3)
    assert s.joint_truth.shape == (NUM_JOINTS,)
    assert s.frame_truth.shape == (tiny_cfg.num_frames,)


def test_sample_label_matches_its_degradations(tiny_cfg):
    for i in range(30):
        s = generate_sample(i, tiny_cfg)
        assert s.quality == pytest.approx(quality_from_degradations(s.degradations))


def test_subject_id_is_within_the_pool(tiny_cfg):
    for i in range(30):
        assert 0 <= generate_sample(i, tiny_cfg).subject < tiny_cfg.num_subjects


def test_ground_truth_sums_to_one_when_defects_exist(tiny_cfg):
    seen = 0
    for i in range(40):
        s = generate_sample(i, tiny_cfg)
        if s.degradations:
            seen += 1
            assert s.joint_truth.sum() == pytest.approx(1.0)
            assert s.frame_truth.sum() == pytest.approx(1.0)
    assert seen > 0


def test_clean_sequences_have_zero_ground_truth(tiny_cfg):
    """A clean sequence has no cause to attribute. A uniform vector would
    silently reward a model that always says 'everything'."""
    seen = 0
    for i in range(80):
        s = generate_sample(i, tiny_cfg)
        if not s.degradations:
            seen += 1
            assert s.joint_truth.sum() == 0.0
            assert s.frame_truth.sum() == 0.0
            assert s.quality == 1.0
    assert seen > 0


def test_ground_truth_is_non_negative(tiny_cfg):
    for i in range(20):
        s = generate_sample(i, tiny_cfg)
        assert (s.joint_truth >= 0).all() and (s.frame_truth >= 0).all()


def test_ground_truth_is_concentrated_not_uniform(clean_cfg):
    """If the truth were near-uniform, attribution precision would be
    uninformative. The top three joints must carry well above the 3/17 a uniform
    vector would give."""
    masses = []
    for i in range(60):
        s = generate_sample(i, clean_cfg)
        if s.degradations:
            masses.append(np.sort(s.joint_truth)[-3:].sum())
    assert np.mean(masses) > 0.35


def test_sensor_model_perturbs_but_does_not_relabel(tiny_cfg):
    from dataclasses import replace

    noisy = generate_sample(3, tiny_cfg)
    clean = generate_sample(3, replace(tiny_cfg, noise_std=0.0, jitter_std=0.0,
                                       dropout_prob=0.0))
    assert noisy.quality == clean.quality
    np.testing.assert_array_equal(noisy.joint_truth, clean.joint_truth)
    assert not np.array_equal(noisy.positions, noisy.clean_positions)


def test_attribution_truth_is_zero_without_defects(clean_cfg):
    j, f = attribution_ground_truth("squat", clean_cfg, 0.0,
                                    subject_limb_scale(np.random.default_rng(0)), [], 0)
    assert j.sum() == 0.0 and f.sum() == 0.0


def test_attribution_truth_localises_a_leg_defect(clean_cfg):
    """A severe left-shank restriction must put more mass on the left leg than
    on the right arm. This is the property the whole fidelity experiment rests
    on, so it is asserted directly."""
    deg = [Degradation("asymmetry", 1.0, ("l_shank_flex",))]
    j, _ = attribution_ground_truth(
        "squat", clean_cfg, 0.0, subject_limb_scale(np.random.default_rng(0)), deg, 0
    )
    left_leg = j[[11, 12, 13]].sum()
    right_arm = j[[8, 9, 10]].sum()
    assert left_leg > right_arm


def test_reference_sequence_is_defect_free_and_stable(tiny_cfg):
    a = reference_sequence("squat", tiny_cfg)
    b = reference_sequence("squat", tiny_cfg)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (tiny_cfg.num_frames, NUM_JOINTS, 3)
    assert np.isfinite(a).all()


def test_all_actions_appear_across_a_reasonable_draw(tiny_cfg):
    seen = {generate_sample(i, tiny_cfg).action for i in range(120)}
    assert seen == set(ACTION_CLASSES)
