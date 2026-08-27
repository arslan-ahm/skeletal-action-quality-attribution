"""Attribution methods and the fidelity metrics that score them."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from saqa.attribution.methods import (
    completeness_error,
    edge_importance_attribution,
    input_gradient,
    integrated_gradients,
    occlusion,
    temporal_baseline,
    to_frame_vector,
    to_joint_vector,
)
from saqa.data.skeleton import NUM_JOINTS
from saqa.metrics.attribution import (
    adaptive_k,
    distribution_scores,
    fidelity_summary,
    per_sequence_fidelity,
    random_attribution,
    temporal_localisation_error,
    topk_scores,
)
from saqa.metrics.monotonicity import (
    ladder_violations,
    monotonicity_report,
    rank_inconsistency,
)
from saqa.models.registry import build_model


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = build_model("saqa_stgcn", channels=(8, 16), strides=(2, 1))
    m.eval()
    return m


@pytest.fixture(scope="module")
def x():
    return torch.randn(3, 3, 16, NUM_JOINTS) * 0.2


# --------------------------------------------------------------------------
# attribution methods
# --------------------------------------------------------------------------


def test_temporal_baseline_freezes_motion_but_keeps_pose(x):
    base = temporal_baseline(x)
    assert base.shape == x.shape
    assert base.std(dim=2).max().item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(base.mean(dim=2), x.mean(dim=2))


def test_integrated_gradients_shape(model, x):
    assert integrated_gradients(model, x, steps=8).shape == (3, 16, NUM_JOINTS)


def test_integrated_gradients_accepts_a_single_sequence(model):
    assert integrated_gradients(model, torch.randn(3, 16, NUM_JOINTS), steps=4).shape == (
        1, 16, NUM_JOINTS
    )


def test_integrated_gradients_satisfies_completeness(model, x):
    """IG of -f sums to f(baseline) - f(x). This is the property that makes the
    method checkable, so it is checked rather than cited."""
    err = completeness_error(model, x, steps=64)
    assert err.max() < 5e-3


def test_completeness_improves_with_more_steps(model, x):
    coarse = completeness_error(model, x, steps=4).mean()
    fine = completeness_error(model, x, steps=64).mean()
    assert fine <= coarse


def test_integrated_gradients_is_zero_at_the_baseline(model):
    base = temporal_baseline(torch.randn(1, 3, 16, NUM_JOINTS))
    attr = integrated_gradients(model, base, steps=8)
    assert np.abs(attr).max() < 1e-6


def test_integrated_gradients_rejects_unknown_baseline(model, x):
    with pytest.raises(ValueError, match="Unknown baseline"):
        integrated_gradients(model, x, baseline="median")


def test_integrated_gradients_rejects_bad_rank(model):
    with pytest.raises(ValueError, match=r"expected \(N, C, T, V\)"):
        integrated_gradients(model, torch.randn(16, NUM_JOINTS))


def test_occlusion_shape_and_finiteness(model, x):
    attr = occlusion(model, x, window=8, stride=4)
    assert attr.shape == (3, 16, NUM_JOINTS)
    assert np.isfinite(attr).all()


def test_occlusion_covers_every_tile(model, x):
    """A stride that does not divide the length must still reach the last
    frames, or the tail of every sequence is silently unexplained."""
    attr = occlusion(model, x, window=7, stride=5)
    assert (np.abs(attr).sum(axis=(0, 2)) > 0).all()


def test_occlusion_of_a_frozen_sequence_is_zero(model):
    """Occluding motion that is not there cannot change the score."""
    frozen = torch.randn(1, 3, 1, NUM_JOINTS).expand(1, 3, 16, NUM_JOINTS).contiguous()
    attr = occlusion(model, frozen, window=8, stride=8)
    assert np.abs(attr).max() < 1e-5


def test_input_gradient_shape(model, x):
    assert input_gradient(model, x).shape == (3, 16, NUM_JOINTS)


def test_joint_and_frame_reductions_are_non_negative(model, x):
    attr = integrated_gradients(model, x, steps=8)
    assert to_joint_vector(attr).shape == (3, NUM_JOINTS)
    assert to_frame_vector(attr).shape == (3, 16)
    assert (to_joint_vector(attr) >= 0).all()
    assert (to_frame_vector(attr) >= 0).all()


def test_reductions_accept_a_single_map():
    attr = np.random.default_rng(0).normal(size=(16, NUM_JOINTS))
    assert to_joint_vector(attr).shape == (1, NUM_JOINTS)


def test_positive_part_reduction_does_not_cancel():
    """A joint costing quality in one phase and helping in another has a real
    localised cost; a signed sum would erase it."""
    attr = np.zeros((1, 4, 2))
    attr[0, 0, 0] = 5.0
    attr[0, 1, 0] = -5.0
    assert to_joint_vector(attr)[0, 0] == pytest.approx(5.0)


def test_edge_importance_is_per_joint_and_non_negative(model):
    edge = edge_importance_attribution(model)
    assert edge.shape == (NUM_JOINTS,)
    assert (edge >= 0).all() and edge.sum() > 0


def test_edge_importance_is_zero_without_edge_weights():
    m = build_model("tcn")
    assert edge_importance_attribution(m).sum() == 0.0


# --------------------------------------------------------------------------
# fidelity metrics
# --------------------------------------------------------------------------


def test_adaptive_k_tracks_concentration():
    peaked = np.zeros(17)
    peaked[3] = 1.0
    assert adaptive_k(peaked) == 1
    assert adaptive_k(np.ones(17)) >= 13


def test_adaptive_k_of_an_empty_truth():
    assert adaptive_k(np.zeros(17)) == 1


def test_topk_of_a_perfect_attribution():
    truth = np.array([0.0, 0.6, 0.4, 0.0])
    out = topk_scores(truth, truth)
    assert out["precision"] == out["recall"] == out["f1"] == out["iou"] == 1.0
    assert out["mass_captured"] == pytest.approx(1.0)


def test_topk_of_a_disjoint_attribution():
    truth = np.array([0.0, 0.0, 0.5, 0.5])
    attr = np.array([1.0, 1.0, 0.0, 0.0])
    out = topk_scores(attr, truth)
    assert out["precision"] == 0.0 and out["iou"] == 0.0


def test_topk_is_nan_for_a_clean_sequence():
    """Precision against an empty cause is undefined, not 1.0."""
    out = topk_scores(np.random.default_rng(0).random(17), np.zeros(17))
    assert all(np.isnan(v) for v in out.values())


def test_topk_uses_the_absolute_value_of_the_attribution():
    truth = np.array([0.0, 1.0, 0.0])
    assert topk_scores(np.array([0.0, -5.0, 0.1]), truth)["precision"] == 1.0


def test_distribution_scores_of_an_identical_vector():
    truth = np.array([0.1, 0.5, 0.4])
    out = distribution_scores(truth, truth)
    assert out["overlap"] == pytest.approx(1.0)
    assert out["rank_corr"] == pytest.approx(1.0)
    assert out["top1_hit"] == 1.0


def test_overlap_of_disjoint_distributions_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert distribution_scores(a, b)["overlap"] == pytest.approx(0.0)


def test_temporal_localisation_error_of_a_matching_centre():
    truth = np.zeros(20)
    truth[10] = 1.0
    assert temporal_localisation_error(truth, truth) == pytest.approx(0.0)


def test_temporal_localisation_error_scales_with_the_offset():
    truth = np.zeros(21)
    truth[0] = 1.0
    attr = np.zeros(21)
    attr[20] = 1.0
    assert temporal_localisation_error(attr, truth) == pytest.approx(1.0)


def test_fidelity_summary_excludes_clean_sequences_from_the_mean():
    truth = np.zeros((3, 8))
    truth[0, 2] = 1.0
    truth[1, 5] = 1.0
    attr = np.zeros((3, 8))
    attr[:, 2] = 1.0
    out = fidelity_summary(attr, truth, "joint")
    assert out["n_precision"] == 2.0
    assert out["precision"] == pytest.approx(0.5)


def test_fidelity_summary_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must match"):
        fidelity_summary(np.zeros((2, 5)), np.zeros((2, 6)))


def test_fidelity_summary_adds_localisation_for_frames():
    truth = np.zeros((2, 10))
    truth[:, 3] = 1.0
    out = fidelity_summary(truth, truth, "frame")
    assert "localisation_error" in out


def test_per_sequence_fidelity_is_nan_for_clean_rows():
    truth = np.zeros((2, 5))
    truth[1, 1] = 1.0
    got = per_sequence_fidelity(np.ones((2, 5)), truth)
    assert np.isnan(got[0]) and np.isfinite(got[1])


def test_random_attribution_is_reproducible():
    np.testing.assert_array_equal(random_attribution((3, 5), 1),
                                  random_attribution((3, 5), 1))


# --------------------------------------------------------------------------
# monotonicity metrics
# --------------------------------------------------------------------------


def test_no_violations_for_a_perfectly_ordered_prediction():
    out = ladder_violations([1.0, 0.8, 0.6], [1.0, 0.8, 0.6])
    assert out["violation_rate"] == 0.0
    assert out["max_increase"] == 0.0
    assert out["n_pairs"] == 3.0


def test_every_pair_violates_for_a_reversed_prediction():
    out = ladder_violations([0.2, 0.5, 0.9], [1.0, 0.8, 0.6])
    assert out["violation_rate"] == 1.0
    assert out["adjacent_violation_rate"] == 1.0
    assert out["max_increase"] == pytest.approx(0.7)


def test_partial_violation_rate():
    # labels 1.0 > 0.8 > 0.6; predictions 0.5, 0.9, 0.7
    # pairs (0,1): 0.9>0.5 violation; (0,2): 0.7>0.5 violation; (1,2): 0.7<0.9 ok
    out = ladder_violations([0.5, 0.9, 0.7], [1.0, 0.8, 0.6])
    assert out["violation_rate"] == pytest.approx(2 / 3)


def test_tolerance_suppresses_tiny_increases():
    out = ladder_violations([0.5, 0.5001], [1.0, 0.8], tolerance=0.001)
    assert out["violation_rate"] == 0.0


def test_ladder_rejects_increasing_labels():
    with pytest.raises(ValueError, match="non-increasing"):
        ladder_violations([1.0, 2.0], [0.5, 0.9])


def test_ladder_rejects_a_single_rung():
    with pytest.raises(ValueError, match="at least two"):
        ladder_violations([1.0], [1.0])


def test_ladder_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must match"):
        ladder_violations([1.0, 2.0], [1.0])


def test_monotonicity_report_aggregates_by_kind():
    ladders = [
        {"quality": np.array([1.0, 0.8]), "kind": "rom"},
        {"quality": np.array([1.0, 0.8]), "kind": "jerk"},
    ]
    preds = [np.array([0.9, 0.7]), np.array([0.5, 0.9])]
    out = monotonicity_report(ladders, preds)
    assert out["violation_rate__rom"] == 0.0
    assert out["violation_rate__jerk"] == 1.0
    assert out["ladders_with_any_violation"] == 0.5
    assert out["n_ladders"] == 2.0


def test_monotonicity_report_rejects_length_mismatch():
    with pytest.raises(ValueError, match="ladders vs"):
        monotonicity_report([{"quality": np.array([1.0, 0.5]), "kind": "rom"}], [])


def test_rank_inconsistency_of_a_valid_cdf_is_zero():
    p = np.array([[0.9, 0.6, 0.2], [0.5, 0.4, 0.1]])
    assert rank_inconsistency(p) == 0.0


def test_rank_inconsistency_detects_crossings():
    p = np.array([[0.2, 0.6, 0.9], [0.9, 0.6, 0.2]])
    assert rank_inconsistency(p) == 0.5


def test_rank_inconsistency_rejects_bad_rank():
    with pytest.raises(ValueError, match=r"expected \(N, K\)"):
        rank_inconsistency(np.zeros(5))


def test_rank_inconsistency_of_a_single_threshold_is_zero():
    assert rank_inconsistency(np.zeros((4, 1))) == 0.0
