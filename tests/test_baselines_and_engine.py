"""DTW, kinematic features, the fitted baselines, and the training loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from saqa.baselines.dtw import (
    DISTANCES,
    dtw_distance,
    dtw_similarity,
    frame_cost_matrix,
    framewise_distance,
    per_joint_dtw_deviation,
)
from saqa.baselines.fitted import (
    DTWBaseline,
    KinematicGBRBaseline,
    conformal_halfwidth,
)
from saqa.baselines.kinematic import feature_matrix, kinematic_features
from saqa.config import load_config
from saqa.data.actions import ACTION_CLASSES
from saqa.data.dataset import split_indices
from saqa.data.skeleton import NUM_JOINTS
from saqa.engine.trainer import iterate_batches, lr_at, predict, set_threads, train_model
from saqa.models.registry import build_model

# --------------------------------------------------------------------------
# DTW
# --------------------------------------------------------------------------


def _seq(seed=0, t=20):
    return np.random.default_rng(seed).normal(0, 0.2, size=(t, NUM_JOINTS, 3))


@pytest.mark.parametrize("distance", DISTANCES)
def test_cost_matrix_shape_and_non_negativity(distance):
    c = frame_cost_matrix(_seq(0, 12), _seq(1, 15), distance)
    assert c.shape == (12, 15)
    assert (c >= -1e-12).all()


@pytest.mark.parametrize("distance", DISTANCES)
def test_self_distance_is_zero(distance):
    q = _seq(2, 14)
    assert dtw_similarity(q, q, distance) == pytest.approx(0.0, abs=1e-9)


def test_cost_matrix_accepts_channel_first():
    q = _seq(3, 10)
    a = frame_cost_matrix(q, q, "l2")
    b = frame_cost_matrix(q.transpose(2, 0, 1), q, "l2")
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_cost_matrix_rejects_unknown_distance():
    with pytest.raises(ValueError, match="Unknown distance"):
        frame_cost_matrix(_seq(), _seq(1), "mahalanobis")


def test_cost_matrix_rejects_bad_rank():
    with pytest.raises(ValueError, match="3-D sequence"):
        frame_cost_matrix(np.zeros((4, 4)), _seq())


def test_dtw_with_a_symmetric_distance_is_symmetric():
    a, b = _seq(4, 12), _seq(5, 12)
    assert dtw_similarity(a, b, "l2") == pytest.approx(dtw_similarity(b, a, "l2"), rel=1e-6)


def test_per_joint_norm_is_deliberately_directed():
    """``per_joint_norm`` divides by the *reference's* per-joint spread, so it is
    a directed distance rather than a metric: d(query, ref) != d(ref, query) by
    construction. That is the correct behaviour for a reference-based baseline --
    the reference defines the scale -- but it is asserted here so nobody later
    'fixes' it into symmetry and silently changes what the baseline measures."""
    a, b = _seq(4, 12), _seq(5, 12)
    assert dtw_similarity(a, b, "per_joint_norm") != pytest.approx(
        dtw_similarity(b, a, "per_joint_norm"), rel=1e-6
    )


def test_dtw_absorbs_a_pure_time_shift():
    """The point of the alignment: a resampled copy should be much closer under
    DTW than under frame-wise comparison."""
    q = _seq(6, 24)
    idx = np.clip(np.linspace(0, 23, 24) ** 1.15 / 23 ** 0.15, 0, 23)
    warped = np.stack(
        [np.stack([np.interp(idx, np.arange(24), q[:, v, c]) for c in range(3)], -1)
         for v in range(NUM_JOINTS)], axis=1
    )
    assert dtw_similarity(q, warped, "l2") < framewise_distance(q, warped, "l2")


def test_dtw_is_never_greater_than_the_diagonal_cost():
    a, b = _seq(7, 16), _seq(8, 16)
    cost = frame_cost_matrix(a, b, "l2")
    assert dtw_similarity(a, b, "l2", band=None) <= np.mean(np.diagonal(cost)) + 1e-9


def test_dtw_accumulated_matrix_shape():
    cost = frame_cost_matrix(_seq(9, 10), _seq(10, 12), "l2")
    _, acc = dtw_distance(cost, band=None)
    assert acc.shape == (11, 13)


def test_impossible_band_gives_nan_not_a_number():
    """A silently-returned number here would be compared against the other
    configurations as if it meant something."""
    cost = frame_cost_matrix(_seq(11, 8), _seq(12, 40), "l2")
    d, _ = dtw_distance(cost, band=0.0)
    assert np.isnan(d)


def test_framewise_handles_unequal_lengths():
    assert np.isfinite(framewise_distance(_seq(13, 10), _seq(14, 25), "l2"))


def test_per_joint_deviation_shape_and_sign():
    dev = per_joint_dtw_deviation(_seq(15, 12), _seq(16, 12))
    assert dev.shape == (NUM_JOINTS,)
    assert (dev >= 0).all()


def test_per_joint_deviation_of_a_self_comparison_is_zero():
    q = _seq(17, 12)
    assert per_joint_dtw_deviation(q, q).max() == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# kinematic features
# --------------------------------------------------------------------------


def test_features_are_finite_and_named(tiny_data):
    x, names = kinematic_features(tiny_data.coords[0], "squat", ACTION_CLASSES)
    assert x.shape == (len(names),)
    assert np.isfinite(x).all()


def test_features_include_the_defect_vocabulary(tiny_data):
    _, names = kinematic_features(tiny_data.coords[0], "squat", ACTION_CLASSES)
    joined = " ".join(names)
    for token in ("angle_range__", "sym_", "jerk_rms", "pelvis_sway", "energy_centroid"):
        assert token in joined


def test_feature_matrix_shape(tiny_data):
    x, names = feature_matrix(tiny_data.coords[:5], list(tiny_data.actions[:5]),
                              ACTION_CLASSES)
    assert x.shape == (5, len(names))


def test_action_one_hot_is_appended(tiny_data):
    x, names = feature_matrix(tiny_data.coords[:3], list(tiny_data.actions[:3]),
                              ACTION_CLASSES)
    onehot = x[:, -len(ACTION_CLASSES):]
    np.testing.assert_allclose(onehot.sum(axis=1), 1.0)
    assert names[-1] == f"action__{ACTION_CLASSES[-1]}"


def test_symmetry_feature_is_small_for_a_symmetric_movement():
    from saqa.data.actions import assemble_angles, dof_trajectories
    from saqa.data.kinematics import canonicalise, forward_kinematics

    pos = canonicalise(forward_kinematics(assemble_angles(dof_trajectories("squat", 32), 32)))
    x, names = kinematic_features(pos)
    idx = names.index("sym_range_diff__l_knee_r_knee")
    assert x[idx] == pytest.approx(0.0, abs=1e-6)


def test_nonfinite_count_is_reported_not_hidden():
    _, names = kinematic_features(np.zeros((8, NUM_JOINTS, 3)))
    assert "n_nonfinite" in names


def test_features_rejects_bad_rank():
    with pytest.raises(ValueError, match="3-D sequence"):
        kinematic_features(np.zeros((4, 4)))


# --------------------------------------------------------------------------
# fitted baselines
# --------------------------------------------------------------------------


def test_conformal_halfwidth_covers_the_nominal_fraction():
    r = np.abs(np.random.default_rng(0).normal(0, 1, 1000))
    h = conformal_halfwidth(r, 0.9)
    assert np.mean(r <= h) >= 0.89


def test_conformal_halfwidth_of_empty_residuals_is_nan():
    assert np.isnan(conformal_halfwidth(np.array([])))


def test_dtw_baseline_predicts_before_fit_raises(tiny_data):
    with pytest.raises(RuntimeError, match="before fit"):
        DTWBaseline().predict(tiny_data)


def test_gbr_baseline_predicts_before_fit_raises(tiny_data):
    with pytest.raises(RuntimeError, match="before fit"):
        KinematicGBRBaseline().predict(tiny_data)


def test_dtw_baseline_round_trip(tiny_data, tiny_cfg):
    idx = split_indices(tiny_data, "subject", seed=0)
    train, test = tiny_data.subset(idx["train"]), tiny_data.subset(idx["test"])
    model = DTWBaseline(generator=tiny_cfg).fit(train)
    score, lo, hi = model.predict(test)
    assert score.shape == lo.shape == hi.shape == (len(test),)
    assert (hi >= lo).all()
    assert np.isfinite(score).all()


def test_dtw_baseline_name_records_its_configuration(tiny_cfg):
    name = DTWBaseline(distance="l2", band=None, reference_source="canonical").name
    assert "l2" in name and "unbanded" in name and "canonical" in name


def test_dtw_exemplar_reference_comes_from_the_training_split(tiny_data, tiny_cfg):
    idx = split_indices(tiny_data, "subject", seed=0)
    train = tiny_data.subset(idx["train"])
    model = DTWBaseline(generator=tiny_cfg, reference_source="exemplar").fit(train)
    for action, ref in model._references.items():
        assert any(
            s.action == action and np.array_equal(s.positions, ref) for s in train.samples
        ), f"reference for {action} is not a training sequence"


def test_gbr_baseline_round_trip(tiny_data):
    idx = split_indices(tiny_data, "subject", seed=0)
    train, test = tiny_data.subset(idx["train"]), tiny_data.subset(idx["test"])
    model = KinematicGBRBaseline(n_estimators=20, action_classes=ACTION_CLASSES).fit(train)
    score, lo, hi = model.predict(test)
    assert score.shape == (len(test),)
    assert (hi >= lo).all(), "quantile regressors must be sorted before returning"


def test_gbr_feature_importance_is_named_and_sorted(tiny_data):
    idx = split_indices(tiny_data, "subject", seed=0)
    model = KinematicGBRBaseline(n_estimators=20).fit(tiny_data.subset(idx["train"]))
    importance = list(model.feature_importance().items())
    assert importance[0][1] >= importance[-1][1]
    assert all(isinstance(k, str) for k, _ in importance)


# --------------------------------------------------------------------------
# training engine
# --------------------------------------------------------------------------


def test_set_threads_caps_the_pool():
    set_threads(2)
    assert torch.get_num_threads() <= 2


def test_batches_cover_every_item_exactly_once():
    coords = np.zeros((10, 3, 8, NUM_JOINTS), dtype=np.float32)
    quality = np.arange(10, dtype=np.float32)
    seen = []
    for _, y in iterate_batches(coords, quality, 4, np.random.default_rng(0)):
        seen.extend(y.numpy().tolist())
    assert sorted(seen) == list(range(10))


def test_batches_are_shuffled_but_deterministic_per_seed():
    coords = np.zeros((10, 3, 8, NUM_JOINTS), dtype=np.float32)
    quality = np.arange(10, dtype=np.float32)

    def order(seed):
        return [v for _, y in iterate_batches(coords, quality, 3,
                                              np.random.default_rng(seed))
                for v in y.numpy().tolist()]

    assert order(0) == order(0)
    assert order(0) != order(1)


def test_lr_schedule_warms_up_then_decays():
    cfg = load_config(None, ["optim.epochs=10", "optim.warmup_epochs=2", "optim.lr=0.01"])
    lrs = [lr_at(e, cfg) for e in range(10)]
    assert lrs[0] < lrs[1] <= cfg.optim.lr
    assert lrs[2] == pytest.approx(cfg.optim.lr)
    assert lrs[-1] < lrs[3]
    assert lrs[-1] >= cfg.optim.lr * cfg.optim.min_lr_factor - 1e-12


def test_lr_schedule_without_warmup():
    cfg = load_config(None, ["optim.epochs=5", "optim.warmup_epochs=0"])
    assert lr_at(0, cfg) == pytest.approx(cfg.optim.lr)


def test_predict_returns_intervals_when_available():
    model = build_model("frame_average")
    out = predict(model, np.zeros((6, 3, 8, NUM_JOINTS), dtype=np.float32), 4)
    assert out["score"].shape == (6,)
    assert "lower" in out and "upper" in out


def test_predict_omits_intervals_without_an_uncertainty_head():
    """A zero-width interval would score perfect sharpness and zero coverage,
    which is not what 'no uncertainty head' means."""
    model = build_model("frame_average", uncertainty="none")
    out = predict(model, np.zeros((4, 3, 8, NUM_JOINTS), dtype=np.float32))
    assert "lower" not in out


def test_training_reduces_the_loss_and_selects_a_checkpoint(tiny_data):
    cfg = load_config(None, ["optim.epochs=3", "data.batch_size=8", "run.seed=0"])
    idx = split_indices(tiny_data, "subject", seed=0)
    train, val = tiny_data.subset(idx["train"]), tiny_data.subset(idx["val"])
    model = build_model("frame_average")
    state = train_model(model, train, val, cfg, verbose=False)
    assert len(state.history) == 3
    assert state.history[-1]["train_total"] < state.history[0]["train_total"]
    assert 0 <= state.best_epoch < 3
    assert state.best_state, "the selected checkpoint must be kept"


def test_training_restores_the_selected_checkpoint(tiny_data):
    """Evaluating the last epoch instead of the selected one is a classic
    silent inflation of late-epoch overfitting."""
    cfg = load_config(None, ["optim.epochs=3", "data.batch_size=8"])
    idx = split_indices(tiny_data, "subject", seed=0)
    model = build_model("frame_average")
    state = train_model(model, tiny_data.subset(idx["train"]), tiny_data.subset(idx["val"]),
                        cfg, verbose=False)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state.best_state[key])


def test_history_is_written_as_jsonl(tiny_data, tmp_path):
    import json

    cfg = load_config(None, ["optim.epochs=2", "data.batch_size=8"])
    idx = split_indices(tiny_data, "subject", seed=0)
    log = tmp_path / "history.jsonl"
    train_model(build_model("frame_average"), tiny_data.subset(idx["train"]),
                tiny_data.subset(idx["val"]), cfg, log_path=log, verbose=False)
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["epoch"] == 0
