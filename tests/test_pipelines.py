"""Pipeline wiring, the shape-override guard, and end-to-end smoke coverage.

The end-to-end test is the one that catches the class of bug that costs the most
credibility: a results table that runs to completion and prints empty.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from saqa.config import load_config
from saqa.metrics.regression import spearman
from saqa.models.registry import build_model
from saqa.pipelines import (
    attribution_completeness,
    attribution_fidelity,
    compute_attribution,
    evaluate_groups,
    evaluate_predictions,
    generator_config,
    make_splits,
    monotonicity_check,
    ordinal_consistency,
    run_single,
)
from saqa.pipelines.core import SHAPE_OWNING_ARCHITECTURES
from saqa.pipelines.sweeps import ABLATIONS
from saqa.utils.complexity import (
    benchmark_latency,
    count_macs,
    count_parameters,
    model_cost,
)

# --------------------------------------------------------------------------
# config projection and splits
# --------------------------------------------------------------------------


def test_generator_config_mirrors_the_data_section():
    cfg = load_config(None, ["data.num_frames=33", "data.noise_std=0.02"])
    g = generator_config(cfg)
    assert g.num_frames == 33 and g.noise_std == pytest.approx(0.02)


def test_make_splits_sizes_add_up(smoke_config):
    sp = make_splits(smoke_config)
    assert sum(sp.sizes().values()) == smoke_config.data.num_sequences


def test_train_limit_truncates_only_the_training_split(smoke_config):
    full = make_splits(smoke_config)
    smoke_config.data.train_limit = 10
    limited = make_splits(smoke_config)
    assert len(limited.train) == 10
    assert len(limited.val) == len(full.val)
    assert len(limited.test) == len(full.test)


def test_train_limit_above_the_pool_is_a_no_op(smoke_config):
    full = make_splits(smoke_config)
    smoke_config.data.train_limit = 10**6
    assert len(make_splits(smoke_config).train) == len(full.train)


def test_train_limit_subsets_are_random_not_a_prefix(smoke_config):
    """Subjects cycle with the sample index, so taking the head of the array
    would silently correlate the label budget with subject identity."""
    smoke_config.data.train_limit = 12
    sp = make_splits(smoke_config)
    assert len(set(sp.train.subjects.tolist())) > 1


# --------------------------------------------------------------------------
# the shape-override guard
# --------------------------------------------------------------------------


def test_named_variants_keep_their_own_shape(smoke_config):
    """Regression test for a real bug: a generic ``model.channels`` in the config
    collapsed stgcn_dense and stgcn_reference into saqa_stgcn, producing three
    identical arms with identical attribution scores in the comparison table."""
    assert smoke_config.model.channels, "the fixture must set channels for this to bite"
    sizes = {}
    for arch in ("saqa_stgcn", "stgcn_dense", "stgcn_reference"):
        result = run_single(smoke_config, name=f"guard_{arch}", architecture=arch,
                            save=False, verbose=False)
        sizes[arch] = result.metrics["params"]
    assert len(set(sizes.values())) == 3, sizes


def test_shape_owning_set_covers_the_named_variants():
    assert {"stgcn_dense", "stgcn_reference", "stgcn_reference_large"} <= (
        SHAPE_OWNING_ARCHITECTURES
    )


def test_graph_switches_are_not_passed_to_graph_free_trunks(smoke_config):
    """Passing them would be a TypeError, which is the right behaviour -- an
    ablation config that silently did nothing would be worse than a crash."""
    for arch in ("tcn", "lstm", "frame_average"):
        run_single(smoke_config, name=f"free_{arch}", architecture=arch, save=False,
                   verbose=False)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def test_evaluate_predictions_without_intervals_omits_uncertainty(tiny_data):
    out = evaluate_predictions(tiny_data, np.asarray(tiny_data.quality))
    assert "spearman" in out
    assert "coverage" not in out and "aurc" not in out


def test_evaluate_predictions_with_intervals_reports_coverage(tiny_data):
    score = np.asarray(tiny_data.quality, dtype=np.float64)
    out = evaluate_predictions(tiny_data, score, score - 0.1, score + 0.1)
    assert out["coverage"] == pytest.approx(1.0)
    assert out["mean_width"] == pytest.approx(0.2)
    assert np.isfinite(out["aurc"])


def test_evaluate_groups_covers_every_action(tiny_data):
    out = evaluate_groups(tiny_data, np.asarray(tiny_data.quality))
    actions = set(tiny_data.actions.tolist())
    for action in actions:
        assert f"action__spearman__{action}" in out


def test_perfect_predictions_score_perfectly(tiny_data):
    out = evaluate_predictions(tiny_data, np.asarray(tiny_data.quality))
    assert out["mae"] == pytest.approx(0.0)
    assert out["spearman"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# analysis wiring
# --------------------------------------------------------------------------


def test_compute_attribution_shapes(smoke_config, tiny_data):
    model = build_model("frame_average")
    for method in ("integrated_gradients", "occlusion", "gradient"):
        attr = compute_attribution(model, tiny_data.coords[:4], method, smoke_config)
        assert attr.shape == tiny_data.coords[:4].shape[0:1] + tiny_data.coords.shape[2:]


def test_compute_attribution_rejects_unknown_method(smoke_config, tiny_data):
    with pytest.raises(ValueError, match="Unknown attribution method"):
        compute_attribution(build_model("frame_average"), tiny_data.coords[:2], "shap",
                            smoke_config)


def test_attribution_fidelity_includes_both_controls(smoke_config, tiny_data):
    """Adebayo et al. (2018): a saliency map can look convincing while being
    independent of the model. The untrained-model control is not optional."""
    out = attribution_fidelity(build_model("frame_average"), tiny_data.subset(np.arange(6)),
                               smoke_config, architecture="frame_average")
    assert "random" in out
    assert "integrated_gradients_untrained" in out


def test_attribution_completeness_reports_counts(smoke_config, tiny_data):
    out = attribution_completeness(build_model("frame_average"), tiny_data, smoke_config,
                                   num=4)
    assert out["n"] == 4.0 and np.isfinite(out["completeness_mean_abs"])


def test_monotonicity_check_returns_aligned_ladders(smoke_config):
    report, ladders, preds = monotonicity_check(build_model("frame_average"), smoke_config)
    assert len(ladders) == len(preds) == smoke_config.eval.monotonicity_ladders
    assert 0.0 <= report["violation_rate"] <= 1.0


def test_ordinal_consistency_is_zero_for_the_shared_latent_head(tiny_data):
    out = ordinal_consistency(build_model("frame_average", head="ordinal"), tiny_data)
    assert out["rank_inconsistency"] == 0.0


def test_ordinal_consistency_is_nan_for_a_regression_head(tiny_data):
    out = ordinal_consistency(build_model("frame_average", head="regression"), tiny_data)
    assert np.isnan(out["rank_inconsistency"])


# --------------------------------------------------------------------------
# complexity utilities
# --------------------------------------------------------------------------


def test_parameter_count_matches_torch():
    model = build_model("saqa_stgcn")
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())


def test_macs_of_a_single_linear_layer_is_exact():
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 17, 7))
    assert count_macs(net, (3, 4, 17)) == 3 * 4 * 17 * 7


def test_macs_of_a_conv_is_exact():
    net = torch.nn.Conv2d(3, 5, kernel_size=(3, 1), padding=(1, 0))
    assert count_macs(net, (3, 8, 17)) == 3 * 5 * 3 * 8 * 17


def test_depthwise_conv_macs_account_for_groups():
    net = torch.nn.Conv2d(8, 8, kernel_size=(3, 1), padding=(1, 0), groups=8)
    assert count_macs(net, (8, 8, 17)) == 8 * 3 * 8 * 17


def test_macs_scale_linearly_with_sequence_length():
    """The graph model is O(T). This is the claim the DTW cost curve is set
    against, so it is asserted rather than assumed."""
    model = build_model("saqa_stgcn")
    a = count_macs(model, (3, 24, 17))
    b = count_macs(model, (3, 48, 17))
    assert b / a == pytest.approx(2.0, rel=0.05)


def test_larger_models_have_more_macs():
    small = count_macs(build_model("saqa_stgcn"), (3, 24, 17))
    large = count_macs(build_model("stgcn_reference_large"), (3, 24, 17))
    assert large > 10 * small


def test_macs_leaves_the_model_in_its_original_mode():
    model = build_model("saqa_stgcn")
    model.train()
    count_macs(model, (3, 8, 17))
    assert model.training


def test_benchmark_latency_records_its_warmup():
    res = benchmark_latency(lambda: None, warmup=8, repeats=10)
    assert res.warmup == 8 and res.repeats == 10
    assert res.median_ms >= 0 and res.iqr_ms >= 0


def test_benchmark_latency_rejects_zero_repeats():
    with pytest.raises(ValueError, match="repeats"):
        benchmark_latency(lambda: None, repeats=0)


def test_model_cost_reports_throughput():
    out = model_cost(build_model("frame_average"), (3, 16, 17), (1,), warmup=8, repeats=10)
    assert out["macs_per_ms_bs1"] > 0
    assert out["params"] > 0


# --------------------------------------------------------------------------
# ablation registry
# --------------------------------------------------------------------------


def test_every_ablation_changes_exactly_one_key():
    for name, change in ABLATIONS.items():
        assert len(change) <= 1, f"{name} changes {len(change)} keys; attribution is lost"


def test_ablation_keys_exist_in_the_config():
    for change in ABLATIONS.values():
        for key in change:
            load_config(None, [f"{key}={list(change.values())[0]}"])


def test_full_ablation_is_the_unmodified_default():
    assert ABLATIONS["full"] == {}


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_run_single_end_to_end_writes_populated_artifacts(smoke_config, tmp_path):
    smoke_config.run.out_dir = str(tmp_path)
    result = run_single(smoke_config, name="e2e", architecture="saqa_stgcn", verbose=False)
    out = tmp_path / "e2e"

    for artifact in ("config.yaml", "history.jsonl", "per_item.csv", "summary.json"):
        path = out / artifact
        assert path.exists(), artifact
        assert path.stat().st_size > 0, f"{artifact} is empty"

    import pandas as pd

    per_item = pd.read_csv(out / "per_item.csv")
    assert len(per_item) == len(result.splits.test)
    assert per_item["score"].notna().all()
    assert {"quality", "score", "abs_error", "lower", "upper"} <= set(per_item.columns)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_test"] == len(result.splits.test)
    assert summary["params"] > 0
    # the table must be populated, not merely present
    assert any(v is not None for k, v in summary.items() if k.startswith("action__"))


@pytest.mark.slow
def test_two_identical_runs_are_bit_identical(smoke_config):
    a = run_single(smoke_config, name="det_a", save=False, verbose=False)
    b = run_single(smoke_config, name="det_b", save=False, verbose=False)
    assert np.abs(a.pred["score"] - b.pred["score"]).max() == 0.0


@pytest.mark.slow
def test_different_seeds_give_different_models(smoke_config):
    a = run_single(smoke_config, name="s0", save=False, verbose=False)
    smoke_config.run.seed = 5
    b = run_single(smoke_config, name="s5", save=False, verbose=False)
    assert np.abs(a.pred["score"] - b.pred["score"]).max() > 0


@pytest.mark.slow
def test_training_fits_its_own_training_set_better_than_random_weights(smoke_config):
    """The training loop's job is to fit. Generalisation is a separate question
    and is deliberately *not* asserted here: at this fixture's 45 training
    sequences an untrained network can outscore a trained one on the test split,
    because a random projection of movement amplitude already correlates with
    defect severity. That is a real property of the task, it is reported as the
    ``untrained_stgcn`` control arm in the results, and pretending otherwise in a
    unit test would hide it."""
    smoke_config.optim.epochs = 12
    result = run_single(smoke_config, name="learns", architecture="tcn", save=False,
                        verbose=False)
    from saqa.engine.trainer import predict as _predict

    trained = spearman(result.splits.train.quality,
                       _predict(result.model, result.splits.train.coords)["score"])
    base = spearman(result.splits.train.quality,
                    _predict(build_model("tcn"), result.splits.train.coords)["score"])
    assert trained > base
