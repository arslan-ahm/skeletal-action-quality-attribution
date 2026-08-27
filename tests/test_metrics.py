"""Metrics against closed forms, invariants and the NaN contract."""

from __future__ import annotations

import numpy as np
import pytest

from saqa.metrics.regression import (
    kendall_tau,
    pearson,
    per_group_summary,
    per_item_absolute_error,
    regression_summary,
    relative_l2,
    spearman,
    tie_fraction,
)
from saqa.metrics.stats import (
    bootstrap_ci,
    bootstrap_metric_difference,
    compare,
    holm_bonferroni,
    noise_scale,
    paired_bootstrap_difference,
    verdict,
)
from saqa.metrics.uncertainty import (
    aurc,
    coverage,
    crossing_rate,
    error_detection_auroc,
    excess_aurc,
    interval_summary,
    mean_width,
    risk_coverage_curve,
)

# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------


def test_spearman_of_a_perfect_ranking_is_one():
    t = np.arange(20, dtype=float)
    assert spearman(t, t * 3.0 + 1.0) == pytest.approx(1.0)


def test_spearman_of_a_reversed_ranking_is_minus_one():
    t = np.arange(20, dtype=float)
    assert spearman(t, -t) == pytest.approx(-1.0)


def test_spearman_is_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(0)
    t, p = rng.random(50), rng.random(50)
    assert spearman(t, p) == pytest.approx(spearman(t, np.exp(3 * p)))


def test_spearman_is_nan_for_a_constant_input():
    """0.0 would be read as 'no relationship measured', which is a different
    claim from 'undefined'."""
    assert np.isnan(spearman(np.ones(10), np.arange(10.0)))
    assert np.isnan(spearman(np.arange(10.0), np.ones(10)))


def test_spearman_is_nan_below_three_points():
    assert np.isnan(spearman([1.0, 2.0], [1.0, 2.0]))


def test_kendall_matches_scipy_on_a_known_case():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 3.0, 2.0, 4.0])
    # 5 concordant, 1 discordant out of 6 pairs -> (5-1)/6
    assert kendall_tau(t, p) == pytest.approx(4.0 / 6.0)


def test_pearson_of_a_linear_map_is_one():
    t = np.arange(30, dtype=float)
    assert pearson(t, 2 * t - 5) == pytest.approx(1.0)


def test_relative_l2_of_a_perfect_fit_is_zero():
    t = np.arange(1.0, 11.0)
    assert relative_l2(t, t) == pytest.approx(0.0)


def test_relative_l2_matches_its_definition():
    t = np.array([3.0, 4.0])
    p = np.array([0.0, 0.0])
    assert relative_l2(t, p) == pytest.approx(1.0)


def test_relative_l2_is_nan_for_an_all_zero_truth():
    assert np.isnan(relative_l2(np.zeros(5), np.ones(5)))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="must match"):
        spearman(np.zeros(4), np.zeros(5))


def test_nan_entries_are_dropped_pairwise():
    t = np.array([1.0, 2.0, np.nan, 4.0])
    p = np.array([1.0, 2.0, 3.0, np.nan])
    assert regression_summary(t, p)["n"] == 2.0


def test_per_item_absolute_error():
    np.testing.assert_allclose(
        per_item_absolute_error([1.0, 2.0], [1.5, 0.0]), [0.5, 2.0]
    )


def test_tie_fraction_counts_shared_values():
    assert tie_fraction([1.0, 1.0, 2.0, 3.0]) == pytest.approx(0.5)
    assert tie_fraction([1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_per_group_summary_marks_thin_groups_nan():
    """Two points do not have a rank correlation, and reporting one would be a
    fabricated measurement."""
    t = np.arange(10.0)
    p = t.copy()
    groups = np.array(["a"] * 8 + ["b"] * 2)
    out = per_group_summary(t, p, groups)
    assert out["a"] == pytest.approx(1.0)
    assert np.isnan(out["b"])
    assert out["b__n"] == 2.0


def test_regression_summary_has_every_key():
    out = regression_summary(np.arange(10.0), np.arange(10.0))
    assert set(out) >= {"spearman", "kendall_tau", "pearson", "relative_l2", "mae",
                        "rmse", "n"}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    iv = bootstrap_ci(rng.normal(5.0, 1.0, 200), n_resamples=500)
    assert iv.lower < iv.estimate < iv.upper
    assert iv.n == 200


def test_bootstrap_interval_narrows_with_more_data():
    rng = np.random.default_rng(1)
    wide = bootstrap_ci(rng.normal(0, 1, 30), n_resamples=800)
    narrow = bootstrap_ci(rng.normal(0, 1, 3000), n_resamples=800)
    assert (narrow.upper - narrow.lower) < (wide.upper - wide.lower)


def test_bootstrap_on_empty_input_is_nan_not_zero():
    iv = bootstrap_ci(np.array([]))
    assert np.isnan(iv.estimate) and iv.n == 0


def test_bootstrap_on_a_single_value_collapses():
    iv = bootstrap_ci(np.array([2.0]))
    assert iv.estimate == iv.lower == iv.upper == 2.0


def test_bootstrap_is_reproducible():
    x = np.random.default_rng(2).normal(size=100)
    assert bootstrap_ci(x, 300, seed=5).lower == bootstrap_ci(x, 300, seed=5).lower


def test_paired_bootstrap_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must match"):
        paired_bootstrap_difference(np.zeros(3), np.zeros(4))


def test_compare_detects_a_real_shift():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 200)
    c = compare(base + 0.8, base, n_resamples=500)
    assert c.p_value < 0.01
    assert c.difference.lower > 0
    assert c.significant


def test_compare_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(4)
    c = compare(rng.normal(size=200), rng.normal(size=200), n_resamples=500)
    assert c.p_value > 0.05


def test_compare_p_value_is_nan_for_identical_inputs():
    """The signed-rank test is undefined when every difference is zero."""
    x = np.arange(20.0)
    assert np.isnan(compare(x, x).p_value)


def test_compare_effect_size_matches_paired_cohens_d():
    rng = np.random.default_rng(5)
    a = rng.normal(size=100)
    b = a + rng.normal(0, 0.1, 100)
    d = a - b
    c = compare(a, b, n_resamples=100)
    assert c.effect_size == pytest.approx(d.mean() / d.std(ddof=1), rel=1e-6)


def test_holm_is_monotone_and_bounded():
    rng = np.random.default_rng(6)
    a = rng.normal(size=80)
    comps = [compare(a + shift, a, n_resamples=200)
             for shift in (0.0001, 0.4, 0.8, 1.2)]
    holm_bonferroni(comps)
    adjusted = sorted(c.p_adjusted for c in comps)
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    assert adjusted == sorted(adjusted)
    for c in comps:
        assert c.p_adjusted >= c.p_value - 1e-12


def test_holm_excludes_nan_from_the_family_size():
    x = np.arange(30.0)
    rng = np.random.default_rng(7)
    comps = [compare(x, x), compare(x + rng.normal(0, 0.1, 30), x, n_resamples=200)]
    holm_bonferroni(comps)
    assert comps[0].p_adjusted is None
    assert comps[1].p_adjusted == pytest.approx(comps[1].p_value)


def test_holm_on_an_empty_family_is_a_no_op():
    assert holm_bonferroni([]) == []


def test_bootstrap_metric_difference_on_a_set_level_statistic():
    """Spearman has no per-item value, so the paired test must resample
    sequence indices and recompute the metric."""
    rng = np.random.default_rng(8)
    truth = rng.random(120)
    good = truth + rng.normal(0, 0.02, 120)
    bad = rng.random(120)
    iv = bootstrap_metric_difference(truth, good, bad, spearman, 300)
    assert iv.estimate > 0.5
    assert iv.lower > 0


def test_bootstrap_metric_difference_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shapes must match"):
        bootstrap_metric_difference(np.zeros(4), np.zeros(4), np.zeros(5), spearman, 10)


def test_noise_scale_is_sqrt_two_times_sd():
    v = np.array([1.0, 2.0, 3.0])
    assert noise_scale(v) == pytest.approx(np.sqrt(2.0) * v.std(ddof=1))


def test_noise_scale_needs_two_runs():
    assert np.isnan(noise_scale(np.array([1.0])))


def test_verdict_thresholds():
    assert verdict(0.35, 0.1) == "robust"
    assert verdict(0.25, 0.1) == "survives"
    assert verdict(0.15, 0.1) == "suggestive"
    assert verdict(0.05, 0.1) == "inside noise"
    assert verdict(0.5, float("nan")) == "unknown"


def test_verdict_is_sign_agnostic():
    assert verdict(-0.35, 0.1) == verdict(0.35, 0.1)


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------


def test_coverage_counts_inclusion():
    t = np.array([0.0, 1.0, 2.0])
    assert coverage(t, np.array([-1.0, 0.9, 5.0]),
                    np.array([1.0, 1.1, 6.0])) == pytest.approx(2 / 3)


def test_coverage_is_inclusive_at_the_bounds():
    assert coverage(np.array([1.0]), np.array([1.0]), np.array([1.0])) == 1.0


def test_coverage_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        coverage(np.zeros(3), np.zeros(3), np.zeros(4))


def test_mean_width_and_crossing_rate():
    lo = np.array([0.0, 1.0])
    hi = np.array([1.0, 0.5])
    assert mean_width(lo, hi) == pytest.approx((1.0 - 0.5) / 2)
    assert crossing_rate(lo, hi) == pytest.approx(0.5)


def test_interval_summary_reports_the_width_split():
    truth = np.array([0.0, 0.0, 0.0, 0.0])
    pred = np.array([0.0, 0.0, 0.5, 0.5])  # last two are wrong
    lo = np.array([-0.1, -0.1, -1.0, -1.0])
    hi = np.array([0.1, 0.1, 1.0, 1.0])
    out = interval_summary(truth, lo, hi, pred, nominal=0.9)
    assert out["n_wrong"] == 2.0
    assert out["width_wrong"] == pytest.approx(2.0)
    assert out["width_right"] == pytest.approx(0.2)
    assert out["width_ratio_wrong_right"] == pytest.approx(10.0)


def test_risk_coverage_is_monotone_for_a_perfect_confidence():
    err = np.array([0.0, 0.1, 0.2, 0.9])
    covs, risks = risk_coverage_curve(err, -err)
    assert covs[0] == pytest.approx(0.25) and covs[-1] == pytest.approx(1.0)
    assert (np.diff(risks) >= -1e-12).all()
    assert risks[-1] == pytest.approx(err.mean())


def test_aurc_of_a_perfect_confidence_beats_a_random_one():
    rng = np.random.default_rng(9)
    err = rng.random(200)
    assert aurc(err, -err) < aurc(err, rng.random(200))


def test_excess_aurc_is_zero_for_the_oracle():
    err = np.random.default_rng(10).random(100)
    out = excess_aurc(err, -err)
    assert out["e_aurc"] == pytest.approx(0.0, abs=1e-12)


def test_excess_aurc_is_non_negative_in_practice():
    rng = np.random.default_rng(11)
    err = rng.random(300)
    assert excess_aurc(err, rng.random(300))["e_aurc"] > 0


def test_error_detection_auroc_of_a_perfect_detector():
    err = np.array([0.0, 0.0, 0.5, 0.5])
    assert error_detection_auroc(err, -err, threshold=0.1) == pytest.approx(1.0)


def test_error_detection_auroc_is_nan_when_a_class_is_empty():
    """0.5 would claim a measurement that was not made."""
    err = np.zeros(10)
    assert np.isnan(error_detection_auroc(err, np.random.default_rng(0).random(10)))


def test_error_detection_auroc_of_an_inverted_detector():
    err = np.array([0.0, 0.0, 0.5, 0.5])
    assert error_detection_auroc(err, err, threshold=0.1) == pytest.approx(0.0)


def test_risk_coverage_on_empty_input():
    covs, risks = risk_coverage_curve(np.array([]), np.array([]))
    assert covs.size == 0 and risks.size == 0
    assert np.isnan(aurc(np.array([]), np.array([])))
