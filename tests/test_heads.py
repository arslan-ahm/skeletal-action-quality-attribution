"""The scoring and uncertainty heads, including the structural guarantees.

Two guarantees are claimed in the documentation and both are asserted here
rather than assumed:

1. the shared-latent ordinal head's survival function is non-increasing in the
   band index for *every* reachable parameter value (rank consistency), and the
   independent-logit variant is not;
2. the quantile head's outputs are non-decreasing in the quantile level for
   every reachable parameter value (no crossed intervals).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from saqa.metrics.monotonicity import rank_inconsistency
from saqa.models.heads import (
    HeteroscedasticHead,
    OrdinalHead,
    QuantileHead,
    RegressionHead,
)


def _feat(n=16, f=12):
    return torch.randn(n, f)


# --------------------------------------------------------------------------
# ordinal head
# --------------------------------------------------------------------------


def test_ordinal_thresholds_are_strictly_ascending_at_init():
    head = OrdinalHead(12, num_bins=10)
    b = head.thresholds().detach().numpy()
    assert b.shape == (9,)
    assert (np.diff(b) > 0).all()


def test_ordinal_thresholds_stay_ascending_under_arbitrary_parameters():
    """The usual implementation stores K free biases and hopes they stay
    ordered. They cross during training. This parameterisation cannot."""
    head = OrdinalHead(12, num_bins=8)
    rng = np.random.default_rng(0)
    for _ in range(30):
        with torch.no_grad():
            head.first_threshold.copy_(torch.tensor(rng.normal(0, 5, size=1),
                                                    dtype=torch.float32))
            head.increments.copy_(torch.tensor(rng.normal(0, 5, size=6),
                                               dtype=torch.float32))
        assert (np.diff(head.thresholds().detach().numpy()) > 0).all()


def test_ordinal_survival_function_is_non_increasing():
    head = OrdinalHead(12, num_bins=10)
    with torch.no_grad():
        head.increments.copy_(torch.randn_like(head.increments) * 3)
    p = torch.sigmoid(head.logits(_feat(64))).detach().numpy()
    assert rank_inconsistency(p) == 0.0


def test_independent_variant_does_produce_crossings():
    """The control that makes 'structural guarantee' a measurable claim rather
    than a description."""
    torch.manual_seed(0)
    head = OrdinalHead(12, num_bins=10, shared_latent=False)
    with torch.no_grad():
        head.independent.weight.mul_(8.0)
        head.independent.bias.copy_(torch.randn_like(head.independent.bias) * 4)
    p = torch.sigmoid(head.logits(_feat(256))).detach().numpy()
    assert rank_inconsistency(p) > 0.5


def test_independent_variant_has_no_shared_thresholds():
    head = OrdinalHead(12, shared_latent=False)
    with pytest.raises(RuntimeError, match="no shared thresholds"):
        head.thresholds()


def test_ordinal_score_is_within_range():
    head = OrdinalHead(12, score_min=0.05, score_max=1.0)
    s = head(_feat(32)).detach().numpy()
    assert (s >= 0.05 - 1e-6).all() and (s <= 1.0 + 1e-6).all()


def test_ordinal_score_increases_monotonically_with_the_latent():
    """The score is a strictly increasing function of z, so score monotonicity
    reduces entirely to whether the *features* move the right way."""
    head = OrdinalHead(1, num_bins=10)
    with torch.no_grad():
        head.latent.weight.fill_(1.0)
        head.latent.bias.zero_()
    z = torch.linspace(-6, 6, 40)[:, None]
    s = head(z).detach().numpy()
    assert (np.diff(s) > 0).all()


def test_ordinal_score_saturates_at_the_ends():
    head = OrdinalHead(1, num_bins=10)
    with torch.no_grad():
        head.latent.weight.fill_(1.0)
        head.latent.bias.zero_()
    lo = head(torch.tensor([[-60.0]])).item()
    hi = head(torch.tensor([[60.0]])).item()
    assert lo == pytest.approx(0.05, abs=1e-3)
    assert hi == pytest.approx(1.0, abs=1e-3)


def test_ordinal_levels_are_soft_and_bounded():
    head = OrdinalHead(4, num_bins=6, score_min=0.0, score_max=1.0)
    levels = head._levels(torch.tensor([0.0, 0.5, 1.0]))
    assert levels.shape == (3, 5)
    assert (levels >= 0).all() and (levels <= 1).all()
    np.testing.assert_allclose(levels[0].numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(levels[2].numpy(), 1.0, atol=1e-6)


def test_ordinal_levels_are_non_increasing_across_k():
    head = OrdinalHead(4, num_bins=8, score_min=0.0, score_max=1.0)
    levels = head._levels(torch.rand(20)).numpy()
    assert (np.diff(levels, axis=1) <= 1e-6).all()


def test_ordinal_levels_clamp_out_of_range_targets():
    head = OrdinalHead(4, num_bins=6, score_min=0.0, score_max=1.0)
    levels = head._levels(torch.tensor([-3.0, 9.0]))
    assert levels[0].sum() == 0.0
    assert levels[1].sum() == pytest.approx(5.0)


def test_ordinal_loss_is_minimised_near_the_truth():
    torch.manual_seed(0)
    head = OrdinalHead(1, num_bins=10)
    with torch.no_grad():
        head.latent.weight.fill_(1.0)
        head.latent.bias.zero_()
    target = torch.full((32,), 0.8)
    good = head.loss(torch.full((32, 1), 1.5), target)
    bad = head.loss(torch.full((32, 1), -4.0), target)
    assert good < bad


def test_ordinal_rejects_bad_bin_count():
    with pytest.raises(ValueError, match="num_bins"):
        OrdinalHead(4, num_bins=1)


def test_ordinal_rejects_empty_range():
    with pytest.raises(ValueError, match="score_max"):
        OrdinalHead(4, score_min=1.0, score_max=1.0)


# --------------------------------------------------------------------------
# regression head
# --------------------------------------------------------------------------


def test_regression_head_stays_in_range():
    s = RegressionHead(12)(_feat(32)).detach().numpy()
    assert (s >= 0.05 - 1e-6).all() and (s <= 1.0 + 1e-6).all()


def test_regression_loss_is_zero_at_a_perfect_fit():
    head = RegressionHead(4)
    feats = _feat(8, 4)
    target = head(feats).detach()
    assert head.loss(feats, target).item() == pytest.approx(0.0, abs=1e-7)


# --------------------------------------------------------------------------
# quantile head
# --------------------------------------------------------------------------


def test_quantiles_never_cross_for_any_parameters():
    torch.manual_seed(0)
    head = QuantileHead(12, (0.05, 0.5, 0.95))
    for _ in range(20):
        with torch.no_grad():
            for p in head.parameters():
                p.copy_(torch.randn_like(p) * 5)
        out = head(_feat(64)).detach().numpy()
        assert (np.diff(out, axis=1) >= -1e-6).all()


def test_quantile_head_output_shape():
    head = QuantileHead(12, (0.1, 0.5, 0.9))
    assert head(_feat(7)).shape == (7, 3)


def test_pinball_loss_matches_its_definition():
    head = QuantileHead(2, (0.1, 0.9))
    feats = torch.zeros(4, 2)
    with torch.no_grad():
        head.base.weight.zero_()
        head.base.bias.fill_(0.0)
        head.deltas.weight.zero_()
        head.deltas.bias.fill_(-20.0)  # softplus(-20) ~ 0, so both quantiles ~ 0
    y = torch.tensor([1.0, 1.0, -1.0, -1.0])
    pred = head(feats).detach()
    q = torch.tensor([0.1, 0.9])
    diff = y[:, None] - pred
    expected = torch.maximum(q * diff, (q - 1.0) * diff).mean()
    assert head.loss(feats, y).item() == pytest.approx(expected.item(), abs=1e-6)


def test_pinball_loss_pushes_the_median_to_the_median():
    torch.manual_seed(0)
    head = QuantileHead(1, (0.25, 0.5, 0.75))
    feats = torch.ones(256, 1)
    y = torch.tensor(np.random.default_rng(0).normal(2.0, 1.0, 256), dtype=torch.float32)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        head.loss(feats, y).backward()
        opt.step()
    pred = head(feats[:1]).detach().numpy()[0]
    assert pred[1] == pytest.approx(float(np.median(y.numpy())), abs=0.15)
    assert pred[0] == pytest.approx(float(np.quantile(y.numpy(), 0.25)), abs=0.2)


def test_quantile_head_rejects_unsorted_levels():
    with pytest.raises(ValueError, match="ascending"):
        QuantileHead(4, (0.9, 0.1))


def test_quantile_head_rejects_out_of_range_levels():
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        QuantileHead(4, (0.0, 0.5))


# --------------------------------------------------------------------------
# heteroscedastic head
# --------------------------------------------------------------------------


def test_heteroscedastic_sigma_is_positive():
    _, sigma = HeteroscedasticHead(12)(_feat(32))
    assert (sigma > 0).all()


def test_heteroscedastic_log_variance_is_clamped():
    head = HeteroscedasticHead(4, min_log_var=-6.0)
    with torch.no_grad():
        head.log_var.bias.fill_(-500.0)
    _, sigma = head(torch.zeros(4, 4))
    assert sigma.min().item() == pytest.approx(float(np.exp(-3.0)), rel=1e-4)


def test_gaussian_nll_prefers_the_right_variance():
    head = HeteroscedasticHead(1)
    feats = torch.ones(512, 1)
    y = torch.tensor(np.random.default_rng(1).normal(0.0, 0.5, 512), dtype=torch.float32)
    with torch.no_grad():
        head.mean.weight.zero_()
        head.mean.bias.zero_()
        head.log_var.weight.zero_()
    losses = {}
    for lv in (-4.0, float(np.log(0.25)), 2.0):
        with torch.no_grad():
            head.log_var.bias.fill_(lv)
        losses[lv] = head.loss(feats, y).item()
    assert min(losses, key=losses.get) == pytest.approx(float(np.log(0.25)))
