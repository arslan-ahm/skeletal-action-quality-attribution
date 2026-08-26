"""Scoring heads: ordinal, plain regression, and two uncertainty heads.

Quality is *ordinal*. A judge does not compute a real number; they decide that an
execution is better than one band and worse than another. Two consequences are
designed for here.

**Rank consistency should be structural, not hoped for.** The ordinal head
follows CORAL (Cao et al., 2020): ``K`` binary sub-problems "is the score above
band ``k``?" share **one** scalar latent ``z`` and differ only in an ordered bias,

.. math::

    P(y > k \\mid x) = \\sigma(z(x) - b_k), \\qquad b_1 < b_2 < \\dots < b_K,

so the predicted survival function is non-increasing in ``k`` *by construction* --
it cannot cross itself. A head with ``K`` independent linear outputs can and does
produce ``P(y>2) > P(y>1)``, which is not a probability distribution over an
ordered variable at all. The ``rank_inconsistency`` metric measures exactly this,
and the independent-logits variant is shipped so the difference can be reported
rather than asserted.

**What the ordinal head does *not* guarantee.** It does not guarantee that adding
a degradation lowers the predicted score. That would require monotonicity in the
*input*, and no pooling-plus-graph-convolution architecture provides it. The
predicted score is a monotone increasing function of ``z``, so degradation
monotonicity reduces to whether ``z`` decreases when a defect is added -- an
empirical question about the learned features. This repository measures it
(:mod:`saqa.metrics.monotonicity`) instead of claiming it, which is the whole
point of the exercise.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as f


class OrdinalHead(nn.Module):
    """CORAL-style ordinal head over ``num_bins`` quality bands.

    Args:
        in_features: Embedding width.
        num_bins: Number of ordered quality bands ``B``; there are ``B - 1``
            thresholds.
        score_min: Bottom of the score range the head maps onto.
        score_max: Top of the score range.
        shared_latent: If ``False``, each threshold gets its own linear map --
            the deliberately-broken variant used to demonstrate rank
            inconsistency.

    Raises:
        ValueError: if ``num_bins < 2`` or the score range is empty.
    """

    def __init__(
        self,
        in_features: int,
        num_bins: int = 10,
        score_min: float = 0.05,
        score_max: float = 1.0,
        shared_latent: bool = True,
    ) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if score_max <= score_min:
            raise ValueError(f"score_max must exceed score_min, got {score_min}, {score_max}")
        self.num_bins = num_bins
        self.num_thresholds = num_bins - 1
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.shared_latent = shared_latent

        if shared_latent:
            self.latent = nn.Linear(in_features, 1)
            # Thresholds are parameterised by unconstrained increments passed
            # through softplus and cumulatively summed, so b_1 < ... < b_K holds
            # for every value the optimiser can reach. Storing K free biases and
            # hoping they stay ordered is the usual implementation and it does
            # not hold: they cross during training.
            self.first_threshold = nn.Parameter(torch.zeros(1))
            self.increments = nn.Parameter(torch.full((self.num_thresholds - 1,), -1.0))
        else:
            self.independent = nn.Linear(in_features, self.num_thresholds)

    def thresholds(self) -> torch.Tensor:
        """The ``(K,)`` ordered thresholds ``b``, ascending."""
        if not self.shared_latent:
            raise RuntimeError("independent-logit variant has no shared thresholds")
        steps = f.softplus(self.increments)
        return torch.cat([self.first_threshold, self.first_threshold + steps.cumsum(0)])

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """``(N, K)`` cumulative logits for ``P(y > k)``."""
        if self.shared_latent:
            z = self.latent(features)  # (N, 1)
            return z - self.thresholds()[None, :]
        return self.independent(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Point score in ``[score_min, score_max]``.

        The score is the mean of the ``K`` survival probabilities, which for the
        shared-latent head is a strictly increasing function of ``z`` -- so the
        head has one degree of freedom for the score and the thresholds only
        shape its calibration across bands.
        """
        p = torch.sigmoid(self.logits(features))
        frac = p.mean(dim=1)
        return self.score_min + (self.score_max - self.score_min) * frac

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CORAL loss: mean binary cross-entropy over the ``K`` sub-problems.

        Args:
            features: ``(N, F)`` embeddings.
            target: ``(N,)`` scores in ``[score_min, score_max]``.

        Returns:
            Scalar loss.
        """
        levels = self._levels(target)
        return f.binary_cross_entropy_with_logits(self.logits(features), levels)

    def _levels(self, target: torch.Tensor) -> torch.Tensor:
        """``(N, K)`` soft indicator matrix ``1[y > band_k]``.

        The band edges are equally spaced over the score range. The indicator is
        *soft* (linear interpolation across the edge it straddles) rather than
        hard, because a hard indicator discards the within-band position of the
        target and makes the head blind to the difference between the bottom and
        the top of a band -- which is a tenth of the whole scale here.
        """
        frac = (target - self.score_min) / (self.score_max - self.score_min)
        frac = frac.clamp(0.0, 1.0) * self.num_thresholds
        k = torch.arange(self.num_thresholds, device=target.device, dtype=target.dtype)
        return (frac[:, None] - k[None, :]).clamp(0.0, 1.0)


class RegressionHead(nn.Module):
    """Plain scalar head with a sigmoid squash onto the score range.

    The ablation counterpart to :class:`OrdinalHead`: same trunk, same training
    budget, no ordinal structure.
    """

    def __init__(self, in_features: int, score_min: float = 0.05,
                 score_max: float = 1.0) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.score_min = float(score_min)
        self.score_max = float(score_max)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        frac = torch.sigmoid(self.fc(features)).squeeze(-1)
        return self.score_min + (self.score_max - self.score_min) * frac

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Smooth-L1 on the score. Less sensitive to the floor pile-up than MSE."""
        return f.smooth_l1_loss(self(features), target, beta=0.05)


class QuantileHead(nn.Module):
    """Non-crossing quantile head trained with the pinball loss.

    Koenker & Bassett (1978). The ``q``-th quantile is predicted as

    .. math::

        \\hat{y}_q = \\hat{y}_{q_1} + \\sum_{i<q} \\mathrm{softplus}(\\delta_i),

    so ``q1 <= q2 <= ... <= qm`` holds for every parameter value. Predicting the
    quantiles independently -- the common implementation -- lets them cross, and
    a crossed interval has negative width, which is not an interval.

    Args:
        in_features: Embedding width.
        quantiles: Ascending levels in ``(0, 1)``.

    Raises:
        ValueError: if the levels are not strictly ascending or are out of range.
    """

    def __init__(self, in_features: int, quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)):
        super().__init__()
        q = tuple(float(x) for x in quantiles)
        if list(q) != sorted(q) or len(set(q)) != len(q):
            raise ValueError(f"quantiles must be strictly ascending, got {q}")
        if any(not 0.0 < x < 1.0 for x in q):
            raise ValueError(f"quantiles must lie in (0, 1), got {q}")
        self.quantiles = q
        self.base = nn.Linear(in_features, 1)
        self.deltas = nn.Linear(in_features, len(q) - 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(N, len(quantiles))`` non-decreasing quantile predictions."""
        base = self.base(features)
        steps = f.softplus(self.deltas(features))
        return torch.cat([base, base + steps.cumsum(dim=1)], dim=1)

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean pinball loss over the levels.

        ``L_q(y, yhat) = max(q * (y - yhat), (q - 1) * (y - yhat))``, which is
        minimised in expectation at the true ``q``-th conditional quantile.
        """
        pred = self(features)
        y = target[:, None]
        q = torch.as_tensor(self.quantiles, device=pred.device, dtype=pred.dtype)[None, :]
        diff = y - pred
        return torch.maximum(q * diff, (q - 1.0) * diff).mean()


class HeteroscedasticHead(nn.Module):
    """Gaussian mean/log-variance head, trained with the Gaussian NLL.

    The alternative uncertainty formulation. Unlike the quantile head it assumes
    symmetry, which is wrong near the top of a bounded score scale -- a fact the
    coverage table shows rather than argues.
    """

    def __init__(self, in_features: int, min_log_var: float = -8.0):
        super().__init__()
        self.mean = nn.Linear(in_features, 1)
        self.log_var = nn.Linear(in_features, 1)
        self.min_log_var = float(min_log_var)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean, sigma)``, both ``(N,)``, with sigma strictly positive."""
        mu = self.mean(features).squeeze(-1)
        lv = self.log_var(features).squeeze(-1).clamp(min=self.min_log_var, max=2.0)
        return mu, torch.exp(0.5 * lv)

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu, sigma = self(features)
        var = sigma**2
        return (0.5 * torch.log(var) + 0.5 * (target - mu) ** 2 / var).mean()
