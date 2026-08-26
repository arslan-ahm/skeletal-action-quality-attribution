"""The assembled quality model: trunk + ordinal score + uncertainty interval.

The model emits three things and they are computed by three different mechanisms
on purpose:

``score``
    The point estimate, from the ordinal head. One scalar latent, ordered
    thresholds.
``interval``
    A predictive interval, from either a non-crossing quantile head or a
    heteroscedastic Gaussian head. Deliberately *not* derived from the ordinal
    distribution: a discretised ordinal CDF has a resolution floor of one band
    (0.095 of the scale here), which would make the interval width uninformative
    below that. The quantile head is continuous.
``features``
    The pooled embedding, exposed because integrated gradients needs to
    differentiate the score with respect to the *input*, and the occlusion
    attribution needs to re-run the trunk on masked inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .heads import HeteroscedasticHead, OrdinalHead, QuantileHead, RegressionHead


@dataclass
class ModelOutput:
    """Everything one forward pass produces.

    Attributes:
        score: ``(N,)`` point quality estimate.
        features: ``(N, F)`` pooled embedding.
        quantiles: ``(N, Q)`` non-decreasing quantiles, or ``None``.
        sigma: ``(N,)`` predictive standard deviation, or ``None``.
        feature_map: ``(N, C, T', V)`` final trunk map, or ``None``.
    """

    score: torch.Tensor
    features: torch.Tensor
    quantiles: torch.Tensor | None = None
    sigma: torch.Tensor | None = None
    feature_map: torch.Tensor | None = None

    def interval(self, level: float = 0.90) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower and upper bound of the predictive interval.

        Args:
            level: Nominal coverage. Only used by the Gaussian head; the quantile
                head's level is fixed by the quantiles it was trained on, and
                asking it for a different level would silently return the wrong
                one.

        Returns:
            ``(lower, upper)``, each ``(N,)``.

        Raises:
            RuntimeError: if the model has no uncertainty head.
        """
        if self.quantiles is not None:
            return self.quantiles[:, 0], self.quantiles[:, -1]
        if self.sigma is not None:
            z = {0.90: 1.6449, 0.95: 1.9600, 0.80: 1.2816}.get(level)
            if z is None:
                raise ValueError(f"Unsupported level {level}; use 0.80, 0.90 or 0.95")
            return self.score - z * self.sigma, self.score + z * self.sigma
        raise RuntimeError("model was built without an uncertainty head")


class QualityModel(nn.Module):
    """Trunk plus scoring head plus optional uncertainty head.

    Args:
        trunk: Any module exposing ``out_channels`` and
            ``forward(x, return_map=False)``.
        head: ``"ordinal"``, ``"ordinal_independent"`` (rank-inconsistent
            variant, for the demonstration) or ``"regression"``.
        uncertainty: ``"quantile"``, ``"heteroscedastic"`` or ``"none"``.
        num_bins: Ordinal bands.
        score_min / score_max: Score range.
        quantiles: Quantile levels when ``uncertainty="quantile"``.
        uncertainty_weight: Weight of the uncertainty loss in the total. The
            uncertainty head must not be allowed to steer the shared trunk hard
            enough to move the point estimate, or the interval is bought with
            accuracy and the comparison against a point-only model is no longer
            about uncertainty.

    Raises:
        ValueError: on an unknown head or uncertainty type.
    """

    def __init__(
        self,
        trunk: nn.Module,
        head: str = "ordinal",
        uncertainty: str = "quantile",
        num_bins: int = 10,
        score_min: float = 0.05,
        score_max: float = 1.0,
        quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
        uncertainty_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.head_kind = head
        self.uncertainty_kind = uncertainty
        self.uncertainty_weight = float(uncertainty_weight)
        feat = int(trunk.out_channels)

        if head == "ordinal":
            self.head: nn.Module = OrdinalHead(feat, num_bins, score_min, score_max, True)
        elif head == "ordinal_independent":
            self.head = OrdinalHead(feat, num_bins, score_min, score_max, False)
        elif head == "regression":
            self.head = RegressionHead(feat, score_min, score_max)
        else:
            raise ValueError(f"Unknown head {head!r}")

        if uncertainty == "quantile":
            self.unc: nn.Module | None = QuantileHead(feat, quantiles)
        elif uncertainty == "heteroscedastic":
            self.unc = HeteroscedasticHead(feat)
        elif uncertainty == "none":
            self.unc = None
        else:
            raise ValueError(f"Unknown uncertainty {uncertainty!r}")

    def forward(self, x: torch.Tensor, return_map: bool = False) -> ModelOutput:
        out = self.trunk(x, return_map=return_map)
        features, fmap = out if return_map else (out, None)
        score = self.head(features)
        quantiles = sigma = None
        if isinstance(self.unc, QuantileHead):
            quantiles = self.unc(features)
        elif isinstance(self.unc, HeteroscedasticHead):
            _, sigma = self.unc(features)
        return ModelOutput(score, features, quantiles, sigma, fmap)

    def score_only(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable ``(N,)`` score. The function attribution explains."""
        return self.head(self.trunk(x))

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Total loss and its components.

        Args:
            x: ``(N, C, T, V)`` input.
            target: ``(N,)`` quality labels.

        Returns:
            ``(total, parts)`` where ``parts`` maps component name to a float.
        """
        features = self.trunk(x)
        score_loss = self.head.loss(features, target)
        total = score_loss
        parts = {"score": float(score_loss.detach())}
        if self.unc is not None:
            unc_loss = self.unc.loss(features, target)
            total = total + self.uncertainty_weight * unc_loss
            parts["uncertainty"] = float(unc_loss.detach())
        parts["total"] = float(total.detach())
        return total, parts
