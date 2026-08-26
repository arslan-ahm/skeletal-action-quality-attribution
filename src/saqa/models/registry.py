"""Named model builders, so a config string is all a script needs.

Six architectures, and the *point* of the set is that they differ in exactly one
structural property at a time:

======================  ==============================================================
``saqa_stgcn``          ours: separable spatio-temporal graph conv, 3 partitions
``saqa_stgcn_tiny``     ours, half the channel budget -- the efficiency floor
``stgcn_reference``     reference-scale ST-GCN: dense temporal convs, wide channels,
                        the architecture the reference repo's lineage would use
``tcn``                 no graph, temporal convolutions only
``lstm``                no graph, recurrent
``frame_average``       no temporal modelling
======================  ==============================================================
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from ..data.skeleton import NUM_JOINTS
from .quality import QualityModel
from .stgcn import STGCNBackbone
from .trunks import FrameAverageTrunk, LSTMTrunk, TemporalCNNTrunk


def _saqa_trunk(**kw) -> nn.Module:
    return STGCNBackbone(
        channels=kw.pop("channels", (32, 64, 64, 128)),
        strides=kw.pop("strides", (1, 2, 1, 2)),
        partitions=kw.pop("partitions", "spatial"),
        temporal_kernel=kw.pop("temporal_kernel", 9),
        separable=kw.pop("separable", True),
        dropout=kw.pop("dropout", 0.05),
        edge_importance=kw.pop("edge_importance", True),
        num_joints=kw.pop("num_joints", NUM_JOINTS),
        in_channels=kw.pop("in_channels", 3),
    )


def _tiny_trunk(**kw) -> nn.Module:
    kw.setdefault("channels", (16, 32, 64))
    kw.setdefault("strides", (1, 2, 2))
    return _saqa_trunk(**kw)


def _reference_trunk(**kw) -> nn.Module:
    """Reference-scale ST-GCN in the spirit of Yan et al. (2018).

    Nine blocks at 64/128/256 channels with **dense** ``9x1`` temporal
    convolutions -- the shape the original architecture takes, scaled down only
    in that the original operates on 300 frames and 25 joints. It is here to be
    beaten on cost, so it is not crippled: same partitions, same edge importance,
    same head, same training budget.
    """
    kw.setdefault("channels", (64, 64, 64, 128, 128, 128, 256, 256, 256))
    kw.setdefault("strides", (1, 1, 1, 2, 1, 1, 2, 1, 1))
    kw.setdefault("separable", False)
    return _saqa_trunk(**kw)


def _tcn_trunk(**kw) -> nn.Module:
    return TemporalCNNTrunk(
        in_channels=kw.pop("in_channels", 3),
        num_joints=kw.pop("num_joints", NUM_JOINTS),
        channels=kw.pop("channels", (64, 64, 96)),
        kernel_size=kw.pop("temporal_kernel", 9),
        dilations=kw.pop("dilations", (1, 2, 4)),
        dropout=kw.pop("dropout", 0.05),
    )


def _lstm_trunk(**kw) -> nn.Module:
    return LSTMTrunk(
        in_channels=kw.pop("in_channels", 3),
        num_joints=kw.pop("num_joints", NUM_JOINTS),
        hidden=kw.pop("hidden", 64),
        layers=kw.pop("layers", 2),
        dropout=kw.pop("dropout", 0.05),
    )


def _frameavg_trunk(**kw) -> nn.Module:
    return FrameAverageTrunk(
        in_channels=kw.pop("in_channels", 3),
        num_joints=kw.pop("num_joints", NUM_JOINTS),
        hidden=kw.pop("hidden", (128, 96)),
        dropout=kw.pop("dropout", 0.05),
    )


TRUNKS: dict[str, Callable[..., nn.Module]] = {
    "saqa_stgcn": _saqa_trunk,
    "saqa_stgcn_tiny": _tiny_trunk,
    "stgcn_reference": _reference_trunk,
    "tcn": _tcn_trunk,
    "lstm": _lstm_trunk,
    "frame_average": _frameavg_trunk,
}

ARCHITECTURES: tuple[str, ...] = tuple(TRUNKS)


def build_trunk(name: str, **kwargs) -> nn.Module:
    """Instantiate a trunk by name.

    Raises:
        KeyError: on an unknown name, listing the known ones.
    """
    if name not in TRUNKS:
        raise KeyError(f"Unknown architecture {name!r}; known: {ARCHITECTURES}")
    return TRUNKS[name](**dict(kwargs))


def build_model(
    architecture: str = "saqa_stgcn",
    head: str = "ordinal",
    uncertainty: str = "quantile",
    num_bins: int = 10,
    score_min: float = 0.05,
    score_max: float = 1.0,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    uncertainty_weight: float = 0.5,
    **trunk_kwargs,
) -> QualityModel:
    """Build a complete :class:`~saqa.models.quality.QualityModel`.

    Args:
        architecture: Key into :data:`TRUNKS`.
        head: Scoring head kind.
        uncertainty: Uncertainty head kind.
        num_bins: Ordinal bands.
        score_min: Bottom of the score scale.
        score_max: Top of the score scale.
        quantiles: Quantile levels.
        uncertainty_weight: Weight on the uncertainty loss.
        **trunk_kwargs: Forwarded to the trunk builder.

    Returns:
        The model, on CPU, in train mode.
    """
    trunk = build_trunk(architecture, **trunk_kwargs)
    return QualityModel(
        trunk,
        head=head,
        uncertainty=uncertainty,
        num_bins=num_bins,
        score_min=score_min,
        score_max=score_max,
        quantiles=tuple(quantiles),
        uncertainty_weight=uncertainty_weight,
    )
