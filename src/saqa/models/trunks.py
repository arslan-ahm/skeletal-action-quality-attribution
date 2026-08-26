"""Graph-free trunks, so the graph can be isolated as a contribution.

Three ablation trunks, each removing exactly one thing:

``TemporalCNNTrunk``
    Keeps temporal modelling, **removes the graph**: joints are flattened into
    ``3V`` channels and mixed by dense ``1x1`` convolutions, so the network can
    still learn any joint interaction it likes -- it just is not *told* the
    skeleton. If the graph is worth anything, it is worth it here.
``LSTMTrunk``
    Keeps temporal modelling with a different inductive bias (recurrent rather
    than convolutional), again with no graph. Included because a temporal CNN and
    an LSTM fail differently, and attributing a loss to "no graph" needs both.
``FrameAverageTrunk``
    **Removes temporal modelling entirely**: the frame axis is averaged before
    anything else happens. Any score it achieves is achievable from the mean
    pose alone, which makes it the floor that every temporal claim in this
    repository has to clear.

All three are given the same channel budget and the same head, so a difference
between them and the ST-GCN is a difference in structure rather than capacity.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalCNNTrunk(nn.Module):
    """Dilated 1-D convolutional trunk over flattened joint coordinates.

    Args:
        in_channels: Coordinate channels (3).
        num_joints: ``V``.
        channels: Width per layer.
        kernel_size: Temporal kernel, odd.
        dilations: Dilation per layer; must match ``channels`` in length. Growing
            dilation is what gives a stack of three layers a receptive field
            comparable to the ST-GCN's strided stack, so the comparison is about
            structure rather than about how far back each model can see.
        dropout: Applied per layer.

    Attributes:
        out_channels: Embedding width.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = 17,
        channels: tuple[int, ...] = (64, 64, 96),
        kernel_size: int = 9,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if len(channels) != len(dilations):
            raise ValueError("channels and dilations must align")
        layers: list[nn.Module] = []
        prev = in_channels * num_joints
        self.input_norm = nn.GroupNorm(1, prev)
        for c, d in zip(channels, dilations, strict=True):
            layers += [
                nn.Conv1d(prev, c, kernel_size, padding=d * (kernel_size - 1) // 2,
                          dilation=d),
                nn.GroupNorm(min(4, c), c),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
            prev = c
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x: torch.Tensor, return_map: bool = False):
        """Args:
            x: ``(N, C, T, V)``.
            return_map: Also return the ``(N, C', T, 1)`` feature map.
        """
        n, c, t, v = x.shape
        h = x.permute(0, 1, 3, 2).reshape(n, c * v, t)
        h = self.net(self.input_norm(h))
        pooled = h.mean(dim=2)
        return (pooled, h[:, :, :, None]) if return_map else pooled


class LSTMTrunk(nn.Module):
    """Bidirectional LSTM over flattened joint coordinates.

    Args:
        in_channels: Coordinate channels.
        num_joints: ``V``.
        hidden: Hidden width per direction.
        layers: Stacked LSTM layers.
        bidirectional: Use both directions. Quality assessment is offline, so
            there is no causality constraint to respect and a unidirectional
            model would be handicapped for no reason.
        dropout: Inter-layer dropout.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = 17,
        hidden: int = 64,
        layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        feat = in_channels * num_joints
        self.input_norm = nn.GroupNorm(1, feat)
        self.lstm = nn.LSTM(
            feat,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out_channels = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, return_map: bool = False):
        n, c, t, v = x.shape
        h = self.input_norm(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        out, _ = self.lstm(h.transpose(1, 2))
        pooled = out.mean(dim=1)
        if return_map:
            return pooled, out.transpose(1, 2)[:, :, :, None]
        return pooled


class FrameAverageTrunk(nn.Module):
    """Average over frames first, then an MLP. No temporal modelling at all.

    Deliberately not a strawman: it still sees every joint coordinate and has a
    comparable parameter count, so what it lacks is *only* the ability to use
    the order of the frames.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = 17,
        hidden: tuple[int, ...] = (128, 96),
        dropout: float = 0.05,
        include_std: bool = True,
    ) -> None:
        super().__init__()
        # The per-joint temporal standard deviation is included because without
        # it the trunk cannot see *any* movement -- the mean pose of a squat and
        # of standing still are nearly identical. Including it makes this a fair
        # "no temporal order" baseline rather than a "no motion" one: it knows
        # how much each joint moved, just not when or in what sequence.
        self.include_std = include_std
        feat = in_channels * num_joints * (2 if include_std else 1)
        layers: list[nn.Module] = []
        prev = feat
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True),
                       nn.Dropout(dropout) if dropout > 0 else nn.Identity()]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x: torch.Tensor, return_map: bool = False):
        n = x.shape[0]
        mean = x.mean(dim=2).reshape(n, -1)
        feats = (
            torch.cat([mean, x.std(dim=2, unbiased=False).reshape(n, -1)], dim=1)
            if self.include_std
            else mean
        )
        pooled = self.net(feats)
        if return_map:
            return pooled, pooled[:, :, None, None]
        return pooled
