"""Spatio-temporal graph convolution, hand-rolled in plain PyTorch.

**Why there is no ``torch-geometric`` or ``mmaction2`` here.** Both are heavy,
version-brittle, and require a compile step or a CUDA-matched wheel that a
reader on a laptop will fight with. And they buy nothing: a skeleton graph has 17
nodes and a *fixed* adjacency, so message passing is one ``einsum`` against a
``(K, V, V)`` constant. Scatter-gather machinery designed for million-node graphs
with dynamic topology is strictly slower here than a dense matmul. The whole
graph convolution below is fifteen lines, is exercised by unit tests against a
hand-computed reference, and makes the repository installable with
``uv sync`` and nothing else. That is a deliberate engineering choice, stated so
it does not read as an omission.

Architecture, following Yan et al. (2018) with two modifications:

* **Separable spatial-temporal factorisation.** The reference ST-GCN block is a
  ``(1, 1)`` graph convolution followed by a dense ``(kt, 1)`` temporal
  convolution costing ``C_out * C_out * kt`` parameters. Here the temporal stage
  is *depthwise* (``groups=C_out``, ``C_out * kt`` parameters) followed by a
  ``1x1`` pointwise mix. That is the Xception/MobileNet factorisation applied
  along time, and at ``kt = 9`` it removes ~``kt``-fold from the temporal stage's
  parameter count.
* **Learnable edge importance** as an elementwise mask on the adjacency, which is
  the cheap half of what 2s-AGCN (Shi et al., 2019) does. The expensive half --
  a data-dependent adjacency computed per sample from an embedded dot product --
  is *not* included, because it costs a second ``V x V`` matmul per layer per
  sample and this project's efficiency claim is about CPU latency. The ablation
  reports what the mask alone is worth.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..data.skeleton import NUM_JOINTS, adjacency


class SpatialGraphConv(nn.Module):
    """Partitioned graph convolution over the joint axis.

    Computes, for input ``x`` of shape ``(N, C_in, T, V)``,

    .. math::

        y_{n,c,t,i} = \\sum_k \\sum_j \\hat{A}^{(k)}_{ij}\\,
                      \\left(W^{(k)} x\\right)_{n,c,t,j}

    i.e. one ``1x1`` convolution per adjacency partition followed by neighbour
    averaging within that partition. ``A[k][i, j]`` carries information *from*
    node ``j`` *into* node ``i``, which is why the ``einsum`` contracts the last
    index of ``A`` with the node index of the transformed features.

    Args:
        in_channels: ``C_in``.
        out_channels: ``C_out``.
        adj: ``(K, V, V)`` normalised adjacency stack.
        edge_importance: Learn a per-edge multiplicative mask, initialised at 1.

    Attributes:
        A: Registered buffer, so it moves with ``.to()`` and is saved in the
           state dict -- a graph is part of the model, not a runtime argument.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj: np.ndarray,
        edge_importance: bool = True,
    ) -> None:
        super().__init__()
        a = torch.as_tensor(np.asarray(adj), dtype=torch.float32)
        if a.ndim != 3 or a.shape[1] != a.shape[2]:
            raise ValueError(f"adj must be (K, V, V), got {tuple(a.shape)}")
        self.num_partitions = int(a.shape[0])
        self.register_buffer("A", a)
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_partitions, kernel_size=1)
        if edge_importance:
            self.edge_weight: nn.Parameter | None = nn.Parameter(torch.ones_like(a))
        else:
            self.edge_weight = None

    def effective_adjacency(self) -> torch.Tensor:
        """The adjacency actually used, after the learnable mask."""
        return self.A if self.edge_weight is None else self.A * self.edge_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_partitions, -1, t, v)
        return torch.einsum("nkctv,kwv->nctw", y, self.effective_adjacency())


class STGCNBlock(nn.Module):
    """One spatio-temporal block: graph conv, temporal conv, residual.

    Args:
        in_channels: ``C_in``.
        out_channels: ``C_out``.
        adj: ``(K, V, V)`` adjacency stack.
        temporal_kernel: ``kt``; must be odd so the convolution is centred.
        stride: Temporal stride (``2`` halves ``T``).
        separable: Depthwise+pointwise temporal stage instead of a dense one.
        dropout: Applied after the temporal stage.
        edge_importance: Passed to :class:`SpatialGraphConv`.
        residual: Add a (projected) skip connection.

    Raises:
        ValueError: if ``temporal_kernel`` is even.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj: np.ndarray,
        temporal_kernel: int = 9,
        stride: int = 1,
        separable: bool = True,
        dropout: float = 0.0,
        edge_importance: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError(f"temporal_kernel must be odd, got {temporal_kernel}")
        pad = (temporal_kernel - 1) // 2

        self.gcn = SpatialGraphConv(in_channels, out_channels, adj, edge_importance)
        # GroupNorm, not BatchNorm. Batch statistics over (N, T, V) are dominated
        # by the frame axis, and at the batch sizes this project trains with
        # (8-32 sequences) the running estimates are noisy enough to change the
        # ranking of two runs. GroupNorm is batch-size independent, which also
        # makes the latency benchmark at batch 1 mean the same thing as training.
        self.norm1 = nn.GroupNorm(min(4, out_channels), out_channels)
        self.act = nn.ReLU(inplace=True)

        temporal: list[nn.Module] = []
        if separable:
            temporal.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(temporal_kernel, 1),
                    stride=(stride, 1),
                    padding=(pad, 0),
                    groups=out_channels,
                )
            )
            temporal.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))
        else:
            temporal.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=(temporal_kernel, 1),
                    stride=(stride, 1),
                    padding=(pad, 0),
                )
            )
        self.tcn = nn.Sequential(*temporal)
        self.norm2 = nn.GroupNorm(min(4, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if not residual:
            self.residual: nn.Module = _Zero()
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.GroupNorm(min(4, out_channels), out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        y = self.act(self.norm1(self.gcn(x)))
        y = self.dropout(self.norm2(self.tcn(y)))
        return self.act(y + res)


class _Zero(nn.Module):
    """A residual branch that contributes nothing (for the no-residual ablation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.zeros(1, dtype=x.dtype, device=x.device)


class STGCNBackbone(nn.Module):
    """Stack of :class:`STGCNBlock` producing a pooled sequence embedding.

    Args:
        in_channels: Input coordinate channels (3 for xyz).
        channels: Output channels per block.
        strides: Temporal stride per block; must match ``channels`` in length.
        partitions: Adjacency scheme -- ``"spatial"``, ``"uniform"`` or
            ``"identity"`` (see :func:`saqa.data.skeleton.adjacency`).
        temporal_kernel: ``kt`` for every block.
        separable: Use the depthwise-separable temporal stage.
        dropout: Block dropout.
        edge_importance: Learnable edge mask.
        num_joints: ``V``.

    Attributes:
        out_channels: Width of the emitted embedding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: tuple[int, ...] = (32, 64, 64, 128),
        strides: tuple[int, ...] = (1, 2, 1, 2),
        partitions: str = "spatial",
        temporal_kernel: int = 9,
        separable: bool = True,
        dropout: float = 0.05,
        edge_importance: bool = True,
        num_joints: int = NUM_JOINTS,
    ) -> None:
        super().__init__()
        if len(channels) != len(strides):
            raise ValueError(
                f"channels and strides must align: {len(channels)} vs {len(strides)}"
            )
        adj = adjacency(partitions)
        if adj.shape[-1] != num_joints:
            raise ValueError(f"adjacency is for {adj.shape[-1]} joints, not {num_joints}")

        # Input normalisation over the joint-coordinate axis. Applied as a
        # GroupNorm over C*V so that each (joint, axis) gets its own affine term:
        # ankles and wrists occupy very different coordinate ranges, and a single
        # shared scale would let the largest-range joint dominate the first layer.
        self.input_norm = nn.GroupNorm(1, in_channels * num_joints)
        self.num_joints = num_joints

        blocks: list[nn.Module] = []
        prev = in_channels
        for c, s in zip(channels, strides, strict=True):
            blocks.append(
                STGCNBlock(
                    prev,
                    c,
                    adj,
                    temporal_kernel=temporal_kernel,
                    stride=s,
                    separable=separable,
                    dropout=dropout,
                    edge_importance=edge_importance,
                )
            )
            prev = c
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = prev

    def forward(self, x: torch.Tensor, return_map: bool = False):
        """Args:
            x: ``(N, C, T, V)``.
            return_map: Also return the final feature map, which the attribution
                code needs.

        Returns:
            ``(N, out_channels)`` pooled embedding, or ``(embedding, feature_map)``.
        """
        n, c, t, v = x.shape
        h = self.input_norm(x.permute(0, 1, 3, 2).reshape(n, c * v, t)).reshape(
            n, c, v, t
        ).permute(0, 1, 3, 2)
        for block in self.blocks:
            h = block(h)
        pooled = h.mean(dim=(2, 3))
        return (pooled, h) if return_map else pooled
