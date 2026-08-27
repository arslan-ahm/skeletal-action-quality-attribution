"""Graph convolution against a hand-computed reference, trunks, and the heads.

The most important test in this file is
:func:`test_graph_conv_matches_hand_computed_message_passing`: the whole argument
for hand-rolling the graph convolution instead of pulling in a library is that it
is short enough to verify by hand. So it is verified by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from saqa.data.skeleton import NUM_JOINTS, adjacency
from saqa.models.quality import QualityModel
from saqa.models.registry import ARCHITECTURES, build_model, build_trunk
from saqa.models.stgcn import SpatialGraphConv, STGCNBackbone, STGCNBlock, make_norm
from saqa.models.trunks import FrameAverageTrunk, LSTMTrunk, TemporalCNNTrunk


def _x(n=2, c=3, t=16, v=NUM_JOINTS):
    return torch.randn(n, c, t, v)


# --------------------------------------------------------------------------
# spatial graph convolution
# --------------------------------------------------------------------------


def test_graph_conv_matches_hand_computed_message_passing():
    """y[n,c,t,i] = sum_k sum_j A[k,i,j] * (W_k x)[n,c,t,j], computed explicitly."""
    torch.manual_seed(0)
    adj = adjacency("spatial")
    conv = SpatialGraphConv(3, 4, adj, edge_importance=False)
    x = _x(1, 3, 5)
    got = conv(x)

    transformed = conv.conv(x).view(1, 3, 4, 5, NUM_JOINTS)
    expected = torch.zeros_like(got)
    a = torch.as_tensor(adj, dtype=torch.float32)
    for k in range(3):
        for i in range(NUM_JOINTS):
            for j in range(NUM_JOINTS):
                expected[:, :, :, i] += a[k, i, j] * transformed[:, k, :, :, j]
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)


def test_graph_conv_output_shape():
    conv = SpatialGraphConv(3, 8, adjacency("spatial"))
    assert conv(_x(2, 3, 12)).shape == (2, 8, 12, NUM_JOINTS)


def test_identity_adjacency_does_not_mix_joints():
    """With self-loops only, joint i's output must depend on joint i alone."""
    torch.manual_seed(1)
    conv = SpatialGraphConv(3, 4, adjacency("identity"), edge_importance=False)
    x = _x(1, 3, 4)
    base = conv(x)
    perturbed = x.clone()
    perturbed[:, :, :, 5] += 10.0
    after = conv(perturbed)
    changed = (after - base).abs().sum(dim=(0, 1, 2))
    assert changed[5] > 0
    assert changed[[j for j in range(NUM_JOINTS) if j != 5]].max() < 1e-6


def test_spatial_adjacency_does_mix_neighbours():
    torch.manual_seed(1)
    conv = SpatialGraphConv(3, 4, adjacency("spatial"), edge_importance=False)
    x = _x(1, 3, 4)
    perturbed = x.clone()
    perturbed[:, :, :, 12] += 10.0  # l_knee
    changed = (conv(perturbed) - conv(x)).abs().sum(dim=(0, 1, 2))
    assert changed[11] > 1e-4 and changed[13] > 1e-4  # l_hip and l_ankle


def test_edge_importance_starts_as_a_no_op():
    torch.manual_seed(2)
    adj = adjacency("spatial")
    with_mask = SpatialGraphConv(3, 4, adj, edge_importance=True)
    torch.testing.assert_close(with_mask.effective_adjacency(), with_mask.A)


def test_edge_importance_is_a_learnable_parameter():
    conv = SpatialGraphConv(3, 4, adjacency("spatial"), edge_importance=True)
    assert conv.edge_weight is not None and conv.edge_weight.requires_grad
    assert SpatialGraphConv(3, 4, adjacency("spatial"), False).edge_weight is None


def test_adjacency_is_a_buffer_not_a_parameter():
    """The graph is part of the model: it must be saved in the state dict and
    move with .to(), but never receive a gradient."""
    conv = SpatialGraphConv(3, 4, adjacency("spatial"))
    assert "A" in dict(conv.named_buffers())
    assert "A" not in dict(conv.named_parameters())
    assert "A" in conv.state_dict()


def test_graph_conv_rejects_bad_adjacency():
    with pytest.raises(ValueError, match=r"adj must be"):
        SpatialGraphConv(3, 4, np.zeros((5, 5)))


# --------------------------------------------------------------------------
# blocks and backbone
# --------------------------------------------------------------------------


def test_block_preserves_shape_at_stride_one():
    block = STGCNBlock(3, 8, adjacency("spatial"))
    assert block(_x(2, 3, 16)).shape == (2, 8, 16, NUM_JOINTS)


def test_block_halves_time_at_stride_two():
    block = STGCNBlock(3, 8, adjacency("spatial"), stride=2)
    assert block(_x(2, 3, 16)).shape == (2, 8, 8, NUM_JOINTS)


def test_block_rejects_even_kernel():
    with pytest.raises(ValueError, match="must be odd"):
        STGCNBlock(3, 8, adjacency("spatial"), temporal_kernel=8)


def test_separable_block_has_fewer_parameters():
    sep = STGCNBlock(32, 32, adjacency("spatial"), separable=True)
    dense = STGCNBlock(32, 32, adjacency("spatial"), separable=False)
    n_sep = sum(p.numel() for p in sep.tcn.parameters())
    n_dense = sum(p.numel() for p in dense.tcn.parameters())
    assert n_sep < n_dense
    # depthwise+pointwise is C*kt + C*C vs C*C*kt
    assert n_dense / n_sep > 4.0


def test_make_norm_kinds():
    assert isinstance(make_norm("batch", 8), torch.nn.BatchNorm2d)
    assert isinstance(make_norm("group", 8), torch.nn.GroupNorm)
    with pytest.raises(ValueError, match="Unknown norm"):
        make_norm("layer", 8)


def test_backbone_pools_to_the_declared_width():
    net = STGCNBackbone(channels=(8, 16), strides=(1, 2))
    out = net(_x(3, 3, 16))
    assert out.shape == (3, net.out_channels) == (3, 16)


def test_backbone_returns_a_feature_map_on_request():
    net = STGCNBackbone(channels=(8, 16), strides=(1, 2))
    pooled, fmap = net(_x(2, 3, 16), return_map=True)
    assert pooled.shape == (2, 16)
    assert fmap.shape == (2, 16, 8, NUM_JOINTS)


def test_backbone_rejects_mismatched_channels_and_strides():
    with pytest.raises(ValueError, match="must align"):
        STGCNBackbone(channels=(8, 16), strides=(1,))


def test_backbone_partitions_change_parameter_count():
    three = STGCNBackbone(channels=(8,), strides=(1,), partitions="spatial")
    one = STGCNBackbone(channels=(8,), strides=(1,), partitions="identity")
    assert sum(p.numel() for p in three.parameters()) > sum(
        p.numel() for p in one.parameters()
    )


@pytest.mark.parametrize("trunk_cls", [TemporalCNNTrunk, LSTMTrunk, FrameAverageTrunk])
def test_graph_free_trunks_pool_correctly(trunk_cls):
    trunk = trunk_cls()
    out = trunk(_x(2, 3, 16))
    assert out.shape == (2, trunk.out_channels)


def test_frame_average_ignores_frame_order():
    """The point of the baseline: shuffling frames must not change its output."""
    trunk = FrameAverageTrunk()
    trunk.eval()
    x = _x(1, 3, 16)
    shuffled = x[:, :, torch.randperm(16), :]
    torch.testing.assert_close(trunk(x), trunk(shuffled), atol=1e-5, rtol=1e-4)


def test_temporal_cnn_does_depend_on_frame_order():
    trunk = TemporalCNNTrunk()
    trunk.eval()
    x = _x(1, 3, 16)
    shuffled = x[:, :, torch.randperm(16), :]
    assert (trunk(x) - trunk(shuffled)).abs().max() > 1e-4


def test_tcn_rejects_even_kernel():
    with pytest.raises(ValueError, match="must be odd"):
        TemporalCNNTrunk(kernel_size=4)


def test_tcn_rejects_misaligned_dilations():
    with pytest.raises(ValueError, match="must align"):
        TemporalCNNTrunk(channels=(8, 8), dilations=(1,))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_every_architecture_builds_and_runs(name):
    model = build_model(name)
    out = model(_x(2, 3, 16))
    assert out.score.shape == (2,)
    assert torch.isfinite(out.score).all()


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_scores_stay_inside_the_declared_range(name):
    out = build_model(name)(_x(4, 3, 16))
    assert (out.score >= 0.05 - 1e-6).all() and (out.score <= 1.0 + 1e-6).all()


def test_named_variants_have_distinct_sizes():
    """The bug this catches: a generic model.* config silently collapsing
    stgcn_dense and stgcn_reference into saqa_stgcn."""
    sizes = {
        name: sum(p.numel() for p in build_model(name).parameters())
        for name in ("saqa_stgcn", "saqa_stgcn_tiny", "stgcn_dense", "stgcn_reference",
                     "stgcn_reference_large")
    }
    assert len(set(sizes.values())) == len(sizes), sizes
    assert sizes["saqa_stgcn"] < sizes["stgcn_dense"] < sizes["stgcn_reference"]
    assert sizes["stgcn_reference"] < sizes["stgcn_reference_large"]


def test_default_model_is_in_the_target_parameter_band():
    n = sum(p.numel() for p in build_model("saqa_stgcn").parameters())
    assert 5e4 <= n <= 5e5, n


def test_unknown_architecture_is_rejected():
    with pytest.raises(KeyError, match="Unknown architecture"):
        build_trunk("transformer")


def test_unknown_head_is_rejected():
    with pytest.raises(ValueError, match="Unknown head"):
        build_model("saqa_stgcn", head="softmax")


def test_unknown_uncertainty_is_rejected():
    with pytest.raises(ValueError, match="Unknown uncertainty"):
        build_model("saqa_stgcn", uncertainty="dropout")


def test_model_is_deterministic_in_eval_mode():
    model = build_model("saqa_stgcn")
    model.eval()
    x = _x(2, 3, 16)
    with torch.no_grad():
        torch.testing.assert_close(model(x).score, model(x).score)


def test_loss_backpropagates_to_every_parameter():
    model = build_model("saqa_stgcn")
    loss, parts = model.loss(_x(4, 3, 16), torch.rand(4) * 0.9 + 0.05)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    assert set(parts) >= {"score", "uncertainty", "total"}


def test_quality_model_requires_an_uncertainty_head_for_intervals():
    model = QualityModel(build_trunk("frame_average"), uncertainty="none")
    with pytest.raises(RuntimeError, match="without an uncertainty head"):
        model(_x(2, 3, 16)).interval()
