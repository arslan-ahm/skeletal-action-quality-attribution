"""Topology and adjacency: the graph the whole model rests on."""

from __future__ import annotations

import numpy as np
import pytest

from saqa.data.skeleton import (
    BONE_LENGTHS,
    JOINT_GROUPS,
    JOINT_NAMES,
    LEFT_RIGHT_PAIRS,
    NUM_JOINTS,
    PARENTS,
    adjacency,
    bone_lengths,
    bones,
    hop_distance_to_root,
    joint_index,
)


def test_parents_are_topologically_ordered():
    """Every parent has a lower index. Forward kinematics depends on this: it
    runs as a single left-to-right pass and would silently read uninitialised
    positions otherwise."""
    for j, p in enumerate(PARENTS):
        assert p < j, f"joint {j} ({JOINT_NAMES[j]}) has parent {p} >= itself"


def test_exactly_one_root():
    assert sum(1 for p in PARENTS if p < 0) == 1
    assert PARENTS[0] == -1


def test_lengths_match_joint_count():
    assert len(PARENTS) == NUM_JOINTS
    assert len(JOINT_NAMES) == NUM_JOINTS
    assert len(BONE_LENGTHS) == NUM_JOINTS


def test_root_has_no_bone():
    assert BONE_LENGTHS[0] == 0.0


def test_all_non_root_bones_positive():
    assert all(length > 0 for length in BONE_LENGTHS[1:])


def test_joint_names_unique():
    assert len(set(JOINT_NAMES)) == NUM_JOINTS


def test_joint_index_round_trips():
    for i, name in enumerate(JOINT_NAMES):
        assert joint_index(name) == i


def test_joint_index_rejects_unknown():
    with pytest.raises(KeyError):
        joint_index("tail")


def test_bones_excludes_root():
    edges = bones()
    assert len(edges) == NUM_JOINTS - 1
    assert all(p >= 0 for p, _ in edges)


def test_tree_is_connected():
    """Every joint reaches the root, so the graph is a tree and not a forest."""
    for j in range(NUM_JOINTS):
        seen, node = set(), j
        while node > 0:
            assert node not in seen, "cycle in the kinematic tree"
            seen.add(node)
            node = PARENTS[node]
        assert node == 0


def test_hop_distance_root_is_zero():
    assert hop_distance_to_root()[0] == 0


def test_hop_distance_increases_down_the_chain():
    dist = hop_distance_to_root()
    for j in range(1, NUM_JOINTS):
        assert dist[j] == dist[PARENTS[j]] + 1


def test_left_right_pairs_are_symmetric_names():
    for left, right in LEFT_RIGHT_PAIRS:
        assert JOINT_NAMES[left].startswith("l_")
        assert JOINT_NAMES[right].startswith("r_")
        assert JOINT_NAMES[left][2:] == JOINT_NAMES[right][2:]


def test_left_right_pairs_have_equal_bone_lengths():
    for left, right in LEFT_RIGHT_PAIRS:
        assert BONE_LENGTHS[left] == pytest.approx(BONE_LENGTHS[right])


def test_joint_groups_partition_the_skeleton():
    covered = sorted(j for group in JOINT_GROUPS.values() for j in group)
    assert covered == list(range(NUM_JOINTS)), "groups must tile the skeleton exactly"


def test_adjacency_spatial_has_three_partitions():
    assert adjacency("spatial").shape == (3, NUM_JOINTS, NUM_JOINTS)


def test_adjacency_uniform_and_identity_have_one():
    assert adjacency("uniform").shape[0] == 1
    assert adjacency("identity").shape[0] == 1


def test_identity_adjacency_is_the_identity_matrix():
    np.testing.assert_allclose(adjacency("identity")[0], np.eye(NUM_JOINTS))


def test_adjacency_rows_are_normalised():
    """Each non-empty row sums to 1, so a partition averages its neighbours."""
    a = adjacency("spatial")
    sums = a.sum(axis=-1)
    nonzero = sums > 0
    np.testing.assert_allclose(sums[nonzero], 1.0, atol=1e-12)


def test_zero_rows_stay_zero_not_nan():
    """The centripetal partition has no incoming edge for leaf joints. Division
    by a zero degree must leave the row zero rather than produce NaN."""
    a = adjacency("spatial")
    assert np.isfinite(a).all()


def test_centripetal_and_centrifugal_are_transposes():
    """Every bone contributes one entry to each direction."""
    a = adjacency("spatial", self_loops=True)
    inward, outward = a[1], a[2]
    assert (inward > 0).sum() == (outward > 0).sum() == NUM_JOINTS - 1
    assert ((inward > 0) & (outward.T > 0)).sum() == NUM_JOINTS - 1


def test_centripetal_points_toward_the_root():
    """A[1][i, j] > 0 must mean j is further from the pelvis than i."""
    a = adjacency("spatial")
    dist = hop_distance_to_root()
    for i, j in zip(*np.nonzero(a[1]), strict=True):
        assert dist[j] > dist[i]


def test_centrifugal_points_away_from_the_root():
    a = adjacency("spatial")
    dist = hop_distance_to_root()
    for i, j in zip(*np.nonzero(a[2]), strict=True):
        assert dist[j] < dist[i]


def test_uniform_merges_both_directions():
    a = adjacency("uniform")[0]
    spatial = adjacency("spatial", self_loops=False)
    merged = (spatial.sum(axis=0) > 0) | np.eye(NUM_JOINTS, dtype=bool)
    assert ((a > 0) == merged).all()


def test_self_loops_flag_removes_the_identity_partition():
    assert adjacency("spatial", self_loops=False).shape[0] == 2


def test_adjacency_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="Unknown partition"):
        adjacency("magic")


def test_bone_lengths_scaling():
    scale = np.full(NUM_JOINTS, 2.0)
    np.testing.assert_allclose(bone_lengths(scale), np.array(BONE_LENGTHS) * 2.0)


def test_bone_lengths_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        bone_lengths(np.ones(3))


def test_adjacency_is_deterministic():
    np.testing.assert_array_equal(adjacency("spatial"), adjacency("spatial"))
