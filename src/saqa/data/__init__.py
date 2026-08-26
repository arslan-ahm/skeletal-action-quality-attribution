"""Procedural skeleton-motion data: topology, kinematics, actions, defects, splits."""

from .actions import ACTION_CLASSES, ACTION_SPECS, DOF, DOF_NAMES
from .dataset import (
    SPLIT_MODES,
    SequenceDataset,
    build_dataset,
    label_stats,
    paired_degradation_sequences,
    split_indices,
)
from .degradations import (
    DEGRADATION_KINDS,
    KIND_WEIGHTS,
    Degradation,
    quality_from_degradations,
)
from .generator import (
    GeneratorConfig,
    Sample,
    attribution_ground_truth,
    generate_sample,
    reference_sequence,
)
from .kinematics import canonicalise, forward_kinematics, joint_angle
from .skeleton import (
    JOINT_GROUPS,
    JOINT_NAMES,
    LEFT_RIGHT_PAIRS,
    NUM_JOINTS,
    PARENTS,
    adjacency,
    bones,
)

__all__ = [
    "ACTION_CLASSES", "ACTION_SPECS", "DOF", "DOF_NAMES",
    "SPLIT_MODES", "SequenceDataset", "build_dataset", "label_stats",
    "paired_degradation_sequences", "split_indices",
    "DEGRADATION_KINDS", "KIND_WEIGHTS", "Degradation", "quality_from_degradations",
    "GeneratorConfig", "Sample", "attribution_ground_truth", "generate_sample",
    "reference_sequence",
    "canonicalise", "forward_kinematics", "joint_angle",
    "JOINT_GROUPS", "JOINT_NAMES", "LEFT_RIGHT_PAIRS", "NUM_JOINTS", "PARENTS",
    "adjacency", "bones",
]
