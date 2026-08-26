"""Attribution methods and the machinery to score them against exact ground truth."""

from .methods import (
    completeness_error,
    edge_importance_attribution,
    input_gradient,
    integrated_gradients,
    occlusion,
    temporal_baseline,
    to_frame_vector,
    to_joint_vector,
)

METHODS = ("integrated_gradients", "occlusion", "gradient")

__all__ = [
    "METHODS",
    "completeness_error", "edge_importance_attribution", "input_gradient",
    "integrated_gradients", "occlusion", "temporal_baseline",
    "to_frame_vector", "to_joint_vector",
]
