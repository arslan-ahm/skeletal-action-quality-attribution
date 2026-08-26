"""Baselines: the reference DTW approach, and handcrafted kinematics + boosting."""

from .dtw import (
    DISTANCES,
    dtw_distance,
    dtw_similarity,
    frame_cost_matrix,
    framewise_distance,
    per_joint_dtw_deviation,
)
from .fitted import DTWBaseline, KinematicGBRBaseline, conformal_halfwidth
from .kinematic import ANGLE_TRIPLES, feature_matrix, kinematic_features

__all__ = [
    "DISTANCES", "dtw_distance", "dtw_similarity", "frame_cost_matrix",
    "framewise_distance", "per_joint_dtw_deviation",
    "DTWBaseline", "KinematicGBRBaseline", "conformal_halfwidth",
    "ANGLE_TRIPLES", "feature_matrix", "kinematic_features",
]
