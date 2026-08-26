"""Metrics: quality regression, uncertainty/abstention, monotonicity, attribution fidelity."""

from .attribution import (
    adaptive_k,
    distribution_scores,
    fidelity_summary,
    per_sequence_fidelity,
    random_attribution,
    temporal_localisation_error,
    topk_scores,
)
from .monotonicity import ladder_violations, monotonicity_report, rank_inconsistency
from .regression import (
    kendall_tau,
    per_group_summary,
    per_item_absolute_error,
    regression_summary,
    relative_l2,
    spearman,
)
from .stats import (
    Comparison,
    Interval,
    bootstrap_ci,
    bootstrap_metric_difference,
    compare,
    holm_bonferroni,
    noise_scale,
    paired_bootstrap_difference,
    verdict,
)
from .uncertainty import (
    aurc,
    coverage,
    error_detection_auroc,
    excess_aurc,
    interval_summary,
    mean_width,
    risk_coverage_curve,
)

__all__ = [
    "adaptive_k", "distribution_scores", "fidelity_summary", "per_sequence_fidelity",
    "random_attribution", "temporal_localisation_error", "topk_scores",
    "ladder_violations", "monotonicity_report", "rank_inconsistency",
    "kendall_tau", "per_group_summary", "per_item_absolute_error",
    "regression_summary", "relative_l2", "spearman",
    "Comparison", "Interval", "bootstrap_ci", "bootstrap_metric_difference", "compare",
    "holm_bonferroni", "noise_scale", "paired_bootstrap_difference", "verdict",
    "aurc", "coverage", "error_detection_auroc", "excess_aurc", "interval_summary",
    "mean_width", "risk_coverage_curve",
]
