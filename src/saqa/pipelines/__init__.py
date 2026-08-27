"""Experiment orchestration. Scripts are thin wrappers over these functions."""

from .analysis import (
    attribution_completeness,
    attribution_fidelity,
    compute_attribution,
    monotonicity_check,
    ordinal_consistency,
    per_sequence_attribution_scores,
)
from .core import (
    RunResult,
    Splits,
    evaluate_groups,
    evaluate_predictions,
    generator_config,
    make_splits,
    run_single,
)
from .experiments import (
    NEURAL_ARMS,
    apply_noise_verdicts,
    fit_baselines,
    method_comparison,
    seed_study,
    statistical_tests,
)
from .sweeps import (
    ABLATIONS,
    data_efficiency,
    dtw_cost_curve,
    efficiency_benchmark,
    ensure_dirs,
    run_ablations,
    split_comparison,
)

__all__ = [
    "attribution_completeness", "attribution_fidelity", "compute_attribution",
    "monotonicity_check", "ordinal_consistency", "per_sequence_attribution_scores",
    "RunResult", "Splits", "evaluate_groups", "evaluate_predictions",
    "generator_config", "make_splits", "run_single",
    "NEURAL_ARMS", "apply_noise_verdicts", "fit_baselines", "method_comparison",
    "seed_study", "statistical_tests",
    "ABLATIONS", "data_efficiency", "dtw_cost_curve", "efficiency_benchmark",
    "ensure_dirs", "run_ablations", "split_comparison",
]
