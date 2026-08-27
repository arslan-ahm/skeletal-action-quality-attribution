"""Render the committed CSVs as Markdown tables.

The documentation quotes numbers. Hand-typing them is how three wrong tables get
shipped, so every table in ``README.md`` and ``docs/RESULTS.md`` is produced by
this module from the CSV it belongs to. If a CSV is missing, the renderer says
``not measured`` rather than inventing a row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("results/tables")


def _fmt(value, digits: int = 4) -> str:
    """Format one cell, keeping ``NaN`` visible as ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "n/a"
    if abs(f) >= 1e5 or (abs(f) < 1e-3 and f != 0):
        return f"{f:.2e}"
    return f"{f:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{f:.0f}"


def to_markdown(df: pd.DataFrame, columns: list[str] | None = None,
                digits: int = 4, bold_max: str | None = None,
                bold_min: str | None = None) -> str:
    """Render a frame as a GitHub Markdown table.

    Args:
        df: The frame.
        columns: Column subset, in order. Missing columns are skipped silently,
            so a renderer keeps working when an optional metric is absent.
        digits: Decimal places.
        bold_max: Column whose maximum should be bolded.
        bold_min: Column whose minimum should be bolded.
    """
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    if not cols:
        return "_not measured_"
    sub = df[cols]
    best: dict[str, int] = {}
    for col, fn in ((bold_max, "idxmax"), (bold_min, "idxmin")):
        if col and col in sub.columns and sub[col].notna().any():
            best[col] = int(getattr(sub[col], fn)())

    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for idx, row in sub.iterrows():
        cells = []
        for col in cols:
            text = _fmt(row[col], digits)
            if best.get(col) == idx:
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read(name: str) -> pd.DataFrame | None:
    """Read a committed table, or ``None`` when it has not been produced."""
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else None


def method_table(digits: int = 4) -> str:
    df = read("method_comparison.csv")
    if df is None:
        return "_not measured_"
    df = df.sort_values("spearman", ascending=False)
    return to_markdown(
        df,
        ["method", "family", "spearman", "kendall_tau", "relative_l2", "mae",
         "coverage", "mean_width", "error_auroc", "params"],
        digits, bold_max="spearman", bold_min="relative_l2",
    )


def efficiency_table(digits: int = 3) -> str:
    df = read("efficiency.csv")
    if df is None:
        return "_not measured_"
    out = df.copy()
    out["params_M"] = out["params"] / 1e6
    out["MMACs"] = out["macs"] / 1e6
    return to_markdown(
        out,
        ["architecture", "params_M", "MMACs", "latency_bs1_ms", "iqr_bs1_ms",
         "latency_bs8_ms", "macs_per_ms_bs1", "params_reduction", "macs_reduction",
         "latency_reduction"],
        digits,
    )


def attribution_table(model: str = "saqa_stgcn", digits: int = 4) -> str:
    df = read("attribution_fidelity.csv")
    if df is None:
        return "_not measured_"
    sub = df[df["model"] == model]
    if sub.empty:
        sub = df
    return to_markdown(
        sub,
        ["attribution", "joint_precision", "joint_recall", "joint_iou",
         "joint_rank_corr", "joint_top1_hit", "joint_overlap",
         "frame_localisation_error", "n_joint_iou"],
        digits,
    )


def monotonicity_table(digits: int = 4) -> str:
    df = read("monotonicity.csv")
    if df is None:
        return "_not measured_"
    return to_markdown(
        df,
        ["method", "violation_rate", "adjacent_violation_rate",
         "ladders_with_any_violation", "max_increase", "rank_inconsistency"],
        digits, bold_min="violation_rate",
    )


def uncertainty_table(digits: int = 4) -> str:
    df = read("uncertainty.csv")
    if df is None:
        return "_not measured_"
    return to_markdown(
        df,
        ["method", "coverage", "mean_width", "crossing_rate",
         "width_ratio_wrong_right", "aurc", "aurc_oracle", "e_aurc", "error_auroc"],
        digits,
    )


def seed_table(digits: int = 4) -> str:
    df = read("seed_variance.csv")
    if df is None:
        return "_not measured_"
    return to_markdown(df, ["metric", "mean", "sd", "min", "max", "range",
                            "noise_scale", "n_runs"], digits)


def ablation_table(digits: int = 4) -> str:
    df = read("ablation_components.csv")
    if df is None:
        return "_not measured_"
    return to_markdown(
        df,
        ["variant", "change", "spearman", "spearman_delta",
         "spearman_ratio_to_noise", "spearman_verdict", "params"],
        digits,
    )


def split_table(digits: int = 4) -> str:
    df = read("split_comparison.csv")
    if df is None:
        return "_not measured_"
    return to_markdown(df, ["split", "spearman", "kendall_tau", "relative_l2",
                            "mae", "coverage", "n_train", "n_test"], digits)


def statistical_table(family: str = "spearman", digits: int = 4) -> str:
    df = read("statistical_tests.csv")
    if df is None:
        return "_not measured_"
    sub = df[df["family"] == family]
    return to_markdown(
        sub,
        ["name_a", "mean_a", "mean_b", "difference", "ci_lower", "ci_upper",
         "p_value", "p_adjusted", "effect_size", "significant"],
        digits,
    )


def dtw_sweep_table(digits: int = 4) -> str:
    df = read("dtw_sweep.csv")
    if df is None:
        return "_not measured_"
    return to_markdown(df.sort_values("val_spearman", ascending=False),
                       ["config", "val_spearman", "test_spearman",
                        "test_relative_l2", "test_mae"], digits,
                       bold_max="val_spearman")


def data_efficiency_table(digits: int = 4) -> str:
    df = read("data_efficiency.csv")
    if df is None:
        return "_not measured_"
    pivot = df.pivot(index="n_train", columns="method", values="spearman").reset_index()
    return to_markdown(pivot, list(pivot.columns), digits)


def cost_curve_table(digits: int = 3) -> str:
    df = read("cost_vs_length.csv")
    if df is None:
        return "_not measured_"
    return to_markdown(df, ["num_frames", "dtw_total_ms", "model_ms", "speedup"], digits)


def verdict_table(metric: str = "spearman", digits: int = 4) -> str:
    """Every method's gap to the reference, as a multiple of the noise scale."""
    df = read("method_verdicts.csv")
    if df is None:
        return "_not measured_"
    sub = df[df["metric"] == metric].sort_values("delta", ascending=False)
    return to_markdown(
        sub, ["method", "value", "reference_value", "delta", "noise_scale",
              "ratio_to_noise", "verdict"], digits,
    )


def per_action_table(digits: int = 4) -> str:
    df = read("per_action.csv")
    if df is None:
        return "_not measured_"
    cols = ["method"] + [c for c in df.columns if not c.endswith("__n") and c != "method"]
    return to_markdown(df, cols, digits)


ALL = {
    "method": method_table,
    "efficiency": efficiency_table,
    "attribution": attribution_table,
    "monotonicity": monotonicity_table,
    "uncertainty": uncertainty_table,
    "seeds": seed_table,
    "ablation": ablation_table,
    "splits": split_table,
    "statistics": statistical_table,
    "dtw_sweep": dtw_sweep_table,
    "data_efficiency": data_efficiency_table,
    "cost_curve": cost_curve_table,
    "per_action": per_action_table,
    "verdicts": verdict_table,
}


def render_all() -> str:
    """Every table, one after another, with its section name."""
    parts = []
    for name, fn in ALL.items():
        parts.append(f"### {name}\n\n{fn()}\n")
    return "\n".join(parts)
