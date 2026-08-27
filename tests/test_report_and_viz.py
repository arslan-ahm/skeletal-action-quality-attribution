"""Table rendering and figure drawing.

These are the two places where a number can silently stop matching its artefact,
so both are tested: the renderer must show ``NaN`` as ``n/a`` rather than as a
plausible-looking value, and it must say ``not measured`` rather than invent a
row when a CSV is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saqa import report, viz


@pytest.fixture
def tables(tmp_path, monkeypatch):
    """Point the renderer at a temporary tables directory."""
    monkeypatch.setattr(report, "TABLES", tmp_path)
    monkeypatch.setattr(viz, "TABLES", tmp_path)
    return tmp_path


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def test_nan_renders_as_na_not_a_number():
    assert report._fmt(float("nan")) == "n/a"
    assert report._fmt(float("inf")) == "n/a"
    assert report._fmt(None) == "n/a"


def test_booleans_render_readably():
    assert report._fmt(True) == "yes"
    assert report._fmt(np.bool_(False)) == "no"


def test_large_and_tiny_values_use_scientific_notation():
    assert "e" in report._fmt(1.2e6)
    assert "e" in report._fmt(3.1e-5)


def test_ordinary_values_keep_their_digits():
    assert report._fmt(0.12345, 3) == "0.123"
    assert report._fmt(0.5, 4) == "0.5"


def test_strings_pass_through():
    assert report._fmt("inside noise") == "inside noise"


# --------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------


def test_markdown_has_a_header_and_a_rule():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    out = report.to_markdown(df).split("\n")
    assert out[0] == "| a | b |"
    assert out[1] == "|---|---|"
    assert len(out) == 4


def test_markdown_selects_and_orders_columns():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert report.to_markdown(df, ["c", "a"]).startswith("| c | a |")


def test_markdown_skips_absent_columns_silently():
    """An optional metric that a run did not produce must not break the table."""
    df = pd.DataFrame({"a": [1]})
    assert report.to_markdown(df, ["a", "nonexistent"]).startswith("| a |")


def test_markdown_with_no_usable_columns_says_not_measured():
    assert report.to_markdown(pd.DataFrame({"a": [1]}), ["z"]) == "_not measured_"


def test_bold_max_marks_the_best_row():
    df = pd.DataFrame({"m": [0.1, 0.9, 0.5]})
    out = report.to_markdown(df, ["m"], bold_max="m")
    assert "**0.9**" in out
    assert out.count("**") == 2


def test_bold_min_marks_the_lowest_row():
    df = pd.DataFrame({"m": [0.1, 0.9]})
    assert "**0.1**" in report.to_markdown(df, ["m"], bold_min="m")


def test_bold_ignores_an_all_nan_column():
    df = pd.DataFrame({"m": [float("nan"), float("nan")]})
    assert "**" not in report.to_markdown(df, ["m"], bold_max="m")


# --------------------------------------------------------------------------
# missing artefacts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(report.ALL))
def test_every_renderer_reports_not_measured_when_the_csv_is_absent(tables, name):
    """A missing CSV must never produce a fabricated table."""
    assert report.ALL[name]() == "_not measured_"


def test_read_returns_none_for_a_missing_table(tables):
    assert report.read("nope.csv") is None


def test_render_all_names_every_section(tables):
    out = report.render_all()
    for name in report.ALL:
        assert f"### {name}" in out


# --------------------------------------------------------------------------
# renderers with data
# --------------------------------------------------------------------------


def _write(tables, name, frame):
    frame.to_csv(tables / name, index=False, lineterminator="\n")


def test_method_table_sorts_by_spearman(tables):
    _write(tables, "method_comparison.csv", pd.DataFrame({
        "method": ["a", "b"], "family": ["neural", "baseline"],
        "spearman": [0.2, 0.8], "relative_l2": [0.5, 0.3], "mae": [0.1, 0.2],
        "params": [10.0, float("nan")],
    }))
    rows = report.method_table().split("\n")
    assert rows[2].startswith("| b |"), "rows must be sorted by Spearman, best first"
    assert "n/a" in rows[2], "a baseline has no parameter count; it must not read 0"
    assert "**0.8**" in rows[2]


def test_efficiency_table_converts_units(tables):
    _write(tables, "efficiency.csv", pd.DataFrame({
        "architecture": ["x"], "params": [2.5e6], "macs": [1e8],
        "latency_bs1_ms": [10.0], "iqr_bs1_ms": [1.0],
    }))
    out = report.efficiency_table()
    assert "params_M" in out and "2.5" in out
    assert "MMACs" in out and "100" in out


def test_attribution_table_filters_to_one_model(tables):
    _write(tables, "attribution_fidelity.csv", pd.DataFrame({
        "model": ["saqa_stgcn", "tcn"], "attribution": ["ig", "ig"],
        "joint_iou": [0.5, 0.9],
    }))
    out = report.attribution_table("saqa_stgcn")
    assert "0.5" in out and "0.9" not in out


def test_attribution_table_falls_back_when_the_model_is_absent(tables):
    _write(tables, "attribution_fidelity.csv", pd.DataFrame({
        "model": ["tcn"], "attribution": ["ig"], "joint_iou": [0.9],
    }))
    assert "0.9" in report.attribution_table("saqa_stgcn")


def test_statistical_table_filters_by_family(tables):
    _write(tables, "statistical_tests.csv", pd.DataFrame({
        "family": ["spearman", "abs_error"], "name_a": ["a", "b"],
        "mean_a": [1.0, 2.0], "mean_b": [0.0, 0.0], "difference": [1.0, 2.0],
        "ci_lower": [0.5, 1.0], "ci_upper": [1.5, 3.0],
        "p_value": [0.01, 0.02], "p_adjusted": [0.02, 0.04],
        "effect_size": [1.0, 1.0], "significant": [True, False],
    }))
    out = report.statistical_table("spearman")
    assert "| a |" in out and "| b |" not in out
    assert "yes" in out


def test_data_efficiency_table_pivots(tables):
    _write(tables, "data_efficiency.csv", pd.DataFrame({
        "method": ["m1", "m2", "m1", "m2"], "n_train": [10, 10, 20, 20],
        "spearman": [0.1, 0.2, 0.3, 0.4],
    }))
    out = report.data_efficiency_table()
    assert "m1" in out and "m2" in out and "n_train" in out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def test_figures_return_none_without_data(tables, tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path / "figs")
    for fn in (viz.plot_method_comparison, viz.plot_cost_curve, viz.plot_efficiency,
               viz.plot_data_efficiency, viz.plot_attribution_fidelity):
        assert fn() is None


def test_all_figures_is_empty_without_data(tables, tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path / "figs")
    assert viz.all_figures(run_dir=tmp_path / "no_such_run") == []


def test_method_comparison_figure_is_written(tables, tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path / "figs")
    _write(tables, "method_comparison.csv", pd.DataFrame({
        "method": ["a", "b"], "family": ["neural", "baseline"],
        "spearman": [0.2, 0.8], "relative_l2": [0.5, 0.3],
    }))
    path = viz.plot_method_comparison("t.png")
    assert path is not None and path.exists() and path.stat().st_size > 0


def test_cost_curve_figure_is_written(tables, tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path / "figs")
    _write(tables, "cost_vs_length.csv", pd.DataFrame({
        "num_frames": [16, 32], "dtw_total_ms": [1.0, 4.0], "model_ms": [1.0, 2.0],
        "dtw_exponent": [2.0, 2.0], "model_exponent": [1.0, 1.0],
    }))
    assert viz.plot_cost_curve("c.png").exists()


def test_skeleton_figure_is_written(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path / "figs")
    path = viz.plot_skeleton_frames(tiny_data.samples[0], frames=(0, 5), name="s.png")
    assert path.exists() and path.stat().st_size > 0


def test_skeleton_figure_clips_out_of_range_frames(tiny_data, tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path / "figs")
    assert viz.plot_skeleton_frames(tiny_data.samples[0], frames=(0, 10_000),
                                    name="s2.png").exists()


def test_attribution_map_figure_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path / "figs")
    rng = np.random.default_rng(0)
    path = viz.plot_attribution_map(rng.normal(size=(12, 17)), rng.random(17),
                                    rng.random(12), name="m.png")
    assert path.exists()


def test_risk_coverage_returns_none_without_a_width_column(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path / "figs")
    path = tmp_path / "per_item.csv"
    pd.DataFrame({"abs_error": [0.1, 0.2]}).to_csv(path, index=False)
    assert viz.plot_risk_coverage(str(path)) is None


def test_risk_coverage_is_written_when_widths_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "FIGURES", tmp_path / "figs")
    path = tmp_path / "per_item.csv"
    pd.DataFrame({"abs_error": [0.1, 0.2, 0.3], "width": [0.2, 0.4, 0.1]}).to_csv(
        path, index=False
    )
    assert viz.plot_risk_coverage(str(path), name="rc.png").exists()
