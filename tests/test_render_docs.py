"""The documentation-injection mechanism.

If this breaks silently, the numbers in README.md stop matching the CSVs and the
repository's central promise -- every number traceable to a committed artefact --
is void. So the marker parser, the staleness check and the real documents are all
tested.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "render_docs", ROOT / "scripts" / "render_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_docs"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rd():
    return _load()


def test_marker_pattern_matches_a_block(rd):
    text = "before\n<!-- table:method -->\nold\n<!-- /table -->\nafter"
    assert rd.PATTERN.search(text) is not None


def test_render_replaces_the_body(rd, tmp_path, monkeypatch):
    from saqa import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    text = "<!-- table:method -->\nSTALE\n<!-- /table -->"
    out, changed = rd.render(text)
    assert "STALE" not in out
    assert "_not measured_" in out
    assert changed == ["method"]


def test_render_reports_no_change_when_already_current(rd, tmp_path, monkeypatch):
    from saqa import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    text = "<!-- table:method -->\n_not measured_\n<!-- /table -->"
    out, changed = rd.render(text)
    assert changed == []
    assert out == text


def test_render_fills_a_real_table(rd, tmp_path, monkeypatch):
    from saqa import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    pd.DataFrame({
        "method": ["a"], "family": ["neural"], "spearman": [0.5],
        "relative_l2": [0.2], "mae": [0.1], "params": [100.0],
    }).to_csv(tmp_path / "method_comparison.csv", index=False, lineterminator="\n")
    out, changed = rd.render("<!-- table:method -->\n\n<!-- /table -->")
    assert "| method |" in out and "0.5" in out
    assert changed == ["method"]


def test_unknown_table_name_is_rejected(rd):
    """A typo in a marker must fail loudly rather than leave a stale table."""
    with pytest.raises(KeyError, match="Unknown table"):
        rd.render("<!-- table:nonexistent -->\n\n<!-- /table -->")


def test_multiple_markers_are_all_filled(rd, tmp_path, monkeypatch):
    from saqa import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    text = ("<!-- table:method -->\nA\n<!-- /table -->\n"
            "<!-- table:seeds -->\nB\n<!-- /table -->")
    out, changed = rd.render(text)
    assert "A" not in out and "B" not in out
    assert set(changed) == {"method", "seeds"}


def test_text_outside_markers_is_untouched(rd, tmp_path, monkeypatch):
    from saqa import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    text = "KEEP ME\n<!-- table:method -->\nx\n<!-- /table -->\nKEEP ME TOO"
    out, _ = rd.render(text)
    assert out.startswith("KEEP ME\n") and out.endswith("KEEP ME TOO")


def test_every_marker_in_the_real_docs_names_a_known_table(rd):
    """Catches a marker typo in README.md or docs/ before it ships."""
    from saqa.report import ALL

    for rel in rd.TARGETS:
        path = ROOT / rel
        if not path.exists():
            continue
        for _, name, _, _ in rd.PATTERN.findall(path.read_text(encoding="utf-8")):
            assert name in ALL, f"{rel} references unknown table {name!r}"


def test_targets_point_at_files_that_exist_or_are_optional(rd):
    assert "README.md" in rd.TARGETS
    assert "docs/RESULTS.md" in rd.TARGETS
