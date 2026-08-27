"""The console entry point declared in pyproject.toml.

``pyproject.toml`` declares ``saqa = "saqa.cli:main"``. If that target is missing
or broken, ``pip install`` produces a script that fails on first use -- a defect
no other test would catch.
"""

from __future__ import annotations

import pytest

from saqa.cli import build_parser, main
from saqa.report import ALL


def test_parser_builds():
    assert build_parser() is not None


def test_every_subcommand_is_registered():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "no subparsers registered"
    assert set(actions[0].choices) == {"data", "train", "bench", "compare", "tables"}


def test_missing_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main([])


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_tables_command_runs_without_artifacts(capsys, tmp_path, monkeypatch):
    from saqa import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    assert main(["tables", "--only", "method"]) == 0
    assert "_not measured_" in capsys.readouterr().out


def test_tables_command_prints_every_section(capsys, tmp_path, monkeypatch):
    from saqa import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    main(["tables"])
    out = capsys.readouterr().out
    for name in ALL:
        assert f"### {name}" in out


def test_tables_rejects_an_unknown_name():
    with pytest.raises(SystemExit):
        main(["tables", "--only", "nonexistent"])


def test_data_command_reports_all_three_split_regimes(capsys):
    code = main(["data", "--set", "data.num_sequences=60", "data.num_frames=24",
                 "data.num_subjects=6"])
    out = capsys.readouterr().out
    assert code == 0
    for mode in ("random", "subject", "combination"):
        assert f"split[{mode}" in out
    assert "actions:" in out


def test_train_command_runs_end_to_end(capsys, tmp_path):
    code = main(["train", "--set", "data.num_sequences=60", "data.num_frames=24",
                 "data.num_subjects=6", "data.batch_size=8", "optim.epochs=1",
                 "model.channels=8,16", "model.strides=2,1",
                 f"run.out_dir={tmp_path.as_posix()}", "run.name=cli"])
    assert code == 0
    assert "spearman" in capsys.readouterr().out
    assert (tmp_path / "cli" / "summary.json").exists()


def test_config_override_reaches_the_command():
    with pytest.raises(KeyError, match="Unknown config key"):
        main(["data", "--set", "data.nonexistent=1"])
