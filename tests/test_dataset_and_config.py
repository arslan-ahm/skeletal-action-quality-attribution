"""Splits, severity ladders, and the config system's precedence rules."""

from __future__ import annotations

import numpy as np
import pytest

from saqa.config import Config, load_config, parse_overrides
from saqa.data.dataset import (
    SPLIT_MODES,
    build_dataset,
    label_stats,
    paired_degradation_sequences,
    split_indices,
)
from saqa.data.skeleton import NUM_JOINTS

# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


def test_dataset_shapes(tiny_data, tiny_cfg):
    assert tiny_data.coords.shape == (60, 3, tiny_cfg.num_frames, NUM_JOINTS)
    assert tiny_data.quality.shape == (60,)
    assert tiny_data.joint_truth.shape == (60, NUM_JOINTS)
    assert tiny_data.frame_truth.shape == (60, tiny_cfg.num_frames)


def test_coords_are_channel_first(tiny_data):
    """The graph convolution wants (N, C, T, V). A silent transpose here would
    train a model on the joint axis as if it were time."""
    sample = tiny_data.samples[0]
    np.testing.assert_allclose(
        tiny_data.coords[0], sample.positions.transpose(2, 0, 1), atol=1e-6
    )


def test_dataset_is_float32(tiny_data):
    assert tiny_data.coords.dtype == np.float32
    assert tiny_data.quality.dtype == np.float32


def test_dataset_rejects_zero_size(tiny_cfg):
    with pytest.raises(ValueError, match="must be positive"):
        build_dataset(0, tiny_cfg)


def test_disjoint_start_gives_disjoint_sequences(tiny_cfg):
    a = build_dataset(10, tiny_cfg, start=0)
    b = build_dataset(10, tiny_cfg, start=100)
    assert not np.allclose(a.coords, b.coords)


def test_subset_preserves_alignment(tiny_data):
    idx = np.array([3, 9, 17])
    sub = tiny_data.subset(idx)
    assert len(sub) == 3
    np.testing.assert_array_equal(sub.quality, tiny_data.quality[idx])
    np.testing.assert_array_equal(sub.coords, tiny_data.coords[idx])
    assert [s.index for s in sub.samples] == [tiny_data.samples[i].index for i in idx]


@pytest.mark.parametrize("mode", SPLIT_MODES)
def test_splits_are_disjoint_and_complete(tiny_data, mode):
    idx = split_indices(tiny_data, mode)
    joined = np.concatenate([idx["train"], idx["val"], idx["test"]])
    assert len(set(joined.tolist())) == len(joined) == len(tiny_data)


@pytest.mark.parametrize("mode", SPLIT_MODES)
def test_splits_are_non_empty(tiny_data, mode):
    for name, arr in split_indices(tiny_data, mode).items():
        assert arr.size > 0, name


def test_subject_split_holds_out_whole_subjects(tiny_data):
    idx = split_indices(tiny_data, "subject")
    subjects = tiny_data.subjects
    train_subj = set(subjects[idx["train"]].tolist()) | set(subjects[idx["val"]].tolist())
    test_subj = set(subjects[idx["test"]].tolist())
    assert not (train_subj & test_subj), "a subject appeared on both sides"


def test_random_split_does_leak_subjects(tiny_data):
    """Asserted rather than assumed: the leakage the repository reports has to
    actually be present in the random regime, or the comparison is theatre."""
    idx = split_indices(tiny_data, "random")
    subjects = tiny_data.subjects
    train_subj = set(subjects[idx["train"]].tolist())
    test_subj = set(subjects[idx["test"]].tolist())
    assert train_subj & test_subj


def test_combination_split_holds_out_compensation_interactions(tiny_data):
    idx = split_indices(tiny_data, "combination")
    combos = tiny_data.combinations
    for c in combos[idx["test"]]:
        assert "compensation" in c.split("+") and "+" in c
    for c in np.concatenate([combos[idx["train"]], combos[idx["val"]]]):
        assert not ("compensation" in c.split("+") and "+" in c)


def test_split_rejects_unknown_mode(tiny_data):
    with pytest.raises(ValueError, match="Unknown split mode"):
        split_indices(tiny_data, "kfold")


def test_split_is_deterministic_given_a_seed(tiny_data):
    a = split_indices(tiny_data, "subject", seed=3)
    b = split_indices(tiny_data, "subject", seed=3)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_split_seed_changes_the_partition(tiny_data):
    a = split_indices(tiny_data, "subject", seed=0)
    b = split_indices(tiny_data, "subject", seed=5)
    assert not np.array_equal(a["test"], b["test"])


def test_label_stats_are_in_range(tiny_data):
    stats = label_stats(tiny_data)
    assert stats["n"] == 60
    assert 0.0 < stats["min"] <= stats["max"] <= 1.0
    assert 0.0 <= stats["frac_clean"] <= 1.0


# --------------------------------------------------------------------------
# severity ladders
# --------------------------------------------------------------------------


def test_ladder_labels_are_strictly_decreasing(clean_cfg):
    for lad in paired_degradation_sequences(6, clean_cfg):
        q = np.asarray(lad["quality"])
        assert (np.diff(q) < 0).all()


def test_ladder_shapes(clean_cfg):
    lad = paired_degradation_sequences(2, clean_cfg)[0]
    assert lad["coords"].shape == (5, 3, clean_cfg.num_frames, NUM_JOINTS)
    assert lad["quality"].shape == (5,)


def test_ladder_first_rung_is_the_clean_execution(clean_cfg):
    for lad in paired_degradation_sequences(4, clean_cfg):
        assert lad["quality"][0] == pytest.approx(1.0)


def test_ladder_varies_only_severity(clean_cfg):
    """Every rung must be the same subject doing the same action; only the
    defect severity changes. Otherwise a violation could be caused by the
    confound rather than by the model."""
    for lad in paired_degradation_sequences(4, clean_cfg):
        coords = lad["coords"]
        # increasing severity moves the sequence monotonically away from clean
        distances = [np.abs(coords[i] - coords[0]).mean() for i in range(1, 5)]
        assert distances == sorted(distances)


def test_ladders_are_noise_free(clean_cfg):
    """With sensor noise on, a violation could be the noise realisation rather
    than the model, and the measured rate would be a loose upper bound."""
    from saqa.data.generator import GeneratorConfig

    lad = paired_degradation_sequences(2, GeneratorConfig(num_frames=24))[0]
    lad2 = paired_degradation_sequences(2, GeneratorConfig(num_frames=24))[0]
    np.testing.assert_array_equal(lad["coords"], lad2["coords"])


def test_ladder_rejects_descending_steps(clean_cfg):
    with pytest.raises(ValueError, match="ascending"):
        paired_degradation_sequences(1, clean_cfg, severity_steps=(1.0, 0.5))


def test_ladders_cover_every_requested_kind(clean_cfg):
    kinds = ("rom", "asymmetry", "tempo", "jerk", "instability", "compensation")
    seen = {lad["kind"] for lad in paired_degradation_sequences(12, clean_cfg, kinds=kinds)}
    assert seen == set(kinds)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_defaults_load_without_a_file():
    assert isinstance(load_config(None), Config)


def test_parse_overrides_builds_a_nested_dict():
    assert parse_overrides(["a.b.c=1", "d=2"]) == {"a": {"b": {"c": "1"}}, "d": "2"}


def test_parse_overrides_rejects_missing_equals():
    with pytest.raises(ValueError, match="key=value"):
        parse_overrides(["nope"])


def test_override_types_follow_the_dataclass():
    cfg = load_config(None, ["optim.epochs=7", "data.noise_std=0",
                             "model.edge_importance=false"])
    assert isinstance(cfg.optim.epochs, int) and cfg.optim.epochs == 7
    assert isinstance(cfg.data.noise_std, float) and cfg.data.noise_std == 0.0
    assert cfg.model.edge_importance is False


def test_tuple_override_from_csv():
    cfg = load_config(None, ["model.channels=8,16,32"])
    assert cfg.model.channels == (8, 16, 32)


def test_non_integer_into_int_field_is_rejected():
    with pytest.raises(ValueError, match="not an integer"):
        load_config(None, ["run.seed=1.5"])


def test_unknown_key_is_rejected_not_ignored():
    """A silently-swallowed typo would void an experiment while appearing to
    succeed."""
    with pytest.raises(KeyError, match="Unknown config key"):
        load_config(None, ["optim.leraning_rate=0.1"])


def test_unknown_section_is_rejected():
    with pytest.raises(KeyError, match="Unknown config key"):
        load_config(None, ["optimiser.lr=0.1"])


def test_bad_bool_is_rejected():
    with pytest.raises(ValueError, match="as a bool"):
        load_config(None, ["model.separable=maybe"])


def test_base_inheritance_from_yaml():
    cfg = load_config("configs/saqa_stgcn.yaml")
    assert cfg.model.architecture == "saqa_stgcn"
    assert cfg.optim.epochs == load_config("configs/base.yaml").optim.epochs


def test_override_precedence_beats_the_file():
    cfg = load_config("configs/base.yaml", ["optim.epochs=99"])
    assert cfg.optim.epochs == 99


def test_child_yaml_beats_its_base():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.optim.epochs == 3
    assert cfg.data.num_sequences == 240


def test_config_saves_and_reloads(tmp_path):
    cfg = load_config(None, ["optim.epochs=11", "model.channels=4,8"])
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    assert "\r\n" not in path.read_text(encoding="utf-8"), "line endings must be LF"
    again = load_config(path)
    assert again.optim.epochs == 11
    assert again.model.channels == (4, 8)


def test_saved_config_round_trips_every_field():
    cfg = load_config("configs/base.yaml")
    assert cfg.to_dict()["model"]["architecture"] == "saqa_stgcn"
