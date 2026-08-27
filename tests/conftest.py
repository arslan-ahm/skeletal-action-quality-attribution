"""Shared fixtures. Everything here is deliberately tiny so the fast suite stays
under a couple of minutes on two CPU threads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch.set_num_threads(1)

from saqa.config import load_config  # noqa: E402
from saqa.data.dataset import build_dataset  # noqa: E402
from saqa.data.generator import GeneratorConfig  # noqa: E402


@pytest.fixture(scope="session")
def tiny_cfg() -> GeneratorConfig:
    """A generator config small enough to build many datasets from."""
    return GeneratorConfig(num_frames=24, num_subjects=6)


@pytest.fixture(scope="session")
def tiny_data(tiny_cfg):
    """60 sequences at 24 frames. Session-scoped: regenerating costs ~0.7 s."""
    return build_dataset(60, tiny_cfg)


@pytest.fixture(scope="session")
def clean_cfg() -> GeneratorConfig:
    """Noise-free generator, for tests that need exact counterfactuals."""
    return GeneratorConfig(
        num_frames=24, num_subjects=4, noise_std=0.0, jitter_std=0.0, dropout_prob=0.0
    )


@pytest.fixture
def smoke_config():
    """A :class:`~saqa.config.Config` sized for a two-epoch end-to-end run."""
    return load_config(
        None,
        [
            "data.num_sequences=60",
            "data.num_frames=24",
            "data.num_subjects=6",
            "data.batch_size=8",
            "model.channels=8,16",
            "model.strides=2,1",
            "optim.epochs=2",
            "eval.bootstrap=50",
            "eval.ig_steps=4",
            "eval.attribution_limit=8",
            "eval.monotonicity_ladders=3",
            "run.name=pytest",
            "run.out_dir=results/runs_test",
        ],
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
