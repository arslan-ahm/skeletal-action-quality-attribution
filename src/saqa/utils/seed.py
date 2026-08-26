"""Deterministic seeding for every entry point."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0, threads: int = 2) -> None:
    """Seed Python, NumPy and torch, and cap the thread pool.

    Determinism on CPU needs the thread count pinned as well as the seeds: some
    reduction kernels pick a different split by thread count, which changes the
    floating-point summation order and therefore the last few bits of a result.
    ``docs/REPRODUCIBILITY.md`` reports the measured max absolute difference
    across two separate invocations under these settings.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, int(threads)))
