"""Dataset assembly and the three splitting regimes.

Splitting is where synthetic-data projects usually leak, so it is explicit here
and all three regimes are reported side by side:

``random``
    Shuffle sequences. **This leaks.** The same subject -- same limb lengths,
    same style factor -- appears in train and test, and so does the same defect
    combination. It is included precisely so the leakage can be *measured*
    rather than asserted.
``subject``
    Entire subjects are held out. The model must generalise across body
    proportions and movement style.
``combination``
    Entire *defect combinations* are held out: every multi-defect sequence whose
    kind set contains ``compensation`` goes to test. Compensation on its own is
    seen in training; compensation *interacting* with another defect is not.
    This is the hard case, because a compensatory pattern is exactly the defect
    whose signature changes when something else is also wrong.

The gap between ``random`` and the other two is the leakage estimate, and it is
reported in ``results/tables/split_comparison.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generator import GeneratorConfig, Sample, generate_sample

SPLIT_MODES: tuple[str, ...] = ("random", "subject", "combination")


@dataclass
class SequenceDataset:
    """A materialised set of sequences plus every label they carry.

    Attributes:
        samples: The sequences, in order.
        coords: ``(N, 3, T, V)`` float32 model input -- channel-first, which is
            the layout the graph convolution wants.
        quality: ``(N,)`` float32 exact labels.
        joint_truth: ``(N, V)`` attribution ground truth per joint.
        frame_truth: ``(N, T)`` attribution ground truth per frame.
    """

    samples: list[Sample]
    coords: np.ndarray
    quality: np.ndarray
    joint_truth: np.ndarray
    frame_truth: np.ndarray

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def actions(self) -> np.ndarray:
        return np.array([s.action for s in self.samples])

    @property
    def subjects(self) -> np.ndarray:
        return np.array([s.subject for s in self.samples])

    @property
    def combinations(self) -> np.ndarray:
        return np.array([s.combination for s in self.samples])

    def subset(self, idx: np.ndarray) -> SequenceDataset:
        """A view-like copy restricted to ``idx``."""
        idx = np.asarray(idx, dtype=np.int64)
        return SequenceDataset(
            samples=[self.samples[i] for i in idx],
            coords=self.coords[idx],
            quality=self.quality[idx],
            joint_truth=self.joint_truth[idx],
            frame_truth=self.frame_truth[idx],
        )


def build_dataset(num_sequences: int, cfg: GeneratorConfig | None = None,
                  start: int = 0) -> SequenceDataset:
    """Materialise ``num_sequences`` samples starting at index ``start``.

    Args:
        num_sequences: How many.
        cfg: Generator config.
        start: First sample index. Distinct ``start`` values give disjoint
            sequences from the same distribution.

    Returns:
        A :class:`SequenceDataset`.

    Raises:
        ValueError: if ``num_sequences`` is not positive.
    """
    if num_sequences <= 0:
        raise ValueError(f"num_sequences must be positive, got {num_sequences}")
    cfg = cfg or GeneratorConfig()
    samples = [generate_sample(start + i, cfg) for i in range(num_sequences)]
    coords = np.stack([s.positions.transpose(2, 0, 1) for s in samples]).astype(np.float32)
    return SequenceDataset(
        samples=samples,
        coords=coords,
        quality=np.array([s.quality for s in samples], dtype=np.float32),
        joint_truth=np.stack([s.joint_truth for s in samples]).astype(np.float32),
        frame_truth=np.stack([s.frame_truth for s in samples]).astype(np.float32),
    )


def split_indices(
    data: SequenceDataset,
    mode: str = "subject",
    val_fraction: float = 0.15,
    seed: int = 0,
    test_fraction: float = 0.25,
) -> dict[str, np.ndarray]:
    """Train / val / test index arrays under one of the three regimes.

    Args:
        data: The materialised dataset.
        mode: One of :data:`SPLIT_MODES`.
        val_fraction: Fraction of the *non-test* pool used for validation.
        seed: RNG seed for the random components of each regime.
        test_fraction: Target test fraction. Honoured exactly by ``random``;
            approximated by ``subject`` (whole subjects) and determined by the
            data for ``combination`` (whichever sequences carry a held-out
            combination).

    Returns:
        ``{"train": idx, "val": idx, "test": idx}``, disjoint and covering all
        indices.

    Raises:
        ValueError: on an unknown mode, or if a split comes out empty -- an
            empty test set would produce metrics that look fine and mean
            nothing.
    """
    n = len(data)
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)

    if mode == "random":
        perm = rng.permutation(n)
        n_test = int(round(test_fraction * n))
        test = perm[:n_test]
        pool = perm[n_test:]
    elif mode == "subject":
        subjects = data.subjects
        uniq = np.unique(subjects)
        held = rng.permutation(uniq)[: max(1, int(round(test_fraction * uniq.size)))]
        test = all_idx[np.isin(subjects, held)]
        pool = all_idx[~np.isin(subjects, held)]
    elif mode == "combination":
        combos = data.combinations
        held_mask = np.array(
            [("compensation" in c.split("+")) and ("+" in c) for c in combos]
        )
        test = all_idx[held_mask]
        pool = all_idx[~held_mask]
    else:
        raise ValueError(f"Unknown split mode {mode!r}; known: {SPLIT_MODES}")

    pool = rng.permutation(pool)
    n_val = int(round(val_fraction * pool.size))
    val, train = pool[:n_val], pool[n_val:]

    out = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    for name, idx in out.items():
        if idx.size == 0:
            raise ValueError(
                f"Split {mode!r} produced an empty {name} set from {n} sequences; "
                "increase the dataset size."
            )
    return out


def label_stats(data: SequenceDataset) -> dict[str, float]:
    """Summary of the quality-label distribution, for the data-audit table."""
    q = data.quality.astype(np.float64)
    return {
        "n": float(q.size),
        "mean": float(q.mean()),
        "sd": float(q.std(ddof=1)) if q.size > 1 else 0.0,
        "min": float(q.min()),
        "max": float(q.max()),
        "frac_clean": float(np.mean(q >= 0.999)),
        "frac_floor": float(np.mean(q <= 0.051)),
    }


def paired_degradation_sequences(
    num_pairs: int,
    cfg: GeneratorConfig | None = None,
    seed: int = 0,
    kinds: tuple[str, ...] = ("rom", "asymmetry", "tempo", "jerk", "instability",
                              "compensation"),
    severity_steps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[dict[str, object]]:
    """Monotone severity ladders: identical execution, increasing defect severity.

    This is the test set for the monotonicity property. Each ladder fixes the
    subject, style, action, defect kind, defect target and *all the defect's
    internal randomness*, and varies only the severity. The exact label is
    therefore strictly decreasing along the ladder, and any increase in a
    model's prediction is unambiguously a violation -- not a confound.

    Sensor noise is switched **off** for the ladders. With noise on, a violation
    could be caused by the noise realisation rather than by the model, and the
    measured violation rate would be an upper bound of unknown looseness.

    Args:
        num_pairs: How many ladders.
        cfg: Generator config; the sensor model is disabled on a copy.
        seed: RNG seed.
        kinds: Defect kinds to build ladders for, cycled.
        severity_steps: Severities along each ladder, ascending.

    Returns:
        A list of dicts with keys ``coords`` ``(S, 3, T, V)``, ``quality``
        ``(S,)``, ``kind``, ``action``, ``severity``.

    Raises:
        ValueError: if ``severity_steps`` is not ascending.
    """
    from dataclasses import replace as _replace

    from .actions import ACTION_SPECS
    from .degradations import Degradation
    from .generator import _build_positions
    from .kinematics import canonicalise, subject_limb_scale

    if list(severity_steps) != sorted(severity_steps):
        raise ValueError("severity_steps must be ascending")
    base = cfg or GeneratorConfig()
    clean_cfg = _replace(base, noise_std=0.0, jitter_std=0.0, dropout_prob=0.0)
    rng = np.random.default_rng(seed)
    out: list[dict[str, object]] = []

    for i in range(num_pairs):
        kind = kinds[i % len(kinds)]
        action = str(clean_cfg.actions[int(rng.integers(0, len(clean_cfg.actions)))])
        spec = ACTION_SPECS[action]
        subject = int(rng.integers(0, clean_cfg.num_subjects))
        subj_rng = np.random.default_rng(555_555 + subject)
        limb_scale = subject_limb_scale(subj_rng)
        style = float(subj_rng.normal(0.0, 0.6))

        if kind == "rom":
            dof = str(rng.choice(spec.primary_dofs))
            dofs = (dof,) if dof not in spec.mirror else (dof, spec.mirror[dof])
            detail: dict[str, float | str] = {}
        elif kind == "asymmetry":
            sided = [d for d in spec.primary_dofs if d in spec.mirror] or list(spec.primary_dofs)
            dofs = (str(rng.choice(sided)),)
            detail = {}
        elif kind == "compensation":
            dofs = (str(rng.choice(spec.primary_dofs)), "spine_flex")
            detail = {"compensation_gain": 0.55}
        elif kind == "jerk":
            dofs = tuple(str(d) for d in rng.choice(spec.primary_dofs, size=2, replace=False))
            detail = {}
        else:
            dofs = ()
            detail = {}

        coords, quality = [], []
        for sev in severity_steps:
            degs = [] if sev <= 0 else [Degradation(kind, float(sev), dofs, (0.0, 1.0), detail)]
            pos = canonicalise(
                _build_positions(action, clean_cfg, style, limb_scale, degs, seed + i)
            )
            coords.append(pos.transpose(2, 0, 1).astype(np.float32))
            quality.append(1.0 - min(sum(d.cost for d in degs), 0.95))
        out.append(
            {
                "coords": np.stack(coords),
                "quality": np.array(quality, dtype=np.float32),
                "severity": np.array(severity_steps, dtype=np.float32),
                "kind": kind,
                "action": action,
            }
        )
    return out
