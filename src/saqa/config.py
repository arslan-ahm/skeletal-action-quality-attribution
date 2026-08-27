"""Typed configuration with ``_base_`` inheritance and ``--set`` overrides.

One YAML file fully describes an experiment and is saved next to its results.
Two properties are enforced rather than documented:

* **Unknown keys are rejected.** A silently-swallowed typo
  (``optim.leraning_rate``) would void an experiment while appearing to succeed.
* **``--set`` overrides are typed by the dataclass field**, not guessed from the
  string, so ``--set optim.epochs=20`` gives an ``int`` and
  ``--set data.noise_std=0`` gives a ``float`` -- and ``--set run.seed=1.5``
  fails loudly instead of truncating.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DataConfig:
    """Dataset size, generator settings and the splitting regime."""

    num_sequences: int = 1400
    num_frames: int = 48
    num_subjects: int = 24
    max_degradations: int = 3
    severity_min: float = 0.15
    severity_max: float = 1.0
    noise_std: float = 0.006
    jitter_std: float = 0.004
    dropout_prob: float = 0.05
    dropout_max_frames: int = 5
    clean_fraction: float = 0.04
    split: str = "subject"
    val_fraction: float = 0.15
    test_fraction: float = 0.25
    batch_size: int = 32
    #: When >0, use only this many training sequences (the data-efficiency curve).
    train_limit: int = 0


@dataclass
class ModelConfig:
    """Architecture and head selection."""

    architecture: str = "saqa_stgcn"
    head: str = "ordinal"
    uncertainty: str = "quantile"
    num_bins: int = 10
    temporal_kernel: int = 9
    partitions: str = "spatial"
    separable: bool = True
    edge_importance: bool = True
    dropout: float = 0.05
    norm: str = "batch"
    uncertainty_weight: float = 0.5
    channels: tuple[int, ...] = ()
    strides: tuple[int, ...] = ()


@dataclass
class OptimConfig:
    """Optimisation schedule."""

    epochs: int = 20
    lr: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    warmup_epochs: int = 2
    #: Cosine decay to ``lr * min_lr_factor``.
    min_lr_factor: float = 0.05


@dataclass
class EvalConfig:
    """Evaluation, statistics and attribution settings."""

    bootstrap: int = 2000
    interval_level: float = 0.90
    attribution_methods: tuple[str, ...] = ("integrated_gradients", "occlusion", "gradient")
    ig_steps: int = 24
    #: Cap on the number of test sequences scored for attribution fidelity.
    #: Integrated gradients costs ``ig_steps`` forward-backward passes per
    #: sequence per model, so the full test split across every arm would dominate
    #: the compute budget. The cap is a *prefix of the shuffled test split*, and
    #: the contributing count is reported next to every fidelity number.
    attribution_limit: int = 150
    monotonicity_ladders: int = 60
    top_k_joints: int = 3


@dataclass
class RunConfig:
    """Bookkeeping."""

    name: str = "default"
    seed: int = 0
    out_dir: str = "results/runs"
    torch_threads: int = 2
    save_checkpoint: bool = False


@dataclass
class Config:
    """The whole experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the resolved config as YAML with LF line endings."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(_plain(self.to_dict()), sort_keys=False)
        p.write_text(text, encoding="utf-8", newline="\n")


def _plain(obj: Any) -> Any:
    """Convert tuples to lists so PyYAML emits plain sequences."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list):
        return [_plain(v) for v in obj]
    return obj


def _coerce(value: Any, annotation: Any) -> Any:
    """Cast ``value`` to the dataclass field's declared type.

    Raises:
        ValueError: if the value cannot be represented in the declared type
            without loss (e.g. ``1.5`` into an ``int``).
    """
    origin = get_origin(annotation)
    if origin in (tuple, list):
        inner = get_args(annotation)
        item_t = inner[0] if inner else str
        seq = value if isinstance(value, list | tuple) else str(value).split(",")
        return tuple(_coerce(v, item_t) for v in seq)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot read {value!r} as a bool")
    if annotation is int:
        f = float(value)
        if f != int(f):
            raise ValueError(f"{value!r} is not an integer")
        return int(f)
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)
    return value


def _apply(target: Any, updates: dict[str, Any], path: str = "") -> None:
    """Recursively apply a nested dict onto a dataclass instance.

    Raises:
        KeyError: on a key the dataclass does not declare.
    """
    known = {f.name for f in fields(target)}
    # ``from __future__ import annotations`` makes ``Field.type`` a *string*, so
    # the declared types have to be resolved through get_type_hints or every
    # override would be coerced as ``str`` and silently break the config.
    hints = get_type_hints(type(target))
    for key, value in updates.items():
        if key not in known:
            where = f"{path}{key}"
            raise KeyError(
                f"Unknown config key {where!r}. Known keys here: {sorted(known)}"
            )
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, path=f"{path}{key}.")
        else:
            setattr(target, key, _coerce(value, hints[key]))


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single ``_base_`` chain relative to it."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping, got {type(raw).__name__}")
    base_name = raw.pop("_base_", None)
    if base_name is None:
        return raw
    base = _load_yaml((path.parent / str(base_name)).resolve())
    return _deep_merge(base, raw)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=x"]`` into ``{"a": {"b": "1"}, "c": "x"}``.

    Raises:
        ValueError: on an item without ``=``.
    """
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must be key=value")
        key, value = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value.strip()
    return out


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> Config:
    """Build a :class:`Config` from an optional YAML file plus ``--set`` items.

    Precedence, lowest first: dataclass defaults, ``_base_`` chain, the file
    itself, then command-line overrides. Verified by
    ``tests/test_config.py::test_override_precedence``.

    Args:
        path: YAML path, or ``None`` for pure defaults.
        overrides: ``["section.key=value", ...]``.

    Returns:
        The resolved config.
    """
    cfg = Config()
    if path is not None:
        _apply(cfg, _load_yaml(Path(path).resolve()))
    _apply(cfg, parse_overrides(overrides))
    return cfg
