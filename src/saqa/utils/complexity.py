"""Parameters, MACs and *measured* wall-clock latency.

Params and MACs are inputs to a cost argument; wall-clock is the argument. They
diverge badly for exactly the operations this project uses -- a depthwise
temporal convolution has ``kt`` MACs per output element and is entirely
memory-bandwidth bound, while a dense one has ``C * kt`` and vectorises -- so all
three are reported together with the achieved MACs-per-millisecond, which is the
number that says whether a FLOP saving converted into time.

**Warm-up is not optional.** With fewer than about eight warm-up iterations,
PyTorch's first-call allocation and kernel selection dominate a small model's
measurement and can make it look several times slower than it is. The defaults
here are 10 warm-up iterations and 30 timed repeats, and the median and IQR are
reported rather than the mean, because a background process on a shared machine
produces occasional outliers that would move a mean and not a median.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count."""
    return int(
        sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
    )


def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Multiply-accumulate count for one forward pass, by module hooks.

    Counts ``Conv1d``, ``Conv2d``, ``Linear``, ``LSTM`` and the adjacency
    ``einsum`` of :class:`~saqa.models.stgcn.SpatialGraphConv`. Normalisation,
    activations and pooling are excluded -- they are elementwise and contribute
    no multiply-accumulates, though they *do* contribute memory traffic, which is
    exactly why the MACs-per-ms ratio is reported instead of MACs alone.

    Args:
        model: The model.
        input_shape: Shape *without* the batch axis, e.g. ``(3, 64, 17)``.

    Returns:
        MACs for a single sample.
    """
    total = 0

    def conv_hook(module, inputs, output):
        nonlocal total
        out_elems = int(np.prod(output.shape[2:]))
        kernel = int(np.prod(module.kernel_size))
        total += (
            module.in_channels // module.groups * module.out_channels * kernel * out_elems
        )

    def linear_hook(module, inputs, output):
        nonlocal total
        lead = int(np.prod(output.shape[:-1])) // max(output.shape[0], 1)
        total += module.in_features * module.out_features * max(lead, 1)

    def lstm_hook(module, inputs, output):
        nonlocal total
        x = inputs[0]
        steps = x.shape[1] if module.batch_first else x.shape[0]
        dirs = 2 if module.bidirectional else 1
        for layer in range(module.num_layers):
            in_size = module.input_size if layer == 0 else module.hidden_size * dirs
            total += dirs * steps * 4 * module.hidden_size * (in_size + module.hidden_size)

    def graph_hook(module, inputs, output):
        nonlocal total
        n, c, t, v = output.shape
        total += module.num_partitions * c * t * v * v

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d | nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.LSTM):
            handles.append(m.register_forward_hook(lstm_hook))
        elif type(m).__name__ == "SpatialGraphConv":
            handles.append(m.register_forward_hook(graph_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    return int(total)


@dataclass
class LatencyResult:
    """Timing summary for one configuration."""

    median_ms: float
    iqr_ms: float
    p10_ms: float
    p90_ms: float
    repeats: int
    warmup: int
    batch_size: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def benchmark_latency(
    fn,
    warmup: int = 10,
    repeats: int = 30,
    batch_size: int = 1,
) -> LatencyResult:
    """Time a zero-argument callable.

    Args:
        fn: The thing to time; called with no arguments.
        warmup: Untimed calls first. Must be >= 8 to be trusted; a warning-worthy
            value is accepted but recorded in the result so a reader can see it.
        repeats: Timed calls.
        batch_size: Recorded for the report; the caller is responsible for making
            ``fn`` actually use it.

    Returns:
        A :class:`LatencyResult` in milliseconds per call.

    Raises:
        ValueError: if ``repeats`` is below 1.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    for _ in range(max(0, warmup)):
        fn()
    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1e3
    q25, q50, q75 = np.percentile(samples, [25, 50, 75])
    p10, p90 = np.percentile(samples, [10, 90])
    return LatencyResult(
        float(q50), float(q75 - q25), float(p10), float(p90), repeats, warmup, batch_size
    )


def model_cost(
    model: nn.Module,
    input_shape: tuple[int, ...],
    batch_sizes: tuple[int, ...] = (1, 8),
    warmup: int = 10,
    repeats: int = 30,
) -> dict[str, float]:
    """Params, MACs and latency at several batch sizes, in one dict.

    Returns keys ``params``, ``macs``, ``latency_bs{N}_ms``, ``iqr_bs{N}_ms`` and
    ``macs_per_ms_bs1`` -- the achieved arithmetic throughput, which is the number
    that shows whether a MAC reduction became a speed-up.
    """
    model.eval()
    out: dict[str, float] = {
        "params": float(count_parameters(model)),
        "macs": float(count_macs(model, input_shape)),
    }
    for bs in batch_sizes:
        x = torch.zeros(bs, *input_shape)

        def run(x=x):
            with torch.no_grad():
                model(x)

        res = benchmark_latency(run, warmup=warmup, repeats=repeats, batch_size=bs)
        out[f"latency_bs{bs}_ms"] = res.median_ms
        out[f"iqr_bs{bs}_ms"] = res.iqr_ms
    if "latency_bs1_ms" in out and out["latency_bs1_ms"] > 0:
        out["macs_per_ms_bs1"] = out["macs"] / out["latency_bs1_ms"] / 1e6
    return out
