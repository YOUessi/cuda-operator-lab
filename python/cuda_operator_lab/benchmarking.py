"""Reusable CUDA benchmark helpers."""

from __future__ import annotations

import statistics
from collections.abc import Callable

import torch


def measure_us(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    before_each: Callable[[], object] | None = None,
) -> list[float]:
    """Measure one CUDA callable with reusable CUDA events.

    before_each is enqueued on the same current CUDA stream before the start
    event. It is useful for controlled cache-state preparation without adding
    the preparation work to the measured interval.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    for _ in range(warmup):
        if before_each is not None:
            before_each()
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(repeats):
        if before_each is not None:
            before_each()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def percentile(samples: list[float], q: float) -> float:
    if not samples:
        raise ValueError("samples must not be empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")

    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples: list[float]) -> tuple[float, float, float]:
    """Return median, P50 and P95 in microseconds."""
    return (
        statistics.median(samples),
        percentile(samples, 0.50),
        percentile(samples, 0.95),
    )
