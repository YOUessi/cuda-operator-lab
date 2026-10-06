from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def implementation():
    path = ROOT / 'timeline_audit.py'
    assert path.exists(), 'R04 timeline implementation is not present yet'
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('timeline_r04', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event(name, cat, ts, dur):
    return dict(name=name, cat=cat, ts=ts, dur=dur, ph='X', args={})


def test_union_counts_overlaps_once():
    mod = implementation()
    assert mod.interval_union_us([(1, 6), (4, 8), (10, 12)]) == 9


def test_union_empty_and_touching_intervals():
    mod = implementation()
    assert mod.interval_union_us([]) == 0
    assert mod.interval_union_us([(3, 5), (1, 3), (5, 5)]) == 4


@pytest.mark.parametrize('intervals', [[(3, 2)], [(float('nan'), 4)], [(1, float('inf'))]])
def test_invalid_intervals_rejected(intervals):
    with pytest.raises(ValueError):
        implementation().interval_union_us(intervals)


def test_trace_metrics_include_real_kernels_not_cpu_ops():
    mod = implementation()
    events = [event('r04_call:0', 'user_annotation', 0, 40),
              event('first', 'kernel', 10, 5),
              event('second', 'kernel', 20, 7),
              event('aten::mm', 'cpu_op', 2, 25),
              event('cudaLaunchKernel', 'cuda_runtime', 5, 2)]
    rows = mod.trace_metrics({'traceEvents': events}, expected_calls=1)
    assert len(rows) == 1
    row = rows[0]
    assert row['kernel_count'] == 2
    assert row['kernel_union_us'] == 12
    assert row['device_span_us'] == 17
    assert row['uncovered_between_kernels_us'] == 5
    assert row['kernel_names'] == ['first', 'second']
    assert row['launch_api_count'] == 1


def test_overlapping_kernel_span_not_negative():
    mod = implementation()
    events = [event('r04_call:0', 'user_annotation', 0, 50),
              event('k0', 'kernel', 10, 10), event('k1', 'kernel', 15, 12)]
    row = mod.trace_metrics({'traceEvents': events}, 1)[0]
    assert row['kernel_union_us'] == 17
    assert row['uncovered_between_kernels_us'] == 0


def test_no_cuda_kernel_is_not_a_gpu_trace():
    mod = implementation()
    with pytest.raises(ValueError, match='kernel'):
        mod.trace_metrics({'traceEvents': [event('r04_call:0', 'user_annotation', 0, 50)]}, 1)


def test_missing_marker_is_rejected():
    with pytest.raises(ValueError):
        implementation().trace_metrics({'traceEvents': [event('k', 'kernel', 1, 1)]}, 1)


def test_kernel_crossing_synchronized_marker_is_rejected():
    events = [event('r04_call:0', 'user_annotation', 5, 10), event('k', 'kernel', 12, 8)]
    with pytest.raises(ValueError):
        implementation().trace_metrics({'traceEvents': events}, 1)


def test_duplicate_markers_rejected():
    events = [event('r04_call:0', 'user_annotation', 0, 20),
              event('r04_call:0', 'user_annotation', 30, 20),
              event('a', 'kernel', 5, 2), event('b', 'kernel', 35, 2)]
    with pytest.raises(ValueError):
        implementation().trace_metrics({'traceEvents': events}, 2)


def test_intercall_kernels_are_not_assigned_to_workload():
    events = [event('r04_call:0', 'user_annotation', 0, 20), event('a', 'kernel', 5, 3),
              event('unrelated', 'kernel', 21, 1),
              event('r04_call:1', 'user_annotation', 30, 20), event('b', 'kernel', 35, 4)]
    rows = implementation().trace_metrics({'traceEvents': events}, 2)
    assert [row['kernel_union_us'] for row in rows] == [3, 4]
