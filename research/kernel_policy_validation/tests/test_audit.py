"""CPU-only contracts; run before collecting any GPU performance evidence."""
from __future__ import annotations

import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[1] / "audit.py"
    assert path.exists(), "Research audit implementation has not been added yet"
    spec = importlib.util.spec_from_file_location("kernel_policy_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("size", [2, 6, 8])
def test_order_balances_every_label_at_every_position(audit, size):
    labels = tuple(f"v{i}" for i in range(size))
    orders = audit.balanced_orders(labels, size * 4, seed=7)
    assert len(orders) == size * 4
    for order in orders:
        assert set(order) == set(labels)
    for position in range(size):
        assert Counter(order[position] for order in orders) == Counter(dict.fromkeys(labels, 4))
    assert orders == audit.balanced_orders(labels, size * 4, seed=7)
    assert orders != audit.balanced_orders(labels, size * 4, seed=8)


@pytest.mark.parametrize("labels,repeats", [([], 4), (["a", "a"], 4), (["a", "b"], 3), (["a", "b"], 0)])
def test_invalid_balancing_request_is_rejected(audit, labels, repeats):
    with pytest.raises(ValueError):
        audit.balanced_orders(labels, repeats, seed=7)


def test_equivalent_shapes_have_equal_work(audit):
    assert audit.validate_same_elements([(256, 512), (128, 1024), (64, 2048)]) == 131072


@pytest.mark.parametrize("shapes", [[], [(0, 4)], [(2, -3)], [(2, 3), (2, 4)]])
def test_unequal_or_empty_work_rejected(audit, shapes):
    with pytest.raises(ValueError):
        audit.validate_same_elements(shapes)


def test_identical_timings_have_no_apparent_speedup(audit):
    rounds = [[1000, 1100, 1200, 1300]] * 6
    result = audit.paired_summary(rounds, rounds, seed=7)
    assert result["speedup_a_over_b"] == pytest.approx(1.0)
    assert result["ci95_low"] == pytest.approx(1.0)
    assert result["ci95_high"] == pytest.approx(1.0)
    assert result["decision"] == "unresolved"


def test_speedup_direction_and_practical_margin(audit):
    a = [[2000, 2000, 2000, 2000]] * 6
    b = [[1000, 1000, 1000, 1000]] * 6
    result = audit.paired_summary(a, b, seed=7)
    assert result["speedup_a_over_b"] == pytest.approx(2.0)
    assert result["decision"] == "b_faster"
    assert result["a_median_ns"] == 2000
    reverse = audit.paired_summary(b, a, seed=7)
    assert reverse["decision"] == "a_faster"


@pytest.mark.parametrize("a,b", [([[1]], [[1]]), ([[1], [2]], [[1]]), ([[0], [0]], [[1], [1]]), ([[math.nan], [2]], [[1], [2]]), ([[1, 2], [2]], [[1], [2]])])
def test_bad_or_unpaired_samples_rejected(audit, a, b):
    with pytest.raises(ValueError):
        audit.paired_summary(a, b, seed=7)


def test_same_seed_reproduces_confidence_interval(audit):
    a = [[900 + i, 1000 + i, 1100 + i] for i in range(6)]
    b = [[850, 1050, 1000] for _ in range(6)]
    assert audit.paired_summary(a, b, seed=17) == audit.paired_summary(a, b, seed=17)
