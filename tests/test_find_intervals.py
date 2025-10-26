#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from circleclust.utils import find_intervals


def to_list(arr):
    """Helper to convert returned ndarray to list of tuples for easy compare."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return []
    return [tuple(map(int, row)) for row in arr]


def test_find_intervals_basic():
    # 0 2 3 0 0 5 0 1 1 0
    x = np.array([0, 2, 3, 0, 0, 5, 0, 1, 1, 0])
    idx = find_intervals(x)
    assert to_list(idx) == [(1, 3), (5, 6), (7, 9)]


def test_find_intervals_singletons():
    # 0 1 0 0 4 0 0 2 0
    x = np.array([0, 1, 0, 0, 4, 0, 0, 2, 0])
    idx = find_intervals(x)
    assert to_list(idx) == [(1, 2), (4, 5), (7, 8)]


def test_find_intervals_all_positive():
    x = np.array([1, 2, 3])
    idx = find_intervals(x)
    assert to_list(idx) == [(0, 3)]


def test_find_intervals_all_zeros():
    x = np.zeros(7, dtype=int)
    idx = find_intervals(x)
    assert to_list(idx) == []


def test_find_intervals_nmin_filter():
    # Intervals: [1,3) len=2, [5,6) len=1, [7,9) len=2
    x = np.array([0, 2, 3, 0, 0, 5, 0, 1, 1, 0])
    idx = find_intervals(x)
    assert to_list(idx) == [(1, 3), (5, 6), (7, 9)]


def test_find_intervals_stable_order():
    # Should keep natural left-to-right order
    x = np.array([0, 2, 3, 0, 0, 5, 0, 1, 1, 0])
    idx = find_intervals(x)
    assert to_list(idx) == [(1, 3), (5, 6), (7, 9)]


def test_find_intervals_invalid_negative():
    x = np.array([0, -1, 2])
    with pytest.raises(AssertionError):
        _ = find_intervals(x)


def test_find_intervals_floats_with_negatives_raises():
    x = np.array([0.0, -0.5, 1.2, 0.0])
    with pytest.raises(AssertionError):
        _ = find_intervals(x)


def test_find_intervals_bools():
    x = np.array([False, True, True, False, True, False], dtype=bool)
    idx = find_intervals(x)
    assert to_list(idx) == [(1, 3), (4, 5)]


def test_find_intervals_touching_edges():
    # Intervals that touch both the start (index 0) and the end (index N)
    x = np.array([1, 1, 0, 2, 2])
    idx = find_intervals(x)
    assert to_list(idx) == [(0, 2), (3, 5)]
