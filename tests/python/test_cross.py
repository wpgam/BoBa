# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""DMRG cross approximation driven by Python callbacks."""

import numpy as np
import pytest

import pyboba


def rank_two_scalar(index):
    """f(i) = 1 + sum(i): exactly TT-rank 2, so cross should reproduce it to roundoff."""
    return 1.0 + float(sum(index))


def rank_two_batch(indices):
    return 1.0 + indices.sum(axis=1).astype(np.float64)


def sampled_error(tt, reference, points):
    return max(abs(tt[p] - reference(p)) for p in points)


def test_required_scalar_workflow():
    shape = (8, 9, 10, 7)

    def f(index):
        i, j, k, l = index
        return 1.0 / (1.0 + i + j + k + l)

    tt = pyboba.cross(
        shape,
        f,
        initial_rank=3,
        tolerance=1e-7,
        n_sweeps=10,
        selection="maxvol",
        dtype=np.float64,
    )

    assert tt.shape == shape
    assert tt.ndim == 4
    assert tt.dtype == np.dtype(np.float64)

    value = tt[2, 3, 4, 1]
    expected = f((2, 3, 4, 1))
    assert abs(value - expected) < 1e-5


def test_required_vectorized_workflow():
    shape = (12, 11, 10, 9)

    def f(indices):
        return np.exp(-0.01 * np.sum(indices.astype(np.float64) ** 2, axis=1))

    tt = pyboba.cross(shape, f, initial_rank=4, vectorized=True, dtype=np.float64)

    assert tt.shape == shape
    probe = [(0, 0, 0, 0), (5, 5, 5, 5), (11, 10, 9, 8), (3, 7, 2, 4)]
    for p in probe:
        expected = float(np.exp(-0.01 * sum(v * v for v in p)))
        assert tt[p] == pytest.approx(expected, rel=1e-4, abs=1e-6)


def test_scalar_callback_is_sampled_not_enumerated():
    # Ten modes of size 8 is about 1.07e9 entries. Two-site DMRG cross touches a number
    # of points that scales with the interface ranks and two mode sizes at a time, never
    # with the dense tensor, so the gap here is several orders of magnitude.
    shape = (8,) * 10
    dense_size = 8**10
    calls = []

    def counting(index):
        calls.append(index)
        return rank_two_scalar(index)

    tt = pyboba.cross(shape, counting, initial_rank=2, tolerance=1e-10)

    assert len(calls) > 0
    distinct = set(calls)
    assert len(distinct) < dense_size / 1000, (
        f"cross sampled {len(distinct)} distinct entries of {dense_size}; "
        "it must sample, not enumerate"
    )

    for index in calls:
        assert isinstance(index, tuple)
        assert len(index) == len(shape)
        assert all(isinstance(v, int) and 0 <= v < 8 for v in index)

    probe = [(0,) * 10, (7,) * 10, tuple(d % 8 for d in range(10))]
    assert sampled_error(tt, rank_two_scalar, probe) < 1e-8


def test_vectorized_callback_receives_real_batches():
    shape = (12, 13, 11)
    batch_sizes = []
    seen = {}

    def counting(indices):
        batch_sizes.append(indices.shape[0])
        seen["ndim"] = indices.ndim
        seen["cols"] = indices.shape[1]
        seen["dtype"] = indices.dtype
        return rank_two_batch(indices)

    tt = pyboba.cross(shape, counting, initial_rank=2, tolerance=1e-10, vectorized=True)

    assert seen["ndim"] == 2
    assert seen["cols"] == len(shape)
    assert seen["dtype"] == np.dtype(np.int64)
    assert max(batch_sizes) > 1, "at least one callback must receive several points"

    probe = [(0, 0, 0), (11, 12, 10), (4, 4, 4)]
    assert sampled_error(tt, rank_two_scalar, probe) < 1e-8


def test_vectorized_uses_far_fewer_callbacks_than_scalar():
    shape = (12, 13, 11)

    scalar_calls = 0

    def scalar(index):
        nonlocal scalar_calls
        scalar_calls += 1
        return rank_two_scalar(index)

    batch_calls = 0

    def batched(indices):
        nonlocal batch_calls
        batch_calls += 1
        return rank_two_batch(indices)

    pyboba.cross(shape, scalar, initial_rank=2, tolerance=1e-10)
    pyboba.cross(shape, batched, initial_rank=2, tolerance=1e-10, vectorized=True)

    assert batch_calls < scalar_calls


def test_float32_cross():
    shape = (7, 6, 8)
    tt = pyboba.cross(shape, rank_two_scalar, initial_rank=2, dtype=np.float32)

    assert tt.dtype == np.dtype(np.float32)
    probe = [(0, 0, 0), (6, 5, 7), (3, 2, 4)]
    assert sampled_error(tt, rank_two_scalar, probe) < 1e-3


def test_float64_cross():
    shape = (7, 6, 8)
    tt = pyboba.cross(shape, rank_two_scalar, initial_rank=2, dtype=np.float64)

    assert tt.dtype == np.dtype(np.float64)
    probe = [(0, 0, 0), (6, 5, 7), (3, 2, 4)]
    assert sampled_error(tt, rank_two_scalar, probe) < 1e-8


def test_default_dtype_is_float64():
    tt = pyboba.cross((4, 5, 6), rank_two_scalar, initial_rank=2)
    assert tt.dtype == np.dtype(np.float64)


def test_initial_rank_scalar():
    tt = pyboba.cross((6, 7, 5, 4), rank_two_scalar, initial_rank=3)
    assert tt.ndim == 4
    probe = [(0, 0, 0, 0), (5, 6, 4, 3)]
    assert sampled_error(tt, rank_two_scalar, probe) < 1e-8


def test_initial_rank_sequence():
    shape = (6, 7, 5, 4, 6)
    tt = pyboba.cross(shape, rank_two_scalar, initial_rank=(2, 3, 5, 2))
    assert tt.ndim == 5
    probe = [(0, 0, 0, 0, 0), (5, 6, 4, 3, 5)]
    assert sampled_error(tt, rank_two_scalar, probe) < 1e-8


def test_initial_rank_sequence_wrong_length_raises():
    with pytest.raises(ValueError, match="interior ranks"):
        pyboba.cross((6, 7, 5, 4), rank_two_scalar, initial_rank=(2, 3))


def test_initial_rank_above_attainable_raises():
    with pytest.raises(ValueError, match="attainable rank"):
        pyboba.cross((2, 3, 4), rank_two_scalar, initial_rank=50)


@pytest.mark.parametrize("selection", ["maxvol", "deim"])
def test_selection_strategies(selection):
    shape = (8, 9, 7)
    tt = pyboba.cross(
        shape, rank_two_scalar, initial_rank=2, selection=selection, tolerance=1e-10
    )
    probe = [(0, 0, 0), (7, 8, 6), (3, 4, 5)]
    assert sampled_error(tt, rank_two_scalar, probe) < 1e-7


def test_unknown_selection_raises():
    with pytest.raises(ValueError, match="maxvol"):
        pyboba.cross((4, 5, 6), rank_two_scalar, selection="mystery")


def test_scalar_callback_exception_propagates():
    class UserFailure(RuntimeError):
        pass

    def broken(index):
        if index[0] == 3:
            raise UserFailure("user evaluator failed")
        return 1.0

    with pytest.raises(UserFailure, match="user evaluator failed"):
        pyboba.cross((8, 6, 7), broken, initial_rank=2)

    # The interpreter must still be usable afterwards.
    tt = pyboba.cross((4, 5, 6), rank_two_scalar, initial_rank=2)
    assert tt.ndim == 3


def test_vectorized_callback_exception_propagates():
    def broken(indices):
        raise KeyError("batched evaluator failed")

    with pytest.raises(KeyError):
        pyboba.cross((6, 5, 7), broken, initial_rank=2, vectorized=True)


def test_scalar_callback_returning_nonscalar_raises():
    def bad(index):
        return "not a number"

    with pytest.raises(TypeError):
        pyboba.cross((4, 5, 6), bad, initial_rank=2)


def test_vectorized_callback_returning_wrong_size_raises():
    def bad(indices):
        return np.zeros(indices.shape[0] + 1)

    with pytest.raises(ValueError, match="were requested"):
        pyboba.cross((4, 5, 6), bad, initial_rank=2, vectorized=True)


def test_vectorized_callback_returning_complex_raises():
    def bad(indices):
        return np.zeros(indices.shape[0], dtype=np.complex128)

    with pytest.raises(TypeError, match="non-real"):
        pyboba.cross((4, 5, 6), bad, initial_rank=2, vectorized=True)


def test_cross_one_dimensional_is_exact():
    calls = []

    def f(index):
        calls.append(index)
        return float(index[0]) ** 2

    tt = pyboba.cross((9,), f, dtype=np.float64)

    assert tt.ndim == 1
    assert tt.shape == (9,)
    for i in range(9):
        assert tt[i] == pytest.approx(float(i) ** 2)
    assert len(calls) == 9
