# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Input validation at the Python boundary.

Every case here must raise an ordinary Python exception. None may trip a BoBa assertion
or terminate the interpreter.
"""

import numpy as np
import pytest

import pyboba


def simple(index):
    return 1.0 + float(sum(index))


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float16, np.complex128, np.bool_])
def test_compress_rejects_unsupported_dtypes(dtype):
    a = np.ones((3, 4), dtype=dtype)
    with pytest.raises(TypeError, match="float32 and float64"):
        pyboba.compress(a)


def test_compress_rejects_zero_extent():
    a = np.zeros((3, 0, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="positive extents"):
        pyboba.compress(a)


def test_compress_rejects_zero_dimensional_array():
    with pytest.raises(ValueError, match="at least one dimension"):
        pyboba.compress(np.float64(1.0))


def test_compress_rejects_negative_tolerances():
    a = np.ones((3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        pyboba.compress(a, rtol=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        pyboba.compress(a, atol=-1.0)


def test_compress_rejects_nonpositive_max_rank():
    a = np.ones((3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="positive integer or None"):
        pyboba.compress(a, max_rank=0)


# ---------------------------------------------------------------------------
# indexing
# ---------------------------------------------------------------------------


@pytest.fixture
def small_train():
    a = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    return pyboba.compress(a)


def test_index_wrong_length(small_train):
    with pytest.raises(IndexError, match="complete multi-index"):
        small_train[1, 2]
    with pytest.raises(IndexError, match="complete multi-index"):
        small_train[1, 2, 3, 4]


def test_index_out_of_bounds(small_train):
    with pytest.raises(IndexError, match="out of bounds"):
        small_train[3, 0, 0]
    with pytest.raises(IndexError, match="out of bounds"):
        small_train[0, 0, 5]


def test_index_negative_is_rejected_clearly(small_train):
    with pytest.raises(IndexError, match="non-negative"):
        small_train[-1, 0, 0]


def test_index_slice_is_rejected(small_train):
    with pytest.raises(TypeError, match="slicing is not supported"):
        small_train[0:2, 0, 0]


def test_index_non_integer_is_rejected(small_train):
    with pytest.raises(TypeError, match="must be integers"):
        small_train[1.5, 0, 0]


def test_numpy_integer_indices_are_accepted(small_train):
    assert small_train[np.int64(1), np.int32(2), np.int64(3)] == pytest.approx(
        small_train[1, 2, 3]
    )


# ---------------------------------------------------------------------------
# cross
# ---------------------------------------------------------------------------


def test_cross_rejects_empty_shape():
    with pytest.raises(ValueError, match="at least one mode"):
        pyboba.cross((), simple)


@pytest.mark.parametrize("shape", [(3, 0, 4), (3, -1, 4)])
def test_cross_rejects_nonpositive_extents(shape):
    with pytest.raises(ValueError, match="must be positive"):
        pyboba.cross(shape, simple)


def test_cross_rejects_non_sequence_shape():
    with pytest.raises(TypeError, match="sequence of positive integers"):
        pyboba.cross(5, simple)


def test_cross_rejects_non_integer_shape_entries():
    with pytest.raises(TypeError, match="must be integers"):
        pyboba.cross((3, 4.5), simple)


def test_cross_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        pyboba.cross((3, 4), 7)


def test_cross_rejects_bad_initial_rank():
    with pytest.raises(ValueError, match="must be positive"):
        pyboba.cross((4, 5, 6), simple, initial_rank=0)
    with pytest.raises(TypeError, match="integers"):
        pyboba.cross((4, 5, 6), simple, initial_rank="three")


def test_cross_rejects_zero_sweeps():
    with pytest.raises(ValueError, match="at least 1"):
        pyboba.cross((4, 5, 6), simple, n_sweeps=0)


def test_cross_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError, match="tolerance must be positive"):
        pyboba.cross((4, 5, 6), simple, tolerance=0.0)


@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.complex128])
def test_cross_rejects_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="float32 and float64"):
        pyboba.cross((4, 5, 6), simple, dtype=dtype)


def test_cross_accepts_dtype_spellings():
    for spelling in ("float32", np.float32, np.dtype(np.float32)):
        tt = pyboba.cross((4, 5, 6), simple, initial_rank=2, dtype=spelling)
        assert tt.dtype == np.dtype(np.float32)


def test_huge_shape_raises_memory_error_not_overflow():
    huge = (2**40, 2**40, 2**40)
    with pytest.raises(MemoryError, match="overflow"):
        pyboba.cross(huge, simple)
