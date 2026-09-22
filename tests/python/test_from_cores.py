# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Building a tensor train directly from its cores."""

import numpy as np
import pytest

import pyboba

from test_tensor_train import reference_tensor


def rank_one_cores(shape, dtype=np.float64):
    """Cores of the outer product of ``1 + arange(n)`` over every mode."""
    return [
        np.asarray(1.0 + np.arange(n), dtype=dtype).reshape(1, n, 1) for n in shape
    ]


def rank_one_dense(shape, dtype=np.float64):
    value = np.ones(shape, dtype=np.float64)
    for axis, n in enumerate(shape):
        vector = 1.0 + np.arange(n)
        value = value * vector.reshape([-1 if a == axis else 1 for a in range(len(shape))])
    return value.astype(dtype)


def test_from_cores_is_the_inverse_of_cores():
    tt = pyboba.compress(reference_tensor((5, 7, 4)))
    rebuilt = pyboba.from_cores(tt.cores)

    assert rebuilt.shape == tt.shape
    assert rebuilt.ranks == tt.ranks
    assert rebuilt.dtype == tt.dtype
    # Nothing is recompressed, so this must be exact, not merely close.
    np.testing.assert_array_equal(rebuilt.to_numpy(), tt.to_numpy())


def test_from_cores_builds_a_known_rank_one_tensor():
    shape = (3, 4, 2)
    tt = pyboba.from_cores(rank_one_cores(shape))

    assert tt.shape == shape
    assert tt.ranks == (1, 1, 1, 1)
    np.testing.assert_allclose(tt.to_numpy(), rank_one_dense(shape), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("ndim", [1, 2, 3, 7, 17])
def test_from_cores_has_no_dimension_limit(ndim):
    shape = tuple(2 + (d % 3) for d in range(ndim))
    tt = pyboba.from_cores(rank_one_cores(shape))

    assert tt.ndim == ndim
    assert tt.shape == shape
    np.testing.assert_allclose(tt.to_numpy(), rank_one_dense(shape), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_from_cores_preserves_dtype(dtype):
    tt = pyboba.from_cores(rank_one_cores((3, 4), dtype))
    assert tt.dtype == np.dtype(dtype)


def test_from_cores_accepts_an_explicit_dtype():
    cores = rank_one_cores((3, 4), np.float64)
    tt = pyboba.from_cores(cores, dtype=np.float32)
    assert tt.dtype == np.dtype(np.float32)


def test_from_cores_accepts_any_memory_layout():
    shape = (4, 5, 3)
    tt = pyboba.compress(reference_tensor(shape))
    contiguous = [np.ascontiguousarray(core) for core in tt.cores]
    fortran = [np.asfortranarray(core) for core in tt.cores]
    # A strided view: pad each mode, then slice back out.
    strided = []
    for core in tt.cores:
        padded = np.zeros((core.shape[0], core.shape[1] * 2, core.shape[2]))
        padded[:, ::2, :] = core
        strided.append(padded[:, ::2, :])

    dense = pyboba.from_cores(contiguous).to_numpy()
    np.testing.assert_array_equal(pyboba.from_cores(fortran).to_numpy(), dense)
    np.testing.assert_array_equal(pyboba.from_cores(strided).to_numpy(), dense)


def test_from_cores_copies_its_input():
    cores = rank_one_cores((3, 4))
    tt = pyboba.from_cores(cores)
    before = tt.to_numpy()

    cores[0][:] = 0.0

    np.testing.assert_array_equal(tt.to_numpy(), before)


def test_from_cores_accepts_nested_lists():
    tt = pyboba.from_cores([[[[1.0], [2.0], [3.0]]]])
    assert tt.shape == (3,)
    np.testing.assert_allclose(tt.to_numpy(), [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_from_cores_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one core"):
        pyboba.from_cores([])


def test_from_cores_rejects_a_non_sequence():
    with pytest.raises(TypeError, match="sequence of 3-D arrays"):
        pyboba.from_cores(42)


def test_from_cores_rejects_wrong_core_rank():
    with pytest.raises(ValueError, match="must have 3 axes"):
        pyboba.from_cores([np.ones((3, 4))])


def test_from_cores_rejects_zero_extent():
    with pytest.raises(ValueError, match="non-positive extent"):
        pyboba.from_cores([np.ones((1, 0, 1))])


def test_from_cores_rejects_broken_interface_rank_chain():
    cores = [np.ones((1, 3, 2)), np.ones((3, 4, 1))]
    with pytest.raises(ValueError, match="interface rank mismatch between cores 0 and 1"):
        pyboba.from_cores(cores)


def test_from_cores_rejects_non_unit_first_boundary_rank():
    cores = [np.ones((2, 3, 2)), np.ones((2, 4, 1))]
    with pytest.raises(ValueError, match="first core must have left rank 1"):
        pyboba.from_cores(cores)


def test_from_cores_rejects_non_unit_last_boundary_rank():
    cores = [np.ones((1, 3, 2)), np.ones((2, 4, 3))]
    with pytest.raises(ValueError, match="last core must have right rank 1"):
        pyboba.from_cores(cores)


def test_from_cores_rejects_mixed_dtypes_without_an_explicit_choice():
    cores = [np.ones((1, 3, 1), dtype=np.float32), np.ones((1, 4, 1), dtype=np.float64)]
    with pytest.raises(TypeError, match="mixed dtypes"):
        pyboba.from_cores(cores)


def test_from_cores_resolves_mixed_dtypes_when_told_which():
    cores = [np.ones((1, 3, 1), dtype=np.float32), np.ones((1, 4, 1), dtype=np.float64)]
    assert pyboba.from_cores(cores, dtype=np.float64).dtype == np.dtype(np.float64)


@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.complex128])
def test_from_cores_rejects_unsupported_dtypes(dtype):
    with pytest.raises(TypeError, match="float32 and float64"):
        pyboba.from_cores([np.ones((1, 3, 1), dtype=dtype)])
