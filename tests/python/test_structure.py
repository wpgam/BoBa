# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Concatenation along a mode and mode slicing.

Neither operation may reconstruct the dense tensor, and the ranks are what prove it.
Concatenation must come back with ranks r_a + r_b and slicing must leave ranks untouched.
A route that decompressed, operated with NumPy and recompressed would produce the right
values and the wrong ranks, so the rank assertions here are the real test and the value
comparisons merely confirm the arithmetic.
"""

import numpy as np
import pytest

import pyboba

from test_tensor_train import reference_tensor
from test_algebra import second_tensor


def interior(ranks):
    """The interface ranks that are free to grow, excluding the unit boundaries."""
    return tuple(ranks[1:-1])


# ---------------------------------------------------------------------------
# concatenation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_concatenate_matches_numpy_and_adds_ranks(mode):
    a_dense = reference_tensor((4, 6, 3))
    b_dense = second_tensor((4, 6, 3))
    a, b = pyboba.compress(a_dense), pyboba.compress(b_dense)

    joined = pyboba.concatenate([a, b], mode=mode)

    expected = np.concatenate([a_dense, b_dense], axis=mode)
    assert joined.shape == expected.shape
    assert interior(joined.ranks) == tuple(
        ra + rb for ra, rb in zip(interior(a.ranks), interior(b.ranks)))
    np.testing.assert_allclose(joined.to_numpy(), expected, rtol=1e-10, atol=1e-10)


def test_concatenate_accepts_differing_extents_on_the_joined_mode():
    a_dense = reference_tensor((4, 6, 3))
    b_dense = second_tensor((4, 2, 3))
    a, b = pyboba.compress(a_dense), pyboba.compress(b_dense)

    joined = pyboba.concatenate([a, b], mode=1)

    assert joined.shape == (4, 8, 3)
    np.testing.assert_allclose(
        joined.to_numpy(), np.concatenate([a_dense, b_dense], axis=1), rtol=1e-10, atol=1e-10)


def test_concatenate_of_three_trains():
    pieces = [reference_tensor((3, 4)), second_tensor((3, 4)), reference_tensor((3, 4)) * 2.0]
    trains = [pyboba.compress(piece) for piece in pieces]

    joined = pyboba.concatenate(trains, mode=0)

    np.testing.assert_allclose(
        joined.to_numpy(), np.concatenate(pieces, axis=0), rtol=1e-10, atol=1e-10)


def test_concatenate_of_a_single_train_is_that_train():
    a = pyboba.compress(reference_tensor((4, 5)))
    joined = pyboba.concatenate([a], mode=0)

    assert joined.shape == a.shape
    np.testing.assert_array_equal(joined.to_numpy(), a.to_numpy())


def test_concatenate_defaults_to_mode_zero():
    a_dense = reference_tensor((4, 5))
    a = pyboba.compress(a_dense)

    np.testing.assert_allclose(
        pyboba.concatenate([a, a]).to_numpy(),
        np.concatenate([a_dense, a_dense], axis=0),
        rtol=1e-10, atol=1e-10)


def test_concatenate_does_not_modify_its_operands():
    a = pyboba.compress(reference_tensor((4, 5)))
    b = pyboba.compress(second_tensor((4, 5)))
    a_ranks, b_ranks, a_dense = a.ranks, b.ranks, a.to_numpy()

    pyboba.concatenate([a, b], mode=1)

    assert a.ranks == a_ranks and b.ranks == b_ranks
    np.testing.assert_array_equal(a.to_numpy(), a_dense)


@pytest.mark.parametrize("ndim", [1, 2, 3, 7])
def test_concatenate_has_no_dimension_limit(ndim):
    shape = tuple(2 + (d % 3) for d in range(ndim))
    a = pyboba.from_cores([np.asarray(1.0 + np.arange(n)).reshape(1, n, 1) for n in shape])
    b = pyboba.from_cores([np.asarray(5.0 + np.arange(n)).reshape(1, n, 1) for n in shape])

    joined = pyboba.concatenate([a, b], mode=ndim - 1)

    expected = np.concatenate([a.to_numpy(), b.to_numpy()], axis=ndim - 1)
    np.testing.assert_allclose(joined.to_numpy(), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_concatenate_preserves_dtype(dtype):
    a = pyboba.compress(reference_tensor((4, 5), dtype))
    assert pyboba.concatenate([a, a], mode=0).dtype == np.dtype(dtype)


# ---------------------------------------------------------------------------
# slicing
# ---------------------------------------------------------------------------


def test_slice_matches_numpy_and_keeps_ranks():
    a_dense = reference_tensor((5, 7, 4))
    a = pyboba.compress(a_dense)

    sliced = a[:, 2:6, :]

    assert sliced.shape == (5, 4, 4)
    assert sliced.ranks == a.ranks
    np.testing.assert_allclose(sliced.to_numpy(), a_dense[:, 2:6, :], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "key",
    [
        (slice(None), slice(None), slice(None)),
        (slice(1, 4), slice(None), slice(None)),
        (slice(None), slice(2, 3), slice(None)),
        (slice(None), slice(None), slice(1, 4)),
        (slice(1, 4), slice(0, 5), slice(2, 4)),
        (slice(None, None, 2), slice(None), slice(None)),
        (slice(None), slice(None, None, 3), slice(None)),
        (slice(-3, None), slice(None), slice(None)),
        (slice(None), slice(None, -2), slice(None)),
        (slice(None, None, -1), slice(None), slice(None)),
        (slice(None), slice(None, None, -2), slice(None)),
    ],
)
def test_slice_follows_numpy_basic_indexing(key):
    a_dense = reference_tensor((5, 7, 4))
    a = pyboba.compress(a_dense)

    sliced = a[key]
    expected = a_dense[key]

    assert sliced.shape == expected.shape
    np.testing.assert_allclose(sliced.to_numpy(), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "key",
    [
        (2, slice(None), slice(None)),
        (slice(None), 3, slice(None)),
        (slice(None), slice(None), 1),
        (2, 3, slice(None)),
        (2, slice(None), 1),
        (slice(None), 3, 1),
        (2, slice(1, 5), 1),
    ],
)
def test_an_integer_among_slices_drops_that_mode(key):
    a_dense = reference_tensor((5, 7, 4))
    a = pyboba.compress(a_dense)

    sliced = a[key]
    expected = a_dense[key]

    assert sliced.ndim == expected.ndim
    assert sliced.shape == expected.shape
    np.testing.assert_allclose(sliced.to_numpy(), expected, rtol=1e-10, atol=1e-10)


def test_dropping_a_mode_does_not_inflate_ranks():
    a = pyboba.compress(reference_tensor((5, 7, 4)))

    dropped = a[:, 3, :]

    assert dropped.ranks == (a.ranks[0], a.ranks[1], a.ranks[3])


def test_a_full_slice_reproduces_the_train():
    a_dense = reference_tensor((5, 7, 4))
    a = pyboba.compress(a_dense)

    whole = a[:, :, :]

    assert whole.ranks == a.ranks
    np.testing.assert_allclose(whole.to_numpy(), a_dense, rtol=1e-10, atol=1e-10)


def test_scalar_indexing_still_returns_a_number():
    a_dense = reference_tensor((5, 7, 4))
    a = pyboba.compress(a_dense)

    value = a[1, 2, 3]

    assert isinstance(value, float)
    assert value == pytest.approx(a_dense[1, 2, 3], rel=1e-10)


def test_slicing_does_not_modify_the_original():
    a_dense = reference_tensor((5, 7, 4))
    a = pyboba.compress(a_dense)
    ranks = a.ranks

    a[:, 2:5, :]
    a[:, 3, :]

    assert a.ranks == ranks
    np.testing.assert_allclose(a.to_numpy(), a_dense, rtol=1e-10, atol=1e-10)


def test_slices_compose():
    a_dense = reference_tensor((6, 8, 5))
    a = pyboba.compress(a_dense)

    twice = a[1:5, :, :][:, 2:6, :]

    np.testing.assert_allclose(
        twice.to_numpy(), a_dense[1:5, 2:6, :], rtol=1e-10, atol=1e-10)


def test_a_sliced_train_still_supports_arithmetic():
    a_dense = reference_tensor((5, 7, 4))
    b_dense = second_tensor((5, 7, 4))
    a, b = pyboba.compress(a_dense), pyboba.compress(b_dense)

    total = a[:, 2:5, :] + b[:, 2:5, :]

    np.testing.assert_allclose(
        total.to_numpy(), a_dense[:, 2:5, :] + b_dense[:, 2:5, :], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_slice_preserves_dtype(dtype):
    a = pyboba.compress(reference_tensor((5, 7, 4), dtype))
    assert a[:, 2:5, :].dtype == np.dtype(dtype)


@pytest.mark.parametrize("ndim", [2, 3, 7])
def test_slicing_has_no_dimension_limit(ndim):
    shape = tuple(3 + (d % 2) for d in range(ndim))
    a = pyboba.from_cores([np.asarray(1.0 + np.arange(n)).reshape(1, n, 1) for n in shape])
    a_dense = a.to_numpy()

    key = tuple(slice(0, 2) if d == 0 else slice(None) for d in range(ndim))

    np.testing.assert_allclose(a[key].to_numpy(), a_dense[key], rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# round trip between the two
# ---------------------------------------------------------------------------


def test_slicing_a_concatenation_recovers_the_pieces():
    a_dense = reference_tensor((4, 6, 3))
    b_dense = second_tensor((4, 2, 3))
    a, b = pyboba.compress(a_dense), pyboba.compress(b_dense)

    joined = pyboba.concatenate([a, b], mode=1)

    np.testing.assert_allclose(
        joined[:, :6, :].to_numpy(), a_dense, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        joined[:, 6:, :].to_numpy(), b_dense, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_concatenate_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one train"):
        pyboba.concatenate([])


def test_concatenate_rejects_non_trains():
    a = pyboba.compress(reference_tensor((4, 5)))
    with pytest.raises(TypeError, match="TensorTrain objects"):
        pyboba.concatenate([a, np.ones((4, 5))])


def test_concatenate_rejects_mismatched_extents_off_the_joined_mode():
    a = pyboba.compress(reference_tensor((4, 6, 3)))
    b = pyboba.compress(reference_tensor((4, 6, 2)))

    with pytest.raises(ValueError, match="matching extents"):
        pyboba.concatenate([a, b], mode=1)


def test_concatenate_rejects_mismatched_dtypes():
    a = pyboba.compress(reference_tensor((4, 5), np.float32))
    b = pyboba.compress(reference_tensor((4, 5), np.float64))

    with pytest.raises(TypeError, match="same dtype"):
        pyboba.concatenate([a, b], mode=0)


def test_concatenate_rejects_mismatched_dimension():
    a = pyboba.compress(reference_tensor((4, 5)))
    b = pyboba.compress(reference_tensor((4, 5, 3)))

    with pytest.raises(ValueError, match="same number of modes"):
        pyboba.concatenate([a, b], mode=0)


def test_concatenate_rejects_an_out_of_range_mode():
    a = pyboba.compress(reference_tensor((4, 5)))

    with pytest.raises(ValueError, match="out of range"):
        pyboba.concatenate([a, a], mode=2)


def test_slice_rejects_an_empty_selection():
    a = pyboba.compress(reference_tensor((5, 7, 4)))

    with pytest.raises(ValueError, match="selects no indices"):
        a[:, 5:5, :]


def test_slice_rejects_an_incomplete_key():
    a = pyboba.compress(reference_tensor((5, 7, 4)))

    with pytest.raises(IndexError, match="complete multi-index"):
        a[:, 2:5]


def test_slice_rejects_an_out_of_bounds_integer():
    a = pyboba.compress(reference_tensor((5, 7, 4)))

    with pytest.raises(IndexError, match="out of bounds"):
        a[:, 99, :]
