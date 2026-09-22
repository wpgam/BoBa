# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exact tensor-train arithmetic: addition, scaling, Hadamard products, norms.

Two properties are checked throughout. The values must match dense NumPy, and the ranks
must show that the operation was carried out in the TT format rather than by
reconstructing, operating densely and recompressing: an exact sum has ranks r_a + r_b
and an exact Hadamard product has ranks r_a * r_b. A rank that comes back smaller than
that would mean truncation crept in somewhere it was not asked for.
"""

import numpy as np
import pytest

import pyboba

from test_tensor_train import reference_tensor


def second_tensor(shape, dtype=np.float64):
    """A different low-rank tensor of the same shape, independent of reference_tensor."""
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    value = np.zeros(shape, dtype=np.float64)
    for axis, grid in enumerate(grids):
        value = value + (axis + 2.0) * np.cos(grid.astype(np.float64) / (shape[axis] + 1))
    return np.ascontiguousarray(value, dtype=dtype)


def operand_pair(shape=(5, 7, 4), dtype=np.float64):
    a_dense = reference_tensor(shape, dtype)
    b_dense = second_tensor(shape, dtype)
    return a_dense, b_dense, pyboba.compress(a_dense), pyboba.compress(b_dense)


# ---------------------------------------------------------------------------
# addition, subtraction, negation
# ---------------------------------------------------------------------------


def test_addition_matches_dense_and_adds_ranks():
    a_dense, b_dense, a, b = operand_pair()
    total = a + b

    expected_ranks = tuple(
        ra + rb if 0 < d < len(a.ranks) - 1 else 1
        for d, (ra, rb) in enumerate(zip(a.ranks, b.ranks))
    )
    assert total.ranks == expected_ranks
    np.testing.assert_allclose(total.to_numpy(), a_dense + b_dense, rtol=1e-10, atol=1e-10)


def test_subtraction_matches_dense():
    a_dense, b_dense, a, b = operand_pair()
    difference = a - b

    np.testing.assert_allclose(
        difference.to_numpy(), a_dense - b_dense, rtol=1e-10, atol=1e-10)


def test_negation_matches_dense_and_keeps_ranks():
    a_dense, _, a, _ = operand_pair()
    negated = -a

    assert negated.ranks == a.ranks
    np.testing.assert_allclose(negated.to_numpy(), -a_dense, rtol=1e-10, atol=1e-10)


def test_adding_a_train_to_itself_doubles_it():
    a_dense, _, a, _ = operand_pair()
    doubled = a + a

    np.testing.assert_allclose(doubled.to_numpy(), 2.0 * a_dense, rtol=1e-10, atol=1e-10)


def test_arithmetic_does_not_modify_its_operands():
    a_dense, b_dense, a, b = operand_pair()
    a_ranks, b_ranks = a.ranks, b.ranks

    a + b
    a - b
    a * b
    -a
    3.0 * a

    assert a.ranks == a_ranks
    assert b.ranks == b_ranks
    np.testing.assert_allclose(a.to_numpy(), a_dense, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b.to_numpy(), b_dense, rtol=1e-10, atol=1e-10)


def test_sum_can_be_rounded_back_down():
    a_dense, b_dense, a, b = operand_pair()
    total = a + b

    rounded = total.round(rtol=1e-12)

    assert all(r <= g for r, g in zip(rounded.ranks, total.ranks))
    np.testing.assert_allclose(
        rounded.to_numpy(), a_dense + b_dense, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scalar", [0.0, 1.0, -2.5, 7])
def test_scalar_multiplication_matches_dense(scalar):
    a_dense, _, a, _ = operand_pair()

    np.testing.assert_allclose(
        (a * scalar).to_numpy(), scalar * a_dense, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        (scalar * a).to_numpy(), scalar * a_dense, rtol=1e-10, atol=1e-10)


def test_scalar_multiplication_keeps_ranks():
    _, _, a, _ = operand_pair()
    assert (a * 3.0).ranks == a.ranks


def test_division_by_a_scalar():
    a_dense, _, a, _ = operand_pair()
    np.testing.assert_allclose((a / 4.0).to_numpy(), a_dense / 4.0, rtol=1e-10, atol=1e-10)


def test_numpy_scalars_multiply_from_the_left():
    """__array_ufunc__ = None keeps NumPy from broadcasting into an object array."""
    a_dense, _, a, _ = operand_pair()
    product = np.float64(2.0) * a

    assert isinstance(product, pyboba.TensorTrain)
    np.testing.assert_allclose(product.to_numpy(), 2.0 * a_dense, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Hadamard product
# ---------------------------------------------------------------------------


def test_hadamard_matches_dense_and_multiplies_ranks():
    a_dense, b_dense, a, b = operand_pair()
    product = a.hadamard(b)

    expected_ranks = tuple(ra * rb for ra, rb in zip(a.ranks, b.ranks))
    assert product.ranks == expected_ranks
    np.testing.assert_allclose(
        product.to_numpy(), a_dense * b_dense, rtol=1e-10, atol=1e-10)


def test_star_operator_is_the_hadamard_product():
    _, _, a, b = operand_pair()
    np.testing.assert_array_equal((a * b).to_numpy(), a.hadamard(b).to_numpy())


def test_hadamard_with_a_ones_train_is_the_identity():
    a_dense, _, a, _ = operand_pair()
    ones = pyboba.from_cores([np.ones((1, n, 1)) for n in a.shape])

    np.testing.assert_allclose(
        a.hadamard(ones).to_numpy(), a_dense, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# inner product, norm, relative error
# ---------------------------------------------------------------------------


def test_inner_product_matches_dense():
    a_dense, b_dense, a, b = operand_pair()

    assert a.inner(b) == pytest.approx(float(np.sum(a_dense * b_dense)), rel=1e-10)


def test_inner_product_is_symmetric():
    _, _, a, b = operand_pair()
    assert a.inner(b) == pytest.approx(b.inner(a), rel=1e-12)


def test_norm_matches_dense():
    a_dense, _, a, _ = operand_pair()

    assert a.norm() == pytest.approx(float(np.linalg.norm(a_dense.ravel())), rel=1e-10)


def test_norm_is_the_square_root_of_the_self_inner_product():
    _, _, a, _ = operand_pair()
    assert a.norm() == pytest.approx(np.sqrt(a.inner(a)), rel=1e-12)


def test_relative_error_is_zero_against_itself():
    _, _, a, _ = operand_pair()
    # Not exactly zero, and not portably below it either. The difference is formed
    # exactly, so a - a is a rank-2r train that cancels only in exact arithmetic; the
    # floating-point residual of <d, d> is about eps * norm(a)**2, and the square root
    # lifts it to sqrt(eps) * norm(a). That is the documented accuracy floor of
    # relative_error, and how close to it a given platform lands depends on summation
    # order in the underlying BLAS.
    assert pyboba.relative_error(a, a) < 1e-6


def test_relative_error_matches_the_dense_definition():
    a_dense, b_dense, a, b = operand_pair()

    expected = np.linalg.norm((a_dense - b_dense).ravel()) / np.linalg.norm(b_dense.ravel())
    assert pyboba.relative_error(a, b) == pytest.approx(expected, rel=1e-8)


def test_relative_error_of_a_truncated_train_is_small_but_nonzero():
    a_dense, _, a, _ = operand_pair()
    truncated = a.round(rtol=0.0, atol=0.0, max_rank=1)

    error = pyboba.relative_error(truncated, a)
    assert 0.0 < error < 1.0


# ---------------------------------------------------------------------------
# dtypes and dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_arithmetic_preserves_dtype(dtype):
    _, _, a, b = operand_pair((5, 6, 4), dtype)

    assert (a + b).dtype == np.dtype(dtype)
    assert (a * b).dtype == np.dtype(dtype)
    assert (a * 2.0).dtype == np.dtype(dtype)


def test_float32_arithmetic_matches_dense_within_single_precision():
    a_dense, b_dense, a, b = operand_pair((5, 6, 4), np.float32)

    np.testing.assert_allclose(
        (a + b).to_numpy(), a_dense + b_dense, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        (a * b).to_numpy(), a_dense * b_dense, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("ndim", [1, 2, 3, 7])
def test_arithmetic_has_no_dimension_limit(ndim):
    shape = tuple(2 + (d % 3) for d in range(ndim))
    a_cores = [np.asarray(1.0 + np.arange(n)).reshape(1, n, 1) for n in shape]
    b_cores = [np.asarray(2.0 + np.arange(n)).reshape(1, n, 1) for n in shape]
    a = pyboba.from_cores(a_cores)
    b = pyboba.from_cores(b_cores)

    a_dense, b_dense = a.to_numpy(), b.to_numpy()

    np.testing.assert_allclose((a + b).to_numpy(), a_dense + b_dense, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose((a * b).to_numpy(), a_dense * b_dense, rtol=1e-10, atol=1e-10)
    assert a.inner(b) == pytest.approx(float(np.sum(a_dense * b_dense)), rel=1e-10)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_addition_rejects_mismatched_shapes():
    a = pyboba.compress(reference_tensor((5, 7, 4)))
    b = pyboba.compress(reference_tensor((5, 7, 3)))

    with pytest.raises(ValueError, match="same shape"):
        a + b


def test_addition_rejects_mismatched_dtypes():
    a = pyboba.compress(reference_tensor((5, 7, 4), np.float32))
    b = pyboba.compress(reference_tensor((5, 7, 4), np.float64))

    with pytest.raises(TypeError, match="same dtype"):
        a + b


def test_hadamard_rejects_mismatched_shapes():
    a = pyboba.compress(reference_tensor((5, 7, 4)))
    b = pyboba.compress(reference_tensor((5, 7, 3)))

    with pytest.raises(ValueError, match="same shape"):
        a.hadamard(b)


def test_inner_rejects_mismatched_shapes():
    a = pyboba.compress(reference_tensor((5, 7, 4)))
    b = pyboba.compress(reference_tensor((5, 7, 3)))

    with pytest.raises(ValueError, match="same shape"):
        a.inner(b)


def test_addition_rejects_a_non_train_operand():
    _, _, a, _ = operand_pair()
    with pytest.raises(TypeError):
        a + 1.0


def test_division_by_zero_is_rejected():
    _, _, a, _ = operand_pair()
    with pytest.raises(ValueError, match="division by zero"):
        a / 0.0


def test_multiplication_rejects_a_string():
    _, _, a, _ = operand_pair()
    with pytest.raises(TypeError, match="real number"):
        a * "two"


def test_relative_error_rejects_a_zero_reference():
    _, _, a, _ = operand_pair()
    zero = pyboba.from_cores([np.zeros((1, n, 1)) for n in a.shape])

    with pytest.raises(ValueError, match="zero norm"):
        pyboba.relative_error(a, zero)


def test_relative_error_rejects_mismatched_shapes():
    a = pyboba.compress(reference_tensor((5, 7, 4)))
    b = pyboba.compress(reference_tensor((5, 7, 3)))

    with pytest.raises(ValueError, match="same shape"):
        pyboba.relative_error(a, b)
