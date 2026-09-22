# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Orthogonalization and rank truncation.

The tests that matter here inflate a train's ranks without changing the tensor it
represents, then check that rounding removes exactly the redundancy and leaves the
values alone. That is the property the rest of the exact-arithmetic surface depends on.
"""

import numpy as np
import pytest

import pyboba

from test_tensor_train import reference_tensor


def pad_interface_with_zeros(cores, interface, extra=2):
    """Widen one interface with a zero block: same tensor, larger ranks."""
    cores = [np.array(core) for core in cores]
    left, right = cores[interface], cores[interface + 1]

    cores[interface] = np.concatenate(
        [left, np.zeros((left.shape[0], left.shape[1], extra))], axis=2)
    cores[interface + 1] = np.concatenate(
        [right, np.zeros((extra, right.shape[1], right.shape[2]))], axis=0)
    return cores


def duplicate_interface(cores, interface):
    """Widen one interface with a redundant but genuinely non-zero block."""
    cores = [np.array(core) for core in cores]
    left, right = cores[interface], cores[interface + 1]

    cores[interface] = np.concatenate([left, left], axis=2)
    cores[interface + 1] = 0.5 * np.concatenate([right, right], axis=0)
    return cores


def left_unfolding(core):
    """The (left_rank * mode_size, right_rank) matrix BoBa orthogonalizes."""
    left, mode, right = core.shape
    return core.reshape(left * mode, right, order="F")


# ---------------------------------------------------------------------------
# rounding
# ---------------------------------------------------------------------------


def test_round_removes_a_zero_padded_interface():
    tt = pyboba.compress(reference_tensor((5, 7, 4)))
    inflated = pyboba.from_cores(pad_interface_with_zeros(tt.cores, 0, extra=3))

    assert inflated.ranks[1] == tt.ranks[1] + 3
    np.testing.assert_allclose(inflated.to_numpy(), tt.to_numpy(), rtol=1e-12, atol=1e-12)

    rounded = inflated.round(rtol=1e-10)

    assert rounded.ranks[1] <= tt.ranks[1]
    np.testing.assert_allclose(rounded.to_numpy(), tt.to_numpy(), rtol=1e-9, atol=1e-9)


def test_round_removes_a_duplicated_interface():
    tt = pyboba.compress(reference_tensor((5, 7, 4)))
    inflated = pyboba.from_cores(duplicate_interface(tt.cores, 1))

    assert inflated.ranks[2] == tt.ranks[2] * 2
    np.testing.assert_allclose(inflated.to_numpy(), tt.to_numpy(), rtol=1e-12, atol=1e-12)

    rounded = inflated.round(rtol=1e-10)

    assert rounded.ranks[2] <= tt.ranks[2]
    np.testing.assert_allclose(rounded.to_numpy(), tt.to_numpy(), rtol=1e-9, atol=1e-9)


def test_round_does_not_modify_the_original():
    tt = pyboba.compress(reference_tensor((5, 7, 4)))
    inflated = pyboba.from_cores(pad_interface_with_zeros(tt.cores, 0, extra=3))

    before_ranks = inflated.ranks
    before_dense = inflated.to_numpy()

    inflated.round(rtol=1e-10)

    assert inflated.ranks == before_ranks
    np.testing.assert_array_equal(inflated.to_numpy(), before_dense)


def test_round_respects_an_integer_max_rank():
    tt = pyboba.compress(reference_tensor((6, 7, 5)))
    rounded = tt.round(rtol=0.0, atol=0.0, max_rank=2)

    assert all(rank <= 2 for rank in rounded.ranks[1:-1])
    assert rounded.shape == tt.shape


def test_round_respects_a_per_interface_max_rank():
    tt = pyboba.compress(reference_tensor((6, 7, 5)))
    bounds = [1, 3, 2, 1]
    assert len(bounds) == tt.ndim + 1

    rounded = tt.round(rtol=0.0, atol=0.0, max_rank=bounds)

    assert rounded.ranks[1] <= 3
    assert rounded.ranks[2] <= 2


def test_round_keeps_an_already_minimal_train():
    tt = pyboba.compress(reference_tensor((5, 7, 4)))
    rounded = tt.round(rtol=1e-12)

    assert rounded.ranks == tt.ranks
    np.testing.assert_allclose(rounded.to_numpy(), tt.to_numpy(), rtol=1e-10, atol=1e-10)


def test_round_of_a_one_mode_train_is_a_no_op():
    tt = pyboba.from_cores([np.arange(1.0, 5.0).reshape(1, 4, 1)])
    rounded = tt.round(rtol=1e-10)

    assert rounded.ranks == (1, 1)
    np.testing.assert_array_equal(rounded.to_numpy(), tt.to_numpy())


def test_round_of_a_zero_train_collapses_to_unit_ranks():
    cores = [np.zeros((1, 4, 3)), np.zeros((3, 5, 2)), np.zeros((2, 3, 1))]
    rounded = pyboba.from_cores(cores).round(rtol=1e-10)

    assert rounded.ranks == (1, 1, 1, 1)
    np.testing.assert_array_equal(rounded.to_numpy(), np.zeros((4, 5, 3)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_round_preserves_dtype(dtype):
    tt = pyboba.compress(reference_tensor((5, 6, 4), dtype))
    rounded = tt.round(rtol=1e-5)

    assert rounded.dtype == np.dtype(dtype)
    np.testing.assert_allclose(rounded.to_numpy(), tt.to_numpy(), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("ndim", [2, 3, 7, 17])
def test_round_has_no_dimension_limit(ndim):
    shape = tuple(2 + (d % 3) for d in range(ndim))
    cores = [np.asarray(1.0 + np.arange(n)).reshape(1, n, 1) for n in shape]
    inflated = pyboba.from_cores(pad_interface_with_zeros(cores, 0, extra=2))

    rounded = inflated.round(rtol=1e-10)

    assert rounded.ndim == ndim
    assert rounded.ranks[1] == 1
    np.testing.assert_allclose(
        rounded.to_numpy(), pyboba.from_cores(cores).to_numpy(), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# orthogonalization
# ---------------------------------------------------------------------------


def test_orthogonalize_preserves_the_tensor_and_the_ranks():
    tt = pyboba.compress(reference_tensor((5, 7, 4)))
    orthogonal = tt.orthogonalize()

    assert orthogonal.ranks == tt.ranks
    np.testing.assert_allclose(orthogonal.to_numpy(), tt.to_numpy(), rtol=1e-10, atol=1e-10)


def test_orthogonalize_makes_every_core_but_the_last_left_orthogonal():
    tt = pyboba.compress(reference_tensor((5, 7, 4)))
    cores = tt.orthogonalize().cores

    for core in cores[:-1]:
        unfolding = left_unfolding(core)
        gram = unfolding.T @ unfolding
        np.testing.assert_allclose(gram, np.eye(gram.shape[0]), rtol=1e-10, atol=1e-10)


def test_orthogonalize_does_not_modify_the_original():
    tt = pyboba.compress(reference_tensor((5, 7, 4)))
    before = tt.to_numpy()
    first_core = tt.cores[0]

    tt.orthogonalize()

    np.testing.assert_array_equal(tt.to_numpy(), before)
    np.testing.assert_array_equal(tt.cores[0], first_core)


def test_orthogonalize_of_a_one_mode_train_is_a_no_op():
    tt = pyboba.from_cores([np.arange(1.0, 5.0).reshape(1, 4, 1)])
    np.testing.assert_array_equal(tt.orthogonalize().to_numpy(), tt.to_numpy())


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_round_rejects_negative_tolerances():
    tt = pyboba.compress(reference_tensor((4, 5)))
    with pytest.raises(ValueError, match="non-negative"):
        tt.round(rtol=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        tt.round(atol=-1.0)


def test_round_rejects_non_positive_max_rank():
    tt = pyboba.compress(reference_tensor((4, 5)))
    with pytest.raises(ValueError, match="positive integer"):
        tt.round(max_rank=0)


def test_round_rejects_a_wrong_length_max_rank_sequence():
    tt = pyboba.compress(reference_tensor((4, 5, 6)))
    with pytest.raises(ValueError, match=r"ndim \+ 1 = 4"):
        tt.round(max_rank=[2, 2])


def test_round_rejects_non_positive_entries_in_a_max_rank_sequence():
    tt = pyboba.compress(reference_tensor((4, 5)))
    with pytest.raises(ValueError, match="entries must be positive"):
        tt.round(max_rank=[1, 0, 1])


def test_round_rejects_a_non_integer_max_rank():
    tt = pyboba.compress(reference_tensor((4, 5)))
    with pytest.raises(TypeError, match="positive integer"):
        tt.round(max_rank="two")
