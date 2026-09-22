# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Cross approximation of a pointwise function of existing tensor trains.

The interesting case is a function with no exact tensor-train form -- 1/x, exp, a ratio
-- since sums and products are already available exactly. One test deliberately uses a
function that *does* have an exact form, the entrywise product, so the approximation can
be checked against the exact answer rather than against a tolerance on itself.
"""

import numpy as np
import pytest

import pyboba


def smooth_positive(shape):
    """A strictly positive, genuinely low-rank tensor, safe to divide by and take logs of."""
    value = np.full(shape, 2.0)
    for axis, n in enumerate(shape):
        grid = 1.0 + np.arange(n)
        value = value + grid.reshape([-1 if a == axis else 1 for a in range(len(shape))])
    return value


def train_and_dense(shape=(6, 7, 5)):
    dense = smooth_positive(shape)
    return pyboba.compress(dense), dense


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_reciprocal_is_approximated_closely():
    t, dense = train_and_dense()

    inverse = pyboba.cross_function(lambda a: 1.0 / a, [t], initial_rank=4, tolerance=1e-12)

    np.testing.assert_allclose(inverse.to_numpy(), 1.0 / dense, rtol=1e-6, atol=1e-8)


def test_exponential_is_approximated_closely():
    t, dense = train_and_dense()

    result = pyboba.cross_function(
        lambda a: np.exp(-a / 20.0), [t], initial_rank=4, tolerance=1e-12)

    np.testing.assert_allclose(
        result.to_numpy(), np.exp(-dense / 20.0), rtol=1e-6, atol=1e-8)


def test_a_function_of_two_trains():
    t, dense = train_and_dense()
    other_dense = 1.0 + 0.5 * dense
    u = pyboba.compress(other_dense)

    ratio = pyboba.cross_function(
        lambda a, b: a / b, [t, u], initial_rank=4, tolerance=1e-12)

    np.testing.assert_allclose(
        ratio.to_numpy(), dense / other_dense, rtol=1e-5, atol=1e-7)


def test_a_function_of_three_trains():
    t, dense = train_and_dense((5, 6, 4))
    u = pyboba.compress(1.0 + 0.5 * dense)
    v = pyboba.compress(3.0 + 0.25 * dense)

    result = pyboba.cross_function(
        lambda a, b, c: a / (b + c), [t, u, v], initial_rank=4, tolerance=1e-12)

    expected = dense / (1.0 + 0.5 * dense + 3.0 + 0.25 * dense)
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5, atol=1e-7)


def test_an_entrywise_product_matches_the_exact_hadamard_product():
    """The one function here with an exact TT form, so the answer is known, not estimated."""
    t, dense = train_and_dense((5, 6, 4))
    u = pyboba.compress(1.0 + 0.5 * dense)

    exact = t.hadamard(u)
    approximated = pyboba.cross_function(
        lambda a, b: a * b, [t, u], initial_rank=4, tolerance=1e-12)

    assert pyboba.relative_error(approximated, exact) < 1e-8


def test_relative_error_reports_the_quality_of_the_approximation():
    t, dense = train_and_dense()

    inverse = pyboba.cross_function(lambda a: 1.0 / a, [t], initial_rank=4, tolerance=1e-12)
    reference = pyboba.compress(1.0 / dense)

    assert pyboba.relative_error(inverse, reference) < 1e-6


# ---------------------------------------------------------------------------
# the callback contract
# ---------------------------------------------------------------------------


def test_the_callback_receives_one_array_per_train():
    t, dense = train_and_dense()
    u = pyboba.compress(1.0 + dense)
    seen = []

    def f(a, b):
        seen.append((type(a), a.shape, a.ndim, b.shape))
        return a + b

    pyboba.cross_function(f, [t, u], initial_rank=3)

    assert seen
    for kind, a_shape, a_ndim, b_shape in seen:
        assert kind is np.ndarray
        assert a_ndim == 1
        assert a_shape == b_shape


def test_the_callback_is_batched_not_called_per_entry():
    """The whole point of the design: one Python call covers many entries."""
    t, _ = train_and_dense()
    calls = []

    def f(a):
        calls.append(a.size)
        return 1.0 / a

    pyboba.cross_function(f, [t], initial_rank=4)

    assert len(calls) > 0
    assert sum(calls) > len(calls), "batches should carry more than one entry each"
    assert max(calls) > 1


def test_the_callback_may_modify_the_arrays_it_receives():
    """They are copies, so scribbling on them cannot disturb the sweep."""
    t, dense = train_and_dense()

    def f(a):
        result = 1.0 / a
        a[:] = np.nan
        return result

    inverse = pyboba.cross_function(f, [t], initial_rank=4, tolerance=1e-12)

    assert np.all(np.isfinite(inverse.to_numpy()))
    np.testing.assert_allclose(inverse.to_numpy(), 1.0 / dense, rtol=1e-6, atol=1e-8)


def test_the_callback_values_are_the_entries_of_the_input_trains():
    """Whatever indices the sweep picks, the values handed over are that train's."""
    t, dense = train_and_dense((4, 5, 3))
    observed = []

    def f(a):
        observed.append(np.array(a))
        return a

    pyboba.cross_function(f, [t], initial_rank=3)

    all_values = np.concatenate(observed)
    assert all_values.min() >= dense.min() - 1e-9
    assert all_values.max() <= dense.max() + 1e-9


# ---------------------------------------------------------------------------
# dtypes and dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_result_dtype_follows_the_inputs(dtype):
    dense = smooth_positive((5, 6, 4)).astype(dtype)
    t = pyboba.compress(dense)

    result = pyboba.cross_function(lambda a: 1.0 / a, [t], initial_rank=3)

    assert result.dtype == np.dtype(dtype)


def test_one_dimensional_input_is_handled():
    dense = smooth_positive((7,))
    t = pyboba.compress(dense)

    inverse = pyboba.cross_function(lambda a: 1.0 / a, [t])

    assert inverse.shape == (7,)
    np.testing.assert_allclose(inverse.to_numpy(), 1.0 / dense, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("ndim", [2, 3, 5])
def test_cross_function_has_no_dimension_limit(ndim):
    shape = tuple(3 for _ in range(ndim))
    dense = smooth_positive(shape)
    t = pyboba.compress(dense)

    inverse = pyboba.cross_function(lambda a: 1.0 / a, [t], initial_rank=3, tolerance=1e-12)

    assert inverse.ndim == ndim
    np.testing.assert_allclose(inverse.to_numpy(), 1.0 / dense, rtol=1e-5, atol=1e-7)


def test_a_sliced_train_can_be_used_as_an_input():
    t, dense = train_and_dense()

    piece = t[:, 1:5, :]
    inverse = pyboba.cross_function(lambda a: 1.0 / a, [piece], initial_rank=4, tolerance=1e-12)

    np.testing.assert_allclose(
        inverse.to_numpy(), 1.0 / dense[:, 1:5, :], rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# validation and error propagation
# ---------------------------------------------------------------------------


def test_rejects_an_empty_train_sequence():
    with pytest.raises(ValueError, match="at least one input train"):
        pyboba.cross_function(lambda a: a, [])


def test_rejects_a_non_callable():
    t, _ = train_and_dense()
    with pytest.raises(TypeError, match="callable"):
        pyboba.cross_function(42, [t])


def test_rejects_non_trains():
    t, dense = train_and_dense()
    with pytest.raises(TypeError, match="TensorTrain objects"):
        pyboba.cross_function(lambda a, b: a + b, [t, dense])


def test_rejects_mismatched_shapes():
    t, _ = train_and_dense((5, 6, 4))
    u = pyboba.compress(smooth_positive((5, 6, 3)))

    with pytest.raises(ValueError, match="same shape"):
        pyboba.cross_function(lambda a, b: a + b, [t, u])


def test_rejects_mismatched_dtypes():
    t = pyboba.compress(smooth_positive((5, 6)).astype(np.float32))
    u = pyboba.compress(smooth_positive((5, 6)))

    with pytest.raises(TypeError, match="same dtype"):
        pyboba.cross_function(lambda a, b: a + b, [t, u])


def test_callback_returning_the_wrong_number_of_values_raises():
    t, _ = train_and_dense()

    with pytest.raises(ValueError, match="were requested"):
        pyboba.cross_function(lambda a: np.zeros(a.size + 1), [t], initial_rank=3)


def test_callback_returning_a_complex_dtype_raises():
    t, _ = train_and_dense()

    with pytest.raises(TypeError, match="non-real"):
        pyboba.cross_function(
            lambda a: np.zeros(a.size, dtype=np.complex128), [t], initial_rank=3)


def test_an_exception_inside_the_callback_propagates_unchanged():
    t, _ = train_and_dense()

    class Marker(Exception):
        pass

    def f(a):
        raise Marker("from the callback")

    with pytest.raises(Marker, match="from the callback"):
        pyboba.cross_function(f, [t], initial_rank=3)


def test_rejects_a_non_positive_tolerance():
    t, _ = train_and_dense()
    with pytest.raises(ValueError, match="tolerance must be positive"):
        pyboba.cross_function(lambda a: a, [t], tolerance=0.0)


def test_rejects_zero_sweeps():
    t, _ = train_and_dense()
    with pytest.raises(ValueError, match="n_sweeps"):
        pyboba.cross_function(lambda a: a, [t], n_sweeps=0)
