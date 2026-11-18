# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Sweep controls of cross / cross_function and the diagnostics in last_cross_info."""

import numpy as np
import pytest

import pyboba

SHAPE = (8, 9, 10, 7)


def inverse_sum(indices):
    """1 / (1 + sum(i)): smooth, not exactly low rank."""
    return 1.0 / (1.0 + indices.sum(axis=1).astype(np.float64))


def dense_inverse_sum(shape=SHAPE):
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def relative_dense_error(tt, dense):
    return np.linalg.norm(tt.to_numpy() - dense) / np.linalg.norm(dense)


def smooth_positive_train(shape=(6, 7, 5)):
    grids = np.meshgrid(*[1.0 + np.arange(n) for n in shape], indexing="ij")
    dense = 2.0 + sum(grids)
    return pyboba.compress(dense), dense


# ---------------------------------------------------------------------------
# defaults are backward compatible
# ---------------------------------------------------------------------------


def test_default_options_match_explicit_native_values():
    pyboba.seed(7)
    implicit = pyboba.cross(SHAPE, inverse_sum, vectorized=True, initial_rank=3)
    pyboba.seed(7)
    explicit = pyboba.cross(SHAPE, inverse_sum, vectorized=True, initial_rank=3,
                            svd_tolerance=0.0, max_rank=0, val_size=0)
    np.testing.assert_array_equal(implicit.to_numpy(), explicit.to_numpy())


def test_default_run_has_no_validation_history():
    pyboba.cross(SHAPE, inverse_sum, vectorized=True)
    info = pyboba.last_cross_info()
    assert info["val_history"] == []
    assert info["sweeps"] >= 1
    assert info["n_evals"] > 0


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_seed_makes_cross_reproducible():
    pyboba.seed(42)
    a = pyboba.cross(SHAPE, inverse_sum, vectorized=True, initial_rank=3, n_sweeps=3)
    pyboba.seed(42)
    b = pyboba.cross(SHAPE, inverse_sum, vectorized=True, initial_rank=3, n_sweeps=3)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


# ---------------------------------------------------------------------------
# svd_tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("svd_tolerance", [1e-10, "auto"])
def test_svd_tolerance_keeps_accuracy(svd_tolerance):
    tt = pyboba.cross(SHAPE, inverse_sum, vectorized=True, tolerance=1e-8,
                      svd_tolerance=svd_tolerance)
    assert relative_dense_error(tt, dense_inverse_sum()) < 1e-6


def test_loose_svd_tolerance_does_not_grow_ranks():
    tight = pyboba.cross(SHAPE, inverse_sum, vectorized=True, svd_tolerance=1e-12)
    loose = pyboba.cross(SHAPE, inverse_sum, vectorized=True, svd_tolerance=1e-3)
    assert max(loose.ranks) <= max(tight.ranks)


def test_rejects_a_negative_svd_tolerance():
    with pytest.raises(ValueError):
        pyboba.cross(SHAPE, inverse_sum, vectorized=True, svd_tolerance=-1.0)


def test_rejects_an_unknown_svd_tolerance_string():
    with pytest.raises(ValueError):
        pyboba.cross(SHAPE, inverse_sum, vectorized=True, svd_tolerance="tight")


# ---------------------------------------------------------------------------
# max_rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_rank", [1, 2, 3])
def test_max_rank_caps_every_interface_rank(max_rank):
    tt = pyboba.cross(SHAPE, inverse_sum, vectorized=True, initial_rank=1,
                      max_rank=max_rank, svd_tolerance=1e-14)
    assert max(tt.ranks) <= max_rank


def test_cross_function_respects_max_rank():
    t, _ = smooth_positive_train()
    inverse = pyboba.cross_function(lambda a: 1.0 / a, [t], initial_rank=1, max_rank=2)
    assert max(inverse.ranks) <= 2


# ---------------------------------------------------------------------------
# validation-set stopping test
# ---------------------------------------------------------------------------


def test_validation_history_is_reported_per_sweep():
    pyboba.cross(SHAPE, inverse_sum, vectorized=True, tolerance=1e-8,
                 val_size=200, n_sweeps=6)
    info = pyboba.last_cross_info()
    assert len(info["val_history"]) == info["sweeps"]
    assert info["final_error"] == info["val_history"][-1]
    assert info["n_evals"] >= 200


def test_validation_error_tracks_the_true_error():
    tt = pyboba.cross(SHAPE, inverse_sum, vectorized=True, tolerance=1e-6,
                      svd_tolerance="auto", val_size=500)
    info = pyboba.last_cross_info()
    assert info["converged"]
    true_error = relative_dense_error(tt, dense_inverse_sum())
    assert true_error < 1e-4
    assert info["final_error"] < 1e-5


def test_validation_seed_is_honoured():
    kwargs = dict(vectorized=True, tolerance=1e-12, val_size=100, n_sweeps=2, val_stall=0.0)
    pyboba.seed(3)
    pyboba.cross(SHAPE, inverse_sum, val_seed=1, **kwargs)
    a = pyboba.last_cross_info()["val_history"]
    pyboba.seed(3)
    pyboba.cross(SHAPE, inverse_sum, val_seed=1, **kwargs)
    b = pyboba.last_cross_info()["val_history"]
    assert a == b


def test_budget_exhaustion_is_reported_as_not_converged():
    pyboba.cross(SHAPE, inverse_sum, vectorized=True, initial_rank=1, max_rank=1,
                 tolerance=1e-14, val_size=100, val_stall=0.0, n_sweeps=2)
    info = pyboba.last_cross_info()
    assert not info["converged"]
    # Like native DMRGCross, n_sweeps = N runs at most N - 1 full sweeps.
    assert info["sweeps"] == 1


def test_one_dimensional_cross_reports_converged():
    pyboba.cross((5,), lambda i: float(i[0]))
    assert pyboba.last_cross_info()["converged"]


# ---------------------------------------------------------------------------
# cross_function block evaluation
# ---------------------------------------------------------------------------


def test_cross_function_block_path_matches_exact_hadamard_product():
    a, dense_a = smooth_positive_train()
    b, dense_b = smooth_positive_train()
    product = pyboba.cross_function(lambda x, y: x * y, [a, b], initial_rank=2,
                                    tolerance=1e-12, val_size=200)
    np.testing.assert_allclose(product.to_numpy(), dense_a * dense_b, rtol=1e-9)
    assert pyboba.last_cross_info()["final_error"] < 1e-9


def test_cross_function_matches_cross_on_the_same_function():
    t, dense = smooth_positive_train()
    via_trains = pyboba.cross_function(lambda x: np.exp(-x / 10.0), [t], tolerance=1e-10)
    via_callback = pyboba.cross(t.shape, lambda idx: np.exp(-dense[tuple(idx.T)] / 10.0),
                                vectorized=True, tolerance=1e-10)
    np.testing.assert_allclose(via_trains.to_numpy(), via_callback.to_numpy(),
                               rtol=1e-6, atol=1e-10)
