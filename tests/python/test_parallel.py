# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Concurrent pyboba calls: thread safety, per-call seeds and pyboba.parallel."""

import threading

import numpy as np
import pytest

import pyboba
from pyboba import parallel

SHAPE = (9, 8, 7, 6)


def inverse_sum(indices):
    return 1.0 / (1.0 + indices.sum(axis=1).astype(np.float64))


def smooth_positive_train(shift):
    grids = np.meshgrid(*[1.0 + np.arange(n) for n in (6, 7, 5)], indexing="ij")
    return pyboba.compress(shift + sum(grids))


def cross_case(i):
    return pyboba.cross(SHAPE, lambda idx: inverse_sum(idx) * (1.0 + 0.1 * i),
                        vectorized=True, tolerance=1e-8, seed=i)


def cross_function_case(i):
    t = smooth_positive_train(1.0 + i)
    return pyboba.cross_function(lambda a: np.exp(-a / 10.0), [t], tolerance=1e-8, seed=i)


# ---------------------------------------------------------------------------
# per-call seed
# ---------------------------------------------------------------------------


def test_seed_kwarg_ignores_earlier_random_state():
    a = pyboba.cross(SHAPE, inverse_sum, vectorized=True, seed=5)
    pyboba.cross(SHAPE, inverse_sum, vectorized=True)  # advance the thread's generator
    b = pyboba.cross(SHAPE, inverse_sum, vectorized=True, seed=5)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


@pytest.mark.parametrize("bad", [-1, 2**32, 1.5, "1", True])
def test_rejects_invalid_seeds(bad):
    with pytest.raises((TypeError, ValueError)):
        pyboba.cross(SHAPE, inverse_sum, vectorized=True, seed=bad)


# ---------------------------------------------------------------------------
# parallel equivalence: threads reproduce serial results bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [cross_case, cross_function_case])
@pytest.mark.parametrize("threads", [2, 4, 8])
def test_threaded_results_match_serial(case, threads):
    items = range(12)
    serial = parallel.map(case, items, threads=1)
    threaded = parallel.map(case, items, threads=threads)
    for s, t in zip(serial, threaded):
        assert s.ranks == t.ranks
        np.testing.assert_array_equal(s.to_numpy(), t.to_numpy())


def test_last_cross_info_is_per_thread():
    pyboba.cross(SHAPE, inverse_sum, vectorized=True, val_size=50, seed=0)
    main_info = pyboba.last_cross_info()
    seen = {}

    def worker():
        pyboba.cross((5, 5), lambda idx: inverse_sum(idx), vectorized=True, seed=0)
        seen["info"] = pyboba.last_cross_info()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen["info"]["val_history"] == []
    assert pyboba.last_cross_info() == main_info


def test_callback_exceptions_propagate_from_threads():
    def case(i):
        def f(idx):
            if i == 3:
                raise RuntimeError("boom")
            return inverse_sum(idx)
        return pyboba.cross(SHAPE, f, vectorized=True, seed=i)

    with pytest.raises(RuntimeError, match="boom"):
        parallel.map(case, range(6), threads=4)


# ---------------------------------------------------------------------------
# worker-count configuration
# ---------------------------------------------------------------------------


def test_num_threads_precedence(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    monkeypatch.setenv("PYBOBA_NUM_THREADS", "3")
    assert parallel.num_threads() == 3
    monkeypatch.delenv("PYBOBA_NUM_THREADS")
    assert parallel.num_threads() == 6
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert parallel.num_threads() >= 1


def test_num_threads_rejects_garbage(monkeypatch):
    monkeypatch.setenv("PYBOBA_NUM_THREADS", "lots")
    with pytest.raises(ValueError):
        parallel.num_threads()


def test_map_preserves_order_and_handles_empty_input():
    assert parallel.map(lambda x: x * x, range(10), threads=4) == [x * x for x in range(10)]
    assert parallel.map(lambda x: x, [], threads=4) == []
