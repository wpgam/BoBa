# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Arbitrary runtime dimension.

These dimensions are deliberately irregular and include values well past anything a
fixed instantiation table would plausibly cover. Nothing in the extension is specialized
per dimension, so the only cost of a large ``ndim`` is memory and time.
"""

import numpy as np
import pytest

import pyboba

# 33 is past the point where an explicit-instantiation scheme would have stopped.
HIGH_DIMENSIONS = [1, 2, 3, 7, 17, 33]


def rank_two_scalar(index):
    return 1.0 + float(sum(index))


def rank_two_batch(indices):
    return 1.0 + indices.sum(axis=1).astype(np.float64)


@pytest.mark.parametrize("ndim", HIGH_DIMENSIONS)
def test_cross_at_arbitrary_dimension(ndim):
    # Mode size 2 and rank 2 keep memory tiny even at ndim = 33.
    shape = tuple(2 for _ in range(ndim))

    tt = pyboba.cross(shape, rank_two_scalar, initial_rank=2, tolerance=1e-10)

    assert tt.ndim == ndim
    assert tt.shape == shape
    assert len(tt.ranks) == ndim + 1
    assert tt.ranks[0] == 1
    assert tt.ranks[-1] == 1
    assert len(tt.cores) == ndim

    probe = [
        tuple(0 for _ in range(ndim)),
        tuple(1 for _ in range(ndim)),
        tuple(d % 2 for d in range(ndim)),
    ]
    for index in probe:
        assert tt[index] == pytest.approx(rank_two_scalar(index), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("ndim", [2, 3, 7, 17, 33])
def test_vectorized_cross_at_arbitrary_dimension(ndim):
    shape = tuple(2 for _ in range(ndim))
    tt = pyboba.cross(shape, rank_two_batch, initial_rank=2, vectorized=True, tolerance=1e-10)

    assert tt.ndim == ndim
    index = tuple(d % 2 for d in range(ndim))
    assert tt[index] == pytest.approx(rank_two_scalar(index), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("ndim", [1, 2, 3, 7, 17])
def test_dense_compression_at_arbitrary_dimension(ndim):
    """Dense reconstruction only, so dimensions stay where 2**ndim is affordable."""
    shape = tuple(2 for _ in range(ndim))
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    dense = 1.0 + sum(g.astype(np.float64) for g in grids)

    tt = pyboba.compress(dense)

    assert tt.ndim == ndim
    assert tt.shape == shape
    np.testing.assert_allclose(tt.to_numpy(), dense, rtol=1e-9, atol=1e-9)


def test_very_high_dimension_entry_evaluation_avoids_dense_size():
    """At ndim = 33 the dense tensor has 2**33 entries, so only entries are touched."""
    ndim = 33
    shape = tuple(2 for _ in range(ndim))
    tt = pyboba.cross(shape, rank_two_scalar, initial_rank=2, tolerance=1e-10)

    assert tt.ndim == ndim
    index = tuple(1 if d % 3 == 0 else 0 for d in range(ndim))
    assert tt[index] == pytest.approx(rank_two_scalar(index), rel=1e-8, abs=1e-8)

    # The cores together are far smaller than the dense tensor.
    stored = sum(core.size for core in tt.cores)
    assert stored < 10_000
