# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Dense compression, reconstruction, entry evaluation and metadata."""

import numpy as np
import pytest

import pyboba


def reference_tensor(shape, dtype=np.float64):
    """A deterministic, genuinely low-rank tensor that depends differently on each axis."""
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    value = np.ones(shape, dtype=np.float64)
    for axis, grid in enumerate(grids):
        value = value * (1.0 + (axis + 1) * grid.astype(np.float64) / (shape[axis] + 1))
    return np.ascontiguousarray(value, dtype=dtype)


def test_package_imports_public_surface():
    assert hasattr(pyboba, "TensorTrain")
    assert callable(pyboba.compress)
    assert callable(pyboba.cross)


def test_compress_float64_roundtrip():
    a = reference_tensor((5, 7, 4), np.float64)
    tt = pyboba.compress(a)

    assert tt.shape == (5, 7, 4)
    assert tt.ndim == 3
    assert tt.dtype == np.dtype(np.float64)

    reconstructed = tt.to_numpy()
    assert reconstructed.shape == (5, 7, 4)
    assert reconstructed.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(reconstructed, a, rtol=1e-10, atol=1e-10)


def test_compress_float32_roundtrip():
    a = reference_tensor((5, 7, 4), np.float32)
    tt = pyboba.compress(a)

    assert tt.dtype == np.dtype(np.float32)
    reconstructed = tt.to_numpy()
    assert reconstructed.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(reconstructed, a, rtol=1e-4, atol=1e-5)


def test_axis_order_is_not_reversed():
    """Unequal extents plus an axis-asymmetric payload catch any transposition."""
    shape = (5, 7, 4)
    a = np.zeros(shape, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                a[i, j, k] = 100.0 * i + 10.0 * j + k

    tt = pyboba.compress(a)
    reconstructed = tt.to_numpy()

    assert reconstructed.shape == shape
    np.testing.assert_allclose(reconstructed, a, rtol=1e-9, atol=1e-9)

    # A shape that could be confused by a reversal would still match above, so check a
    # couple of explicitly asymmetric positions too.
    assert tt[4, 0, 0] == pytest.approx(400.0, abs=1e-9)
    assert tt[0, 6, 0] == pytest.approx(60.0, abs=1e-9)
    assert tt[0, 0, 3] == pytest.approx(3.0, abs=1e-9)


def test_entry_evaluation_matches_source_without_to_numpy():
    shape = (5, 7, 4)
    a = reference_tensor(shape)
    tt = pyboba.compress(a)

    for index in [(0, 0, 0), (4, 6, 3), (2, 3, 1), (1, 5, 2), (3, 0, 3)]:
        assert tt[index] == pytest.approx(float(a[index]), rel=1e-9, abs=1e-10)


def test_metadata_and_repr():
    a = reference_tensor((8, 9, 10))
    tt = pyboba.compress(a)

    assert tt.ndim == 3
    assert tt.shape == (8, 9, 10)
    assert isinstance(tt.shape, tuple)

    ranks = tt.ranks
    assert isinstance(ranks, tuple)
    assert len(ranks) == tt.ndim + 1
    assert ranks[0] == 1
    assert ranks[-1] == 1

    cores = tt.cores
    assert len(cores) == 3
    for d, core in enumerate(cores):
        assert isinstance(core, np.ndarray)
        assert core.dtype == tt.dtype
        assert core.shape == (ranks[d], tt.shape[d], ranks[d + 1])

    text = repr(tt)
    assert text.startswith("TensorTrain(shape=(8, 9, 10), ranks=(")
    assert text.endswith("dtype=float64)")


def test_cores_are_copies_not_views():
    a = reference_tensor((4, 5, 3))
    tt = pyboba.compress(a)

    before = tt[1, 2, 1]
    core = tt.cores[0]
    core[...] = 1234.5
    after = tt[1, 2, 1]

    assert after == pytest.approx(before)


def test_max_rank_truncation_is_lossy_but_bounded():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 7, 5))

    exact = pyboba.compress(a)
    truncated = pyboba.compress(a, max_rank=2)

    assert max(truncated.ranks) <= 2
    assert max(exact.ranks) > 2

    error = np.linalg.norm(truncated.to_numpy() - a) / np.linalg.norm(a)
    assert 0.0 < error < 1.0


def test_rtol_controls_rank():
    shape = (6, 7, 5)
    rng = np.random.default_rng(11)
    base = rng.normal(size=shape)
    a = base + 1e-6 * rng.normal(size=shape)

    loose = pyboba.compress(a, rtol=1e-1)
    tight = pyboba.compress(a, rtol=1e-14)
    assert max(loose.ranks) <= max(tight.ranks)


def test_noncontiguous_input_is_copied_in_the_right_order():
    shape = (5, 7, 4)
    a = reference_tensor(shape)

    padded = np.zeros((shape[0], shape[1], 2 * shape[2]), dtype=np.float64)
    padded[:, :, ::2] = a
    view = padded[:, :, ::2]
    assert not view.flags["C_CONTIGUOUS"]

    tt = pyboba.compress(view)
    np.testing.assert_allclose(tt.to_numpy(), a, rtol=1e-10, atol=1e-10)


def test_fortran_ordered_input():
    shape = (5, 7, 4)
    a = reference_tensor(shape)
    fortran = np.asfortranarray(a)

    tt = pyboba.compress(fortran)
    np.testing.assert_allclose(tt.to_numpy(), a, rtol=1e-10, atol=1e-10)


def test_one_dimensional_tensor_train():
    a = np.linspace(0.0, 1.0, 9)
    tt = pyboba.compress(a)

    assert tt.ndim == 1
    assert tt.shape == (9,)
    assert tt.ranks == (1, 1)
    np.testing.assert_allclose(tt.to_numpy(), a, rtol=1e-12, atol=1e-12)
    assert tt[4] == pytest.approx(a[4])
