# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Python interface to the BoBa tensor decomposition library.

This first release exposes tensor trains: dense compression, DMRG cross approximation
from a callable, dense reconstruction and entry evaluation.

Notes
-----
* Tensor dimension is a runtime value. Nothing in the extension enumerates a fixed set
  of dimensions, so there is no artificial upper bound on ``ndim``.
* Supported dtypes are ``float32`` and ``float64``. Complex types are not exposed yet.
* Computation runs on the host in this release. BoBa's C++ CUDA and HIP backends are
  unaffected; they are simply not reachable from Python yet.
* Arrays exchanged with NumPy are copied in both directions, so returned arrays are
  always independent of BoBa-owned storage.
* Indices are zero-based and non-negative.
"""

from ._pyboba import TensorTrain, compress, cross, from_cores, relative_error

__all__ = [
    "TensorTrain",
    "compress",
    "cross",
    "from_cores",
    "relative_error",
]

__version__ = "0.1.0"
