# BoBa Tensor Decomposition Library

[![GitHub release](https://img.shields.io/github/release/llnl/BoBa.svg)](https://github.com/llnl/BoBa/releases/latest)

Huge datasets and tables, as well as high-dimensional structured problems, are challenging to work with due to their large memory requirements, even by today's standards. This motivates the development of data compression techniques that allow us to work directly with data in its compressed format. The **BoBa library** aims to provide performance-portable abstractions that enable tensor methods that are low-memory, controllably accurate, and efficient across heterogeneous hardware.

BoBa is a (mostly) header-only C++ library containing algorithms for matrices, tensors, and tensor decompositions designed for next-generation heterogeneous architectures. It enables the creation of matrix, tensor, and tensor algorithms that achieve high performance on both laptops and HPC clusters.

BoBa is very much a work in progress. Contributions are welcome!

## Core Features

At its core, BoBa offers a variety of features for matrices and tensors:

- **Performance portability** across CPUs and GPUs for dense matrix and tensor algebra, including support for CUDA, HIP, and [Eigen](https://gitlab.com/libeigen/eigen).
- **Dense, memory-owning object abstractions** for arrays, vectors, matrices, and tensors that follow the "resource allocation is initialization" philosophy.
- Capabilities to create **views** of such objects, including 'atomic' and 'const' views.
- Routines for calling open-source and vendor linear algebra operations, including **LU**, **Cholesky**, **QR**, **SVD**, and more.
- Routines for calling open-source and vendor tensor algebra operations, including **permutations**, **contractions**, **reductions**, and more.
- **Portable multi-dimensional loops and reductions** enabled by [RAJA](https://github.com/LLNL/RAJA).
- **Dense object memory pooling** enabled by [Umpire](https://github.com/LLNL/Umpire).
- **Static (compile-time sized) versions** of matrices and tensors, with relevant algorithms.
- A **C++20 feature set**.

## Tensor Decomposition Library

Built atop BoBa's core functionality is a performance-portable tensor library that includes implementations for arbitrary-dimensional tensor decompositions:

- **tensor trains**
- **Tucker decompositions**
- **Hierarchical Tucker decompositions**
- **Canonical Polyadic decompositions**
- Matrix/operator variations of some of the above, e.g. **tensor train matrices**
- **Static (compile-time sized) views** of some of these decompositions, useful for reading data from decompositions that were computed offline

Further, BoBa offers implementations for arbitrary-dimensional **multilinear and nonlinear solvers**.

Lastly, tutorials, informative unit tests, and exercises demonstrate all of the above.

## Citation
Please use this citation when citing BoBa
```
  @misc{bobalibrary,
    title  = {BoBa: Performance-portable Tensor Decomposition Library},
    author = {Guthrey, Pierson and
              Blomquist, Matthew Thomas and
              Burmark, Jason and
              Yao, Jin and
              Vuchkov, Radoslav and
              De, Saibal and
              Nelson, Austin and
              Sands, Bill and
              Ricketson, Lee and
              Jones, Holger and
              Walton, Steven and
              Demir, Sinan and
              Joseph, Ilon and
              Minner, Paul and
              Irving, Samuel},
    howpublished = {[Computer Software] \url{https://github.com/llnl/BoBa}},
    abstractNote = {BoBa is a C++ software library for working with large matrices, tensors, and tensor decompositions. The library provides tools for dense matrix and tensor operations, tensor decompositions, and tensor decomposition methods that support modern CPU and GPU architectures. It includes portable abstractions for linear algebra, tensor algebra, and multidimensional computation. BoBa is intended for scientific computing applications that involve large multidimensional data sets or high dimensional mathematical models. Its capabilities support tasks such as data compression, linear algebra, efficient numerical computation, and the development of scalable algorithms for heterogeneous hardware. Tutorials, tests, and example applications are included to help users learn and apply the library.},
    url    = {https://github.com/llnl/BoBa},
    year   = {2026},
    releasenumber = {LLNL-CODE-2022014},
    doi = {10.11578/dc.20260804.1},
  }
```

# License

BoBa is distributed under the Apache-2.0 with LLVM exception License. See [LICENSE](LICENSE) for
the full terms and [NOTICE](NOTICE) for the LLNL/DOE government notice.

LLNL-CODE-2022014

## Getting Started

- Our tutorials [examples/tutorials](examples/tutorials) explain some BoBa basics and are a great place to start learning the library.
- You can self-generate the Doxygen documentation in the documentation directory to easily find information in HTML format.
- When you feel ready to run some examples, follow the installation instructions below.
- Next, start in the [examples/tests](examples/tests) directory which has very simple examples that demonstrate the interesting features of tensor methods, but also serve as our CI testing suite.
- Then, move on to our [examples/exercises](examples/exercises), which demonstrate some more elaborate use cases of tensor decompositions.
- When you are ready to integrate BoBa into your project, take a look at our [examples/cmake_examples](examples/cmake_examples), which provide exemplary integrations on various platforms.
- Additionally, browse our list of peer-reviewed publications in the bibtex files in [documentation/bibtex](documentation/bibtex)

## Developer Navigation

- CI test matrix and example invocations: [`ci.yaml`](ci.yaml)
- Makefile build wiring: [`Makefile`](Makefile) and [`Makefile_boba`](Makefile_boba)
- CMake target wiring: [`cmake/SetupMacros.cmake`](cmake/SetupMacros.cmake)
- Repository-wide contributor and agent guidance: [`AGENTS.md`](AGENTS.md)
- Repository-local skill index: [`skills/SKILLS.md`](skills/SKILLS.md)
- Python bindings (`pyboba`): [`python/`](python) and [`tests/python/`](tests/python)

## Installing TPLs

BoBa requires a few third-party libraries (TPLs). Install them by calling the builder directly; it auto-selects the appropriate machine recipe on supported hosts:

```bash
./boba_builder.py
```

You may need to add a recipe and builder logic for your system.

If you are reproducing a CI-style build, add `--ci`.

Third-party dependencies currently include:

- BLT (BSD-3-Clause)
- Caliper (BSD-3-Clause)
- camp (BSD-3-Clause)
- Eigen (MPL-2.0 with additional bundled notices)
- fmt (MIT)
- HDF5 (HDF5 license)
- RAJA (BSD-3-Clause)
- Umpire (MIT)

See [tpl/license/README.md](tpl/license/README.md) for more information.

BoBa's link-time dependencies currently include:

- Apple Accelerate
- Apple Metal
- NVIDIA libraries: cuBLAS, cuSOLVER, cuSPARSE, cuTENSOR
- AMD hipBLAS, hipSOLVER, rocBLAS, rocSOLVER, hipTENSOR

## Quick example

To ensure that everything is ready to go, let's try to build everything.
```
make clean && make all -j 10
```
This is an excellent test to make sure the code compiles for the given system configuration. Now we can start running the built examples, e.g.:
```
./test_boba_tensor_train_cpu.out
```

## Python interface

BoBa ships an optional Python package, `pyboba`, that exposes tensor trains: dense
compression, DMRG cross approximation from a Python callable, dense reconstruction and
entry evaluation. It is a thin, private pybind11 extension over BoBa's own numerical kernels —
NumPy is only the interchange format, never the compute backend.

### Prerequisites

The Python package builds the BoBa library from source, so the usual BoBa build
prerequisites apply first:

- a C++20 compiler
- CMake 3.21 or newer
- BoBa's third-party libraries, installed with `./boba_builder.py`
- BLT, which lives in the `tpl/blt` submodule (`git submodule update --init tpl/blt`)

The build looks for BLT in `tpl/blt` and Eigen in `install/eigen_cpu`, which is where
`boba_builder.py` places them. Point somewhere else with the `CMAKE_ARGS` environment
variable:

```bash
CMAKE_ARGS="-DEIGEN_DIR=/path/to/eigen -DBLT_SOURCE_DIR=/path/to/blt" python -m pip install .
```

`pybind11` and `scikit-build-core` are declared as build dependencies in
`pyproject.toml`, so pip fetches them automatically. A normal C++ BoBa build needs
neither Python nor pybind11: the bindings are behind the `BOBA_BUILD_PYTHON` CMake
option, which defaults to `OFF`.

### Installing

```bash
python -m pip install .
```

or, for development:

```bash
python -m pip install -e .
```

Then:

```python
import pyboba
```

The public surface is `pyboba.TensorTrain`, `pyboba.compress` and `pyboba.cross`. The compiled
module `pyboba._pyboba` is an implementation detail and should not be imported directly.

### Dense compression

```python
import numpy as np
import pyboba

rng = np.random.default_rng(42)
a = rng.normal(size=(5, 7, 4)).astype(np.float32)

tt = pyboba.compress(a)

assert tt.shape == (5, 7, 4)
assert tt.ndim == 3
assert tt.dtype == np.dtype(np.float32)

a_hat = tt.to_numpy()
```

`compress` runs BoBa's TT-SVD. Accuracy and size are controlled with `rtol`, `atol` and
`max_rank`, which map to BoBa's `svd_tolerance_relative`, `svd_tolerance_absolute` and
`max_kept_singular_values`:

```python
tt = pyboba.compress(a, rtol=1e-6, max_rank=8)
```

### Cross approximation from a callable

`pyboba.cross` approximates a tensor that is never stored, sampling it only at the entries
the DMRG sweep selects.

```python
import numpy as np
import pyboba

shape = (8, 9, 10, 7)

def f(index):
    i, j, k, l = index
    return 1.0 / (1.0 + i + j + k + l)

tt = pyboba.cross(
    shape,
    f,
    initial_rank=3,
    tolerance=1e-7,
    n_sweeps=10,
    selection="maxvol",
    dtype=np.float64,
)

value = tt[2, 3, 4, 1]
```

The scalar callback receives a tuple of zero-based Python integers of length
`len(shape)`.

`initial_rank` accepts either an integer, applied to every interior interface, or a
sequence of exactly `len(shape) - 1` interior ranks; the unit boundary ranks are
implicit. A rank larger than the interface can mathematically carry raises `ValueError`
rather than being silently truncated.

### Vectorized callbacks

Passing `vectorized=True` switches to a batched protocol, which is substantially faster
because it collapses many C++-to-Python transitions into one call:

```python
import numpy as np
import pyboba

shape = (12, 11, 10, 9)

def f(indices):
    return np.exp(-0.01 * np.sum(indices.astype(np.float64) ** 2, axis=1))

tt = pyboba.cross(shape, f, initial_rank=4, vectorized=True, dtype=np.float64)
```

The callback receives a C-contiguous `int64` array of shape `(N, len(shape))` and must
return `N` real values. `vectorized` is an explicit switch: the callable is never probed
to guess which protocol it implements.

Exceptions raised inside either callback propagate to the caller unchanged and abort the
sweep cleanly.

### Submatrix selection

`selection="maxvol"` (the default, matching BoBa's C++ default) and `selection="deim"`
map onto BoBa's existing MAXVOL and DEIM implementations. Any other value raises
`ValueError`.

### Tensor train API

| Member | Meaning |
| --- | --- |
| `tt.ndim` | Number of tensor modes |
| `tt.shape` | Tuple of mode extents |
| `tt.ranks` | All `ndim + 1` interface ranks, `(1, r1, ..., 1)` |
| `tt.dtype` | NumPy dtype, comparable with `np.dtype(...)` |
| `tt.cores` | List of NumPy arrays shaped `(left_rank, mode_size, right_rank)` |
| `tt.to_numpy()` | Dense reconstruction with exactly `tt.shape` |
| `tt[i, j, k]` | Single entry from a complete multi-index |
| `repr(tt)` | e.g. `TensorTrain(shape=(8, 9, 10), ranks=(1, 4, 5, 1), dtype=float64)` |

Indices are zero-based and must be non-negative; negative indexing and slicing are
rejected with a clear error rather than silently reinterpreted. Entry evaluation
contracts only the selected core slices, so it is far cheaper than `to_numpy()`.

`to_numpy()` raises `MemoryError` when the dense size cannot be represented — worth
remembering, since a tensor train with many modes routinely describes a dense tensor far
larger than any machine.

### Supported dtypes

`float32` and `float64`. Complex types are not exposed yet.

`compress` takes its dtype from the input array and rejects any other dtype rather than
converting silently, so convert explicitly if needed:

```python
tt = pyboba.compress(np.asarray(a, dtype=np.float64))
```

`cross` has no input array to infer from, so it takes `dtype=` and defaults to
`float64`.

### Runtime dimensionality

The tensor dimension is a runtime value. The extension contains no per-dimension
instantiation table and no dimension dispatch, so there is no artificial maximum `ndim`;
the practical limits are memory, integer range and the algorithm itself.

```python
shape = (2,) * 33
tt = pyboba.cross(shape, lambda index: 1.0 + sum(index), initial_rank=2)
assert tt.ndim == 33
```

### Memory ownership

In this first release every exchange with NumPy is a copy. Input arrays are copied into
BoBa-owned storage, and `tt.cores` and `tt.to_numpy()` return freshly allocated NumPy
arrays. Nothing returned to Python aliases BoBa memory, so no array can dangle and
writing to a returned core cannot corrupt the train. BoBa-owned memory remains the
canonical representation, which leaves room for zero-copy views in a later release.

### CPU-only for now

The Python package computes on the host. This is a limitation of the bindings, not of
BoBa: the C++ library's CUDA and HIP backends are unaffected and fully supported. GPU
execution is simply not reachable from Python yet, and the Python API deliberately
avoids baking the host execution space into its types so device support can be added
without replacing it.

### Running the Python tests

```bash
python -m pip install -e ".[test]"
python -m pytest tests/python
```

`examples/tests/test_python_runtime_parity.cpp` is the matching C++ test. It checks that
the runtime-dimensional implementation behind the bindings stays numerically faithful to
BoBa's native, dimension-templated tensor train and `DMRGCross`.

## Tips and Tricks

- BoBa has [Caliper](https://github.com/LLNL/Caliper) integration, which allows you to see runtime hotspots in the code. This is done by building with the `BOBA_ENABLE_CALIPER=1` flag and exporting the `CALI_CONFIG=runtime-report` environment variable. Use `BOBA_ENABLE_CALIPER_EXTERNAL=1` when you want Caliper scopes in code outside BoBa internals, such as exercises and tests:
```
source caliper_lib_path_info_cpu
make clean && make test_boba_tensor_train BOBA_ENABLE_CALIPER=1
CALI_CONFIG=runtime-report ./test_boba_tensor_train_cpu_cali.out
```
The `caliper_lib_path_info_cpu` file is generated by `./boba_builder.py`.

- When building, add the `BOBA_CHECKPOINTS=1` flag to activate checkpointing throughout the code so that you can find the spot where your code runs into a bug.
```
make clean && make test_boba_tensor_train BOBA_CHECKPOINTS=1 && ./test_boba_tensor_train_cpu.out
```

- When building, add the flag `BOBA_DEBUG=1` to (for example) perform bounds checking on all array accesses (except inside GPU kernels) and generate other debugging information. This may help find out-of-bounds array access and other pesky bugs.
```
make clean && make test_boba_tensor_train BOBA_DEBUG=1 && ./test_boba_tensor_train_cpu_debug.out
```

## Quick example on AMD GPUs (on LC)

- Log on to a machine such as `tuolumne`, then try to build and run a test on the GPU
```
make clean && make test_boba_tensor_train BOBA_HIP=1
```
- Now run the produced file
```
./test_boba_tensor_train_hip.out
```
