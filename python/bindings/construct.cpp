// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "py_tensor_train.hpp"

#include "detail/errors.hpp"
#include "detail/numpy_interop.hpp"
#include "detail/runtime_tensor_train.hpp"

#include <pybind11/numpy.h>

#include <cstring>
#include <string>
#include <vector>

/**
 * \file
 * \brief Building a tensor train directly from its cores.
 *
 * This is the inverse of `TensorTrain.cores`, and the entry point for trains that were
 * not produced by compression or cross approximation -- analytic constructions, trains
 * loaded from a file, or output from another library.
 */

namespace boba_python
{

namespace
{

/**
 * \brief Validates a user-supplied core sequence against the tensor-train invariants.
 *
 * Each check here corresponds to a way a malformed core list would otherwise surface:
 * as a wrong answer, or as a BoBa assertion firing deep inside a later sweep with no
 * indication of which core was at fault.
 *
 *   - exactly three axes per core, ``(left_rank, mode_size, right_rank)``;
 *   - positive extents throughout, since a zero extent has no valid interpretation;
 *   - chained interface ranks, ``right_rank(d) == left_rank(d + 1)``;
 *   - unit boundary ranks, which every routine in this extension relies on.
 */
std::vector<py::array> parse_cores(py::handle cores_like)
{
  if (!py::isinstance<py::sequence>(cores_like) || py::isinstance<py::str>(cores_like))
  {
    throw py::type_error("from_cores() expects a sequence of 3-D arrays");
  }
  auto sequence = py::reinterpret_borrow<py::sequence>(cores_like);

  const std::size_t ndim = py::len(sequence);
  if (ndim == 0)
  {
    throw py::value_error("from_cores() requires at least one core");
  }

  std::vector<py::array> arrays;
  arrays.reserve(ndim);

  std::size_t expected_left_rank = 1;
  for (std::size_t d = 0; d < ndim; d++)
  {
    py::array core = py::array::ensure(sequence[d]);
    if (!core)
    {
      throw py::type_error(
        "core " + std::to_string(d) + " is not an array or array-like object");
    }

    if (core.ndim() != 3)
    {
      throw py::value_error(
        "core " + std::to_string(d) + " must have 3 axes (left_rank, mode_size, "
        "right_rank); got " + std::to_string(core.ndim()));
    }

    for (py::ssize_t axis = 0; axis < 3; axis++)
    {
      if (core.shape(axis) <= 0)
      {
        throw py::value_error(
          "core " + std::to_string(d) + " has a non-positive extent on axis " +
          std::to_string(axis));
      }
    }

    const auto left_rank = static_cast<std::size_t>(core.shape(0));
    const auto right_rank = static_cast<std::size_t>(core.shape(2));

    if (left_rank != expected_left_rank)
    {
      if (d == 0)
      {
        throw py::value_error(
          "the first core must have left rank 1; got " + std::to_string(left_rank));
      }
      throw py::value_error(
        "interface rank mismatch between cores " + std::to_string(d - 1) + " and " +
        std::to_string(d) + ": right rank " + std::to_string(expected_left_rank) +
        " does not match left rank " + std::to_string(left_rank));
    }

    if (d + 1 == ndim && right_rank != 1)
    {
      throw py::value_error(
        "the last core must have right rank 1; got " + std::to_string(right_rank));
    }

    expected_left_rank = right_rank;
    arrays.push_back(std::move(core));
  }

  return arrays;
}

/// Resolves the scalar type to build with, refusing to guess when the cores disagree.
ScalarKind resolve_kind(std::vector<py::array> const& cores, py::object const& dtype_object)
{
  if (!dtype_object.is_none())
  {
    return scalar_kind_from_dtype(dtype_object);
  }

  const int first = cores.front().dtype().num();
  for (std::size_t d = 1; d < cores.size(); d++)
  {
    if (cores[d].dtype().num() != first)
    {
      throw py::type_error(
        "cores have mixed dtypes; pass dtype= explicitly to choose one");
    }
  }

  if (first == py::dtype::of<float>().num())
  {
    return ScalarKind::Float32;
  }
  if (first == py::dtype::of<double>().num())
  {
    return ScalarKind::Float64;
  }

  // Same strictness as compress(): an implicit conversion would silently change the
  // accuracy of every later operation on this train.
  throw py::type_error(
    "from_cores() supports float32 and float64 cores; got dtype " +
    py::str(cores.front().dtype()).cast<std::string>() +
    ". Pass dtype= explicitly to convert.");
}

/// Copies validated cores into BoBa-owned storage for one scalar type.
template <typename data_t>
PyTensorTrain build_typed(std::vector<py::array> const& cores)
{
  using core_type = typename RuntimeTensorTrain<data_t>::core_type;

  std::vector<core_type> built(cores.size());
  for (std::size_t d = 0; d < cores.size(); d++)
  {
    // NumPy performs the reordering copy into BoBa's first-index-fastest layout.
    auto fortran_core = as_fortran_order<data_t>(cores[d]);

    const auto left_rank = static_cast<std::size_t>(fortran_core.shape(0));
    const auto mode_size = static_cast<std::size_t>(fortran_core.shape(1));
    const auto right_rank = static_cast<std::size_t>(fortran_core.shape(2));

    built[d] = core_type({left_rank, mode_size, right_rank});
    std::memcpy(
      built[d].data(),
      fortran_core.data(),
      left_rank * mode_size * right_rank * sizeof(data_t));
  }

  return PyTensorTrain(RuntimeTensorTrain<data_t>(std::move(built)));
}

PyTensorTrain from_cores(py::object const& cores_like, py::object const& dtype_object)
{
  auto cores = parse_cores(cores_like);
  const ScalarKind kind = resolve_kind(cores, dtype_object);

  if (kind == ScalarKind::Float32)
  {
    return build_typed<float>(cores);
  }
  return build_typed<double>(cores);
}

} // namespace

void register_construct(py::module_& module)
{
  module.def(
    "from_cores",
    &from_cores,
    py::arg("cores"),
    py::kw_only(),
    py::arg("dtype") = py::none(),
    R"doc(
Build a tensor train directly from its cores.

The inverse of :attr:`TensorTrain.cores`: ``from_cores(t.cores)`` reproduces ``t``.
Nothing is decompressed or recompressed, so the ranks of the result are exactly the
ranks implied by the cores.

Parameters
----------
cores : sequence of numpy.ndarray
    One array per mode, shaped ``(left_rank, mode_size, right_rank)``. Interface ranks
    must chain -- each core's right rank equals the next core's left rank -- and the
    boundary ranks must both be 1. C-contiguous, Fortran-contiguous and strided arrays
    are all accepted and copied into BoBa-owned memory.
dtype : numpy dtype specifier or None, optional
    Scalar type to build with, ``float32`` or ``float64``. ``None`` (the default) takes
    the dtype shared by the cores, and raises if they disagree.

Returns
-------
TensorTrain
    A train of ``len(cores)`` modes.

Raises
------
ValueError
    If a core does not have 3 axes, has a non-positive extent, breaks the interface
    rank chain, or has a non-unit boundary rank.
TypeError
    If the cores have mixed dtypes and no explicit ``dtype`` was given, or the dtype is
    not supported.
)doc");
}

} // namespace boba_python
