// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "py_tensor_train.hpp"

#include "detail/errors.hpp"
#include "detail/numpy_interop.hpp"
#include "detail/runtime_tensor_train.hpp"

#include <pybind11/numpy.h>

#include <limits>
#include <string>
#include <vector>

namespace boba_python
{

namespace
{

/**
 * \brief Runs the runtime TT-SVD for one scalar type.
 *
 * The NumPy array is converted to Fortran order (a copy when the input is C-contiguous
 * or non-contiguous) so its payload matches BoBa's first-index-fastest storage, then
 * handed to the runtime compression routine.
 */
template <typename data_t>
PyTensorTrain compress_typed(
  py::array const& array,
  std::vector<std::size_t> const& shape,
  double relative_tolerance,
  double absolute_tolerance,
  std::size_t max_rank)
{
  auto fortran_array = as_fortran_order<data_t>(array);
  data_t const* payload = fortran_array.data();

  // The decomposition itself never touches Python.
  py::gil_scoped_release release;
  return PyTensorTrain(
    compress_dense<data_t>(payload, shape, relative_tolerance, absolute_tolerance, max_rank));
}

PyTensorTrain compress(
  py::object const& array_like,
  double relative_tolerance,
  double absolute_tolerance,
  py::object const& max_rank_object)
{
  py::array array = py::array::ensure(array_like);
  if (!array)
  {
    throw py::type_error("compress() expects a NumPy array or array-like object");
  }

  if (array.ndim() < 1)
  {
    throw py::value_error("compress() requires an array with at least one dimension");
  }

  const auto dtype_kind = array.dtype().num();
  ScalarKind kind;
  if (dtype_kind == py::dtype::of<float>().num())
  {
    kind = ScalarKind::Float32;
  }
  else if (dtype_kind == py::dtype::of<double>().num())
  {
    kind = ScalarKind::Float64;
  }
  else
  {
    // Deliberately strict: an implicit widening or narrowing of the user's data would
    // silently change the accuracy of the result.
    throw py::type_error(
      "compress() supports float32 and float64 arrays; got dtype " +
      py::str(array.dtype()).cast<std::string>() +
      ". Convert explicitly, e.g. numpy.asarray(a, dtype=numpy.float64).");
  }

  std::vector<std::size_t> shape(static_cast<std::size_t>(array.ndim()));
  for (std::size_t d = 0; d < shape.size(); d++)
  {
    const auto extent = array.shape(static_cast<py::ssize_t>(d));
    if (extent <= 0)
    {
      throw py::value_error(
        "compress() requires positive extents; axis " + std::to_string(d) + " has size " +
        std::to_string(static_cast<long long>(extent)));
    }
    shape[d] = static_cast<std::size_t>(extent);
  }
  (void)checked_product(shape, "array");

  if (relative_tolerance < 0.0 || absolute_tolerance < 0.0)
  {
    throw py::value_error("rtol and atol must be non-negative");
  }

  std::size_t max_rank = std::numeric_limits<std::size_t>::max();
  if (!max_rank_object.is_none())
  {
    const long long requested = max_rank_object.cast<long long>();
    if (requested <= 0)
    {
      throw py::value_error("max_rank must be a positive integer or None");
    }
    max_rank = static_cast<std::size_t>(requested);
  }

  if (kind == ScalarKind::Float32)
  {
    return compress_typed<float>(array, shape, relative_tolerance, absolute_tolerance, max_rank);
  }
  return compress_typed<double>(array, shape, relative_tolerance, absolute_tolerance, max_rank);
}

} // namespace

void register_compress(py::module_& module)
{
  module.def(
    "compress",
    &compress,
    py::arg("array"),
    py::kw_only(),
    py::arg("rtol") = 1.0e-12,
    py::arg("atol") = 1.0e-12,
    py::arg("max_rank") = py::none(),
    R"doc(
Compress a dense NumPy array into a tensor train.

Runs BoBa's TT-SVD: a sequence of truncated SVDs over successive unfoldings, using
BoBa's own SVD, fold and matrix primitives. The tensor dimension is taken from the
array at run time and is not bounded by the build.

Parameters
----------
array : numpy.ndarray
    Input tensor. Must be ``float32`` or ``float64``; other dtypes are rejected rather
    than converted. C-contiguous, Fortran-contiguous and strided arrays are all
    accepted, and are copied into BoBa-owned memory in BoBa's storage order.
rtol : float, optional
    Relative singular-value threshold, mapped to BoBa's ``svd_tolerance_relative``.
    Defaults to the native ``1e-12``.
atol : float, optional
    Absolute singular-value threshold, mapped to BoBa's ``svd_tolerance_absolute``.
    Defaults to the native ``1e-12``.
max_rank : int or None, optional
    Upper bound on every interface rank, mapped to BoBa's
    ``max_kept_singular_values``. ``None`` (the default) keeps all significant values.

Returns
-------
TensorTrain
    A train whose dtype matches the input array.
)doc");
}

} // namespace boba_python
