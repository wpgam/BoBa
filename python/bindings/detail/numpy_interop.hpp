// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "errors.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <limits>
#include <vector>

/**
 * \file
 * \brief NumPy conversion helpers.
 *
 * Layout note. BoBa stores tensors with the first index varying fastest, the same
 * convention NumPy calls Fortran order. NumPy arrays are usually C-contiguous, so a
 * transposing copy is required in both directions. Rather than reinterpreting strides
 * by hand, conversions go through NumPy itself: incoming arrays are requested as
 * F-contiguous (which copies when needed, and also handles non-contiguous inputs), and
 * outgoing arrays are constructed with explicit Fortran strides.
 *
 * Ownership in round one: every conversion copies. NumPy never aliases BoBa-owned
 * memory and BoBa never aliases NumPy-owned memory, so no returned array can dangle and
 * no Python-side write can break a tensor-train invariant.
 */

namespace boba_python
{

namespace py = pybind11;

/// Fortran-order byte strides for \p shape.
[[nodiscard]] inline std::vector<py::ssize_t> fortran_strides(
  std::vector<std::size_t> const& shape,
  std::size_t item_size)
{
  std::vector<py::ssize_t> strides(shape.size());
  std::size_t running = item_size;
  for (std::size_t d = 0; d < shape.size(); d++)
  {
    strides[d] = static_cast<py::ssize_t>(running);
    running *= shape[d];
  }
  return strides;
}

/**
 * \brief Copies a BoBa-owned column-major payload into a fresh NumPy array.
 *
 * The result owns its memory; the caller may free \p data immediately afterwards.
 */
template <typename data_t>
[[nodiscard]] py::array copy_to_numpy(
  data_t const* data,
  std::vector<std::size_t> const& shape)
{
  std::vector<py::ssize_t> numpy_shape(shape.size());
  for (std::size_t d = 0; d < shape.size(); d++)
  {
    numpy_shape[d] = static_cast<py::ssize_t>(shape[d]);
  }

  // pybind11 copies when no base handle is supplied.
  return py::array_t<data_t>(numpy_shape, fortran_strides(shape, sizeof(data_t)), data);
}

/**
 * \brief Returns \p input as an F-contiguous array of \p data_t.
 *
 * NumPy performs the reordering copy, so no stride arithmetic is assumed here. The
 * caller must already have checked that the dtype is exactly \p data_t; `forcecast` is
 * present only so layout conversion is allowed.
 */
template <typename data_t>
[[nodiscard]] py::array_t<data_t, py::array::f_style | py::array::forcecast> as_fortran_order(
  py::array const& input)
{
  return py::array_t<data_t, py::array::f_style | py::array::forcecast>(input);
}

/**
 * \brief Multiplies extents, saturating instead of overflowing.
 *
 * Used for mathematically valid TT rank bounds, where the exact product is irrelevant
 * once it exceeds any representable rank.
 */
[[nodiscard]] inline std::size_t saturating_product(
  std::vector<std::size_t> const& extents,
  std::size_t begin,
  std::size_t end)
{
  constexpr std::size_t ceiling = std::numeric_limits<std::size_t>::max();
  std::size_t total = 1;
  for (std::size_t d = begin; d < end; d++)
  {
    if (extents[d] != 0 && total > ceiling / extents[d])
    {
      return ceiling;
    }
    total *= extents[d];
  }
  return total;
}

} // namespace boba_python
