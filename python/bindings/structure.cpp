// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "py_tensor_train.hpp"

#include "detail/errors.hpp"
#include "detail/runtime_structure.hpp"

#include <pybind11/pybind11.h>

#include <string>
#include <type_traits>
#include <vector>

/**
 * \file
 * \brief Concatenating trains along a mode, from Python.
 *
 * Mode slicing is the other half of this pair, but it reaches Python through
 * `TensorTrain.__getitem__` in tensor_train.cpp rather than through a named function.
 */

namespace boba_python
{

namespace
{

/// Joins two compatible trains along one mode.
PyTensorTrain concatenate_pair(
  PyTensorTrain const& a,
  PyTensorTrain const& b,
  std::size_t mode)
{
  return PyTensorTrain(std::visit(
    [mode](auto const& x, auto const& y) -> PyTensorTrain::storage_type
  {
    if constexpr (std::is_same_v<std::decay_t<decltype(x)>, std::decay_t<decltype(y)>>)
    {
      py::gil_scoped_release release;
      return PyTensorTrain::storage_type(concatenate_mode(x, y, mode));
    }
    else
    {
      // Unreachable: the dtypes are checked before this runs.
      throw py::type_error("mismatched scalar types");
    }
  },
    a.storage(), b.storage()));
}

/**
 * \brief Checks that \p candidate can be joined onto \p first along \p mode.
 *
 * Every mode extent except the concatenation axis must agree, and the scalar types must
 * match exactly; neither is converted implicitly.
 */
void require_joinable(
  PyTensorTrain const& first,
  PyTensorTrain const& candidate,
  std::size_t mode,
  std::size_t position)
{
  if (first.scalar_kind() != candidate.scalar_kind())
  {
    throw py::type_error(
      "concatenate() requires every train to have the same dtype; train 0 has " +
      py::str(first.dtype()).cast<std::string>() + " and train " +
      std::to_string(position) + " has " +
      py::str(candidate.dtype()).cast<std::string>());
  }

  if (first.ndim() != candidate.ndim())
  {
    throw py::value_error(
      "concatenate() requires every train to have the same number of modes; train 0 has " +
      std::to_string(first.ndim()) + " and train " + std::to_string(position) + " has " +
      std::to_string(candidate.ndim()));
  }

  auto first_shape = first.shape();
  auto candidate_shape = candidate.shape();

  for (std::size_t d = 0; d < first.ndim(); d++)
  {
    if (d == mode)
    {
      continue;
    }
    if (!first_shape[d].equal(candidate_shape[d]))
    {
      throw py::value_error(
        "concatenate() requires matching extents on every mode except mode " +
        std::to_string(mode) + "; train 0 has " +
        py::str(first_shape).cast<std::string>() + " and train " +
        std::to_string(position) + " has " +
        py::str(candidate_shape).cast<std::string>());
    }
  }
}

PyTensorTrain concatenate(py::object const& trains_like, py::object const& mode_object)
{
  if (!py::isinstance<py::sequence>(trains_like) || py::isinstance<py::str>(trains_like))
  {
    throw py::type_error("concatenate() expects a sequence of TensorTrain objects");
  }
  auto sequence = py::reinterpret_borrow<py::sequence>(trains_like);

  const std::size_t count = py::len(sequence);
  if (count == 0)
  {
    throw py::value_error("concatenate() requires at least one train");
  }

  std::vector<PyTensorTrain const*> trains;
  trains.reserve(count);
  for (std::size_t i = 0; i < count; i++)
  {
    py::handle item = sequence[i];
    if (!py::isinstance<PyTensorTrain>(item))
    {
      throw py::type_error(
        "concatenate() expects TensorTrain objects; entry " + std::to_string(i) +
        " is " + py::str(py::type::handle_of(item)).cast<std::string>());
    }
    trains.push_back(&item.cast<PyTensorTrain const&>());
  }

  const std::size_t ndim = trains[0]->ndim();

  if (!PyIndex_Check(mode_object.ptr()))
  {
    throw py::type_error("mode must be an integer");
  }
  const long long requested_mode = mode_object.cast<long long>();
  if (requested_mode < 0 || static_cast<std::size_t>(requested_mode) >= ndim)
  {
    throw py::value_error(
      "mode " + std::to_string(requested_mode) + " is out of range for a train with " +
      std::to_string(ndim) + " modes");
  }
  const auto mode = static_cast<std::size_t>(requested_mode);

  for (std::size_t i = 1; i < count; i++)
  {
    require_joinable(*trains[0], *trains[i], mode, i);
  }

  if (count == 1)
  {
    return *trains[0];
  }

  PyTensorTrain result = concatenate_pair(*trains[0], *trains[1], mode);
  for (std::size_t i = 2; i < count; i++)
  {
    result = concatenate_pair(result, *trains[i], mode);
  }
  return result;
}

} // namespace

void register_structure(py::module_& module)
{
  module.def(
    "concatenate",
    &concatenate,
    py::arg("trains"),
    py::kw_only(),
    py::arg("mode") = 0,
    R"doc(
Join tensor trains along one mode, without reconstructing them.

The result represents the tensor you would get by concatenating the dense tensors along
``mode``, and is built purely from the cores: the operands are placed in disjoint ranges
of the concatenation axis and combined block-diagonally in the rank indices.

Parameters
----------
trains : sequence of TensorTrain
    Trains to join, in order. They must share a dtype and agree on every mode extent
    except ``mode``. A sequence of one is returned unchanged.
mode : int, optional
    The mode to join along. Defaults to 0.

Returns
-------
TensorTrain
    Shape equal to the operands' shape with ``mode`` summed over the inputs.

Notes
-----
Interface ranks are the sums of the operands' ranks, so joining many trains grows ranks
linearly in the number of operands. That growth is exact, not an artifact; call
:meth:`TensorTrain.round` afterwards if the joined train is more redundant than it looks.
)doc");
}

} // namespace boba_python
