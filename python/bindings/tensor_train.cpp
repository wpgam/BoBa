// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "py_tensor_train.hpp"

#include "detail/errors.hpp"
#include "detail/numpy_interop.hpp"
#include "detail/runtime_rounding.hpp"

#include <pybind11/numpy.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace boba_python
{

namespace
{

/// Formats a list of extents or ranks the way Python prints a tuple.
std::string format_tuple(std::vector<std::size_t> const& values)
{
  std::ostringstream stream;
  stream << "(";
  for (std::size_t i = 0; i < values.size(); i++)
  {
    if (i > 0)
    {
      stream << ", ";
    }
    stream << values[i];
  }
  if (values.size() == 1)
  {
    stream << ",";
  }
  stream << ")";
  return stream.str();
}

py::tuple to_tuple(std::vector<std::size_t> const& values)
{
  py::tuple result(values.size());
  for (std::size_t i = 0; i < values.size(); i++)
  {
    result[i] = py::cast(values[i]);
  }
  return result;
}

/// Reads one index from a `__getitem__` key, rejecting slices and negatives.
std::size_t parse_index(py::handle item, std::size_t axis, std::size_t extent)
{
  if (py::isinstance<py::slice>(item))
  {
    throw py::type_error("TensorTrain indexing requires a complete scalar multi-index; "
                         "slicing is not supported");
  }

  if (!py::isinstance<py::int_>(item))
  {
    // NumPy integer scalars expose __index__, so accept anything integral.
    if (!PyIndex_Check(item.ptr()))
    {
      throw py::type_error("TensorTrain indices must be integers");
    }
  }

  const long long value = item.cast<long long>();
  if (value < 0)
  {
    throw py::index_error(
      "TensorTrain uses zero-based, non-negative indexing; got " + std::to_string(value) +
      " for axis " + std::to_string(axis));
  }

  const auto index = static_cast<std::size_t>(value);
  if (index >= extent)
  {
    throw py::index_error(
      "index " + std::to_string(index) + " is out of bounds for axis " +
      std::to_string(axis) + " with size " + std::to_string(extent));
  }
  return index;
}

/**
 * \brief Expands the `max_rank` argument into one bound per interface.
 *
 * The result is indexed the way `boba::TensorTrain::max_ranks` is: entry `d` bounds the
 * interface between cores `d - 1` and `d`. That gives `ndim + 1` entries, lining up
 * element for element with `TensorTrain.ranks`, which is the sequence a user reads to
 * decide on bounds in the first place.
 */
std::vector<std::size_t> parse_max_ranks(py::object const& max_rank_object, std::size_t ndim)
{
  std::vector<std::size_t> bounds(ndim + 1, std::numeric_limits<std::size_t>::max());

  if (max_rank_object.is_none())
  {
    return bounds;
  }

  if (PyIndex_Check(max_rank_object.ptr()))
  {
    const long long requested = max_rank_object.cast<long long>();
    if (requested <= 0)
    {
      throw py::value_error("max_rank must be a positive integer, a sequence, or None");
    }
    std::fill(bounds.begin(), bounds.end(), static_cast<std::size_t>(requested));
    return bounds;
  }

  if (!py::isinstance<py::sequence>(max_rank_object) || py::isinstance<py::str>(max_rank_object))
  {
    throw py::type_error(
      "max_rank must be a positive integer, a sequence of them, or None");
  }
  auto sequence = py::reinterpret_borrow<py::sequence>(max_rank_object);

  if (py::len(sequence) != ndim + 1)
  {
    throw py::value_error(
      "a sequence max_rank must have ndim + 1 = " + std::to_string(ndim + 1) +
      " entries, matching TensorTrain.ranks; got " + std::to_string(py::len(sequence)));
  }

  for (std::size_t d = 0; d <= ndim; d++)
  {
    py::handle item = sequence[d];
    if (!PyIndex_Check(item.ptr()))
    {
      throw py::type_error("max_rank entries must be integers");
    }
    const long long requested = item.cast<long long>();
    if (requested <= 0)
    {
      throw py::value_error(
        "max_rank entries must be positive; got " + std::to_string(requested) +
        " at position " + std::to_string(d));
    }
    bounds[d] = static_cast<std::size_t>(requested);
  }

  return bounds;
}

} // namespace

py::object dtype_object(ScalarKind kind)
{
  return (kind == ScalarKind::Float32) ? py::dtype::of<float>() : py::dtype::of<double>();
}

ScalarKind scalar_kind_from_dtype(py::handle dtype_like)
{
  py::dtype requested;
  try
  {
    requested = py::dtype::from_args(py::reinterpret_borrow<py::object>(dtype_like));
  }
  catch (py::error_already_set&)
  {
    throw py::type_error("dtype must be a NumPy dtype specifier");
  }

  if (requested.num() == py::dtype::of<float>().num())
  {
    return ScalarKind::Float32;
  }
  if (requested.num() == py::dtype::of<double>().num())
  {
    return ScalarKind::Float64;
  }

  throw py::value_error(
    "unsupported dtype " + py::str(requested).cast<std::string>() +
    "; pyboba supports float32 and float64");
}

std::vector<std::size_t> parse_shape(py::handle shape_like)
{
  if (!py::isinstance<py::sequence>(shape_like) || py::isinstance<py::str>(shape_like))
  {
    throw py::type_error("shape must be a sequence of positive integers");
  }
  auto sequence = py::reinterpret_borrow<py::sequence>(shape_like);

  const std::size_t ndim = py::len(sequence);
  if (ndim == 0)
  {
    throw py::value_error("shape must have at least one mode");
  }

  std::vector<std::size_t> extents(ndim);
  for (std::size_t d = 0; d < ndim; d++)
  {
    py::handle item = sequence[d];
    if (!PyIndex_Check(item.ptr()))
    {
      throw py::type_error("shape entries must be integers");
    }
    const long long value = item.cast<long long>();
    if (value <= 0)
    {
      throw py::value_error(
        "shape entries must be positive; got " + std::to_string(value) + " at position " +
        std::to_string(d));
    }
    extents[d] = static_cast<std::size_t>(value);
  }

  // Raises MemoryError rather than silently wrapping around.
  (void)checked_product(extents, "shape");
  return extents;
}

py::object PyTensorTrain::dtype() const
{
  return dtype_object(scalar_kind());
}

std::size_t PyTensorTrain::ndim() const
{
  return std::visit([](auto const& train)
  {
    return train.ndim();
  },
                    m_storage);
}

py::tuple PyTensorTrain::shape() const
{
  return to_tuple(std::visit([](auto const& train)
  {
    return train.shape();
  },
                             m_storage));
}

py::tuple PyTensorTrain::ranks() const
{
  return to_tuple(std::visit([](auto const& train)
  {
    return train.ranks();
  },
                             m_storage));
}

py::list PyTensorTrain::cores() const
{
  py::list result;
  std::visit([&result](auto const& train)
  {
    for (std::size_t d = 0; d < train.ndim(); d++)
    {
      auto const& core = train.core(d);
      std::vector<std::size_t> core_shape{core.sizes(0), core.sizes(1), core.sizes(2)};
      result.append(copy_to_numpy(core.const_data(), core_shape));
    }
  },
             m_storage);
  return result;
}

py::array PyTensorTrain::to_numpy() const
{
  return std::visit([](auto const& train) -> py::array
  {
    auto shape = train.shape();

    // Contracting the whole train can take a while and touches no Python state, so the
    // GIL is dropped for it. The resulting payload is already column-major over `shape`,
    // which is exactly what copy_to_numpy expects.
    typename std::decay_t<decltype(train)>::core_type flat;
    {
      py::gil_scoped_release release;
      flat = train.reconstruct();
    }

    return copy_to_numpy(flat.const_data(), shape);
  },
                    m_storage);
}

py::object PyTensorTrain::getitem(py::object const& key) const
{
  const std::size_t dimension = ndim();

  py::tuple index_tuple;
  if (py::isinstance<py::tuple>(key))
  {
    index_tuple = py::reinterpret_borrow<py::tuple>(key);
  }
  else
  {
    index_tuple = py::make_tuple(key);
  }

  if (py::len(index_tuple) != dimension)
  {
    throw py::index_error(
      "TensorTrain requires a complete multi-index of length " + std::to_string(dimension) +
      "; got " + std::to_string(py::len(index_tuple)));
  }

  std::vector<std::size_t> indices(dimension);
  auto extents = std::visit([](auto const& train)
  {
    return train.shape();
  },
                            m_storage);

  for (std::size_t d = 0; d < dimension; d++)
  {
    indices[d] = parse_index(index_tuple[d], d, extents[d]);
  }

  return std::visit([&indices](auto const& train) -> py::object
  {
    return py::cast(train.entry(indices));
  },
                    m_storage);
}

PyTensorTrain PyTensorTrain::orthogonalized() const
{
  return PyTensorTrain(std::visit([](auto const& train) -> storage_type
  {
    // Trains are immutable from Python, so the sweep runs on a deep copy. BoBa's Tensor
    // copy constructor allocates and copies, so no core is shared with the original.
    auto working_copy = train;
    {
      py::gil_scoped_release release;
      orthogonalize(working_copy);
    }
    return storage_type(std::move(working_copy));
  },
                                  m_storage));
}

PyTensorTrain PyTensorTrain::rounded(
  double relative_tolerance,
  double absolute_tolerance,
  py::object const& max_rank_object) const
{
  if (relative_tolerance < 0.0 || absolute_tolerance < 0.0)
  {
    throw py::value_error("rtol and atol must be non-negative");
  }

  auto bounds = parse_max_ranks(max_rank_object, ndim());

  return PyTensorTrain(std::visit(
    [relative_tolerance, absolute_tolerance, &bounds](auto const& train) -> storage_type
  {
    auto working_copy = train;
    {
      py::gil_scoped_release release;
      round(working_copy, relative_tolerance, absolute_tolerance, bounds);
    }
    return storage_type(std::move(working_copy));
  },
    m_storage));
}

std::string PyTensorTrain::repr() const
{
  auto extents = std::visit([](auto const& train)
  {
    return train.shape();
  },
                            m_storage);
  auto interface_ranks = std::visit([](auto const& train)
  {
    return train.ranks();
  },
                                    m_storage);

  const std::string dtype_name = (scalar_kind() == ScalarKind::Float32) ? "float32" : "float64";

  return "TensorTrain(shape=" + format_tuple(extents) + ", ranks=" +
         format_tuple(interface_ranks) + ", dtype=" + dtype_name + ")";
}

void register_tensor_train(py::module_& module)
{
  py::class_<PyTensorTrain>(module, "TensorTrain", R"doc(
A tensor train of arbitrary runtime dimension.

Instances are produced by :func:`pyboba.compress` and :func:`pyboba.cross`; there is no
public constructor in this first release.

Storage is owned by BoBa. Every array handed back to Python -- ``cores`` and
``to_numpy()`` -- is a fresh copy, so nothing can dangle and Python cannot corrupt the
train's rank or shape invariants.
)doc")
    .def_property_readonly("ndim", &PyTensorTrain::ndim,
                           "Number of tensor modes.")
    .def_property_readonly("shape", &PyTensorTrain::shape,
                           "Mode extents as a tuple of length ``ndim``.")
    .def_property_readonly("ranks", &PyTensorTrain::ranks,
                           "All ``ndim + 1`` interface ranks, including the unit boundary "
                           "ranks: ``(1, r1, ..., r_{d-1}, 1)``.")
    .def_property_readonly("dtype", &PyTensorTrain::dtype,
                           "NumPy dtype of the stored scalars: ``float32`` or ``float64``.")
    .def_property_readonly("cores", &PyTensorTrain::cores,
                           "List of ``ndim`` NumPy arrays shaped "
                           "``(left_rank, mode_size, right_rank)``. Copies, not views.")
    .def("to_numpy", &PyTensorTrain::to_numpy,
         "Reconstruct the dense tensor as a new NumPy array of shape ``shape``.\n\n"
         "Raises MemoryError when the dense size cannot be represented.")
    .def("__getitem__", &PyTensorTrain::getitem,
         "Evaluate a single entry from a complete zero-based multi-index.\n\n"
         "Contracts only the selected core slices; the dense tensor is never formed.")
    .def("orthogonalize", &PyTensorTrain::orthogonalized,
         R"doc(
Return a left-orthogonalized copy of this train.

Runs BoBa's left-to-right QR sweep, after which every core but the last is
left-orthogonal. The represented tensor is unchanged up to floating-point roundoff, and
so are the ranks; only the way the tensor is factored changes.

Returns
-------
TensorTrain
    A new train. This object is not modified.
)doc")
    .def("round", &PyTensorTrain::rounded,
         py::kw_only(),
         py::arg("rtol") = 1.0e-12,
         py::arg("atol") = 1.0e-12,
         py::arg("max_rank") = py::none(),
         R"doc(
Return a recompressed copy of this train with truncated interface ranks.

Runs BoBa's rounding: a left-to-right QR sweep followed by a right-to-left truncated SVD
sweep. This is the operation that undoes the rank growth of exact arithmetic -- addition
and Hadamard products enlarge ranks without adding information, and rounding removes the
excess at a controlled accuracy. The dense tensor is never formed.

Parameters
----------
rtol : float, optional
    Relative singular-value threshold, mapped to BoBa's ``svd_tolerance_relative``.
    Defaults to the native ``1e-12``.
atol : float, optional
    Absolute singular-value threshold, mapped to BoBa's ``svd_tolerance_absolute``.
    Defaults to the native ``1e-12``.
max_rank : int, sequence of int, or None, optional
    Upper bound on the retained interface ranks. An integer bounds every interface. A
    sequence must have ``ndim + 1`` entries, matching the layout of
    :attr:`TensorTrain.ranks`; its two boundary entries are ignored, since boundary
    ranks are always 1. ``None`` (the default) applies no rank bound, leaving ``rtol``
    and ``atol`` in control.

Returns
-------
TensorTrain
    A new train. This object is not modified.

Notes
-----
A one-mode train has no interface to truncate and is returned unchanged.
)doc")
    .def("__len__", [](PyTensorTrain const& train)
    {
      return train.shape()[0].cast<std::size_t>();
    })
    .def("__repr__", &PyTensorTrain::repr);
}

} // namespace boba_python
