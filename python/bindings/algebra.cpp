// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "py_tensor_train.hpp"

#include "detail/errors.hpp"
#include "detail/runtime_algebra.hpp"

#include <pybind11/pybind11.h>

#include <string>
#include <type_traits>

/**
 * \file
 * \brief Exact tensor-train arithmetic as seen from Python.
 *
 * None of these operations truncate. Addition gives interface ranks `r_a + r_b` and the
 * Hadamard product gives `r_a * r_b`, so a chain of them grows ranks quickly; call
 * `TensorTrain.round()` when the growth stops paying for itself.
 */

namespace boba_python
{

namespace
{

/// Formats a shape the way Python prints a tuple, for error messages.
std::string describe_shape(py::tuple const& shape)
{
  return py::str(shape).cast<std::string>();
}

/**
 * \brief Rejects operand pairs that cannot be combined entrywise.
 *
 * Both trains must carry the same scalar type and represent tensors of the same shape.
 * Neither is converted implicitly: a silent dtype promotion would change the accuracy of
 * the result, and a silent reshape has no sensible meaning.
 */
void require_compatible(PyTensorTrain const& a, PyTensorTrain const& b, char const* operation)
{
  if (a.scalar_kind() != b.scalar_kind())
  {
    throw py::type_error(
      std::string(operation) + " requires both trains to have the same dtype; got " +
      py::str(a.dtype()).cast<std::string>() + " and " +
      py::str(b.dtype()).cast<std::string>() +
      ". Rebuild one with pyboba.from_cores(t.cores, dtype=...).");
  }

  if (!a.shape().equal(b.shape()))
  {
    throw py::value_error(
      std::string(operation) + " requires both trains to have the same shape; got " +
      describe_shape(a.shape()) + " and " + describe_shape(b.shape()));
  }
}

/// Applies a two-train runtime operation once the operands are known to be compatible.
template <typename Function>
PyTensorTrain combine(PyTensorTrain const& a, PyTensorTrain const& b, Function&& function)
{
  return PyTensorTrain(std::visit(
    [&function](auto const& x, auto const& y) -> PyTensorTrain::storage_type
  {
    if constexpr (std::is_same_v<std::decay_t<decltype(x)>, std::decay_t<decltype(y)>>)
    {
      // The arithmetic touches no Python state.
      py::gil_scoped_release release;
      return PyTensorTrain::storage_type(function(x, y));
    }
    else
    {
      // Unreachable: require_compatible has already matched the scalar kinds. The branch
      // exists because std::visit instantiates every combination.
      throw py::type_error("mismatched scalar types");
    }
  },
    a.storage(), b.storage()));
}

/// Reduces a two-train runtime operation to a scalar.
template <typename Function>
double reduce(PyTensorTrain const& a, PyTensorTrain const& b, Function&& function)
{
  return std::visit([&function](auto const& x, auto const& y) -> double
  {
    if constexpr (std::is_same_v<std::decay_t<decltype(x)>, std::decay_t<decltype(y)>>)
    {
      py::gil_scoped_release release;
      return static_cast<double>(function(x, y));
    }
    else
    {
      throw py::type_error("mismatched scalar types");
    }
  },
                    a.storage(), b.storage());
}

PyTensorTrain add_trains(PyTensorTrain const& a, PyTensorTrain const& b)
{
  require_compatible(a, b, "addition");
  return combine(a, b, [](auto const& x, auto const& y)
  {
    return add(x, y);
  });
}

PyTensorTrain scale_train(PyTensorTrain const& a, double scalar)
{
  return PyTensorTrain(std::visit([scalar](auto const& x) -> PyTensorTrain::storage_type
  {
    using data_t = typename std::decay_t<decltype(x)>::value_type;
    py::gil_scoped_release release;
    return PyTensorTrain::storage_type(scale(x, static_cast<data_t>(scalar)));
  },
                                  a.storage()));
}

PyTensorTrain subtract_trains(PyTensorTrain const& a, PyTensorTrain const& b)
{
  require_compatible(a, b, "subtraction");
  return add_trains(a, scale_train(b, -1.0));
}

PyTensorTrain hadamard_trains(PyTensorTrain const& a, PyTensorTrain const& b)
{
  require_compatible(a, b, "elementwise multiplication");
  return combine(a, b, [](auto const& x, auto const& y)
  {
    return hadamard(x, y);
  });
}

double inner_trains(PyTensorTrain const& a, PyTensorTrain const& b)
{
  require_compatible(a, b, "the inner product");
  return reduce(a, b, [](auto const& x, auto const& y)
  {
    return inner_product(x, y);
  });
}

double norm_train(PyTensorTrain const& a)
{
  return std::visit([](auto const& x) -> double
  {
    py::gil_scoped_release release;
    return static_cast<double>(norm_frobenius(x));
  },
                    a.storage());
}

/// Reads a Python number as a scalar, or reports why it is not usable as one.
double as_scalar(py::handle value, char const* operation)
{
  if (py::isinstance<PyTensorTrain>(value))
  {
    throw py::type_error(std::string(operation) + " expects a number, not a TensorTrain");
  }

  try
  {
    return value.cast<double>();
  }
  catch (py::cast_error const&)
  {
    throw py::type_error(
      std::string(operation) + " expects a real number; got " +
      py::str(py::type::handle_of(value)).cast<std::string>());
  }
}

py::object multiply(PyTensorTrain const& self, py::object const& other)
{
  if (py::isinstance<PyTensorTrain>(other))
  {
    return py::cast(hadamard_trains(self, other.cast<PyTensorTrain const&>()));
  }
  return py::cast(scale_train(self, as_scalar(other, "multiplication")));
}

double relative_error(PyTensorTrain const& approximation, PyTensorTrain const& reference)
{
  require_compatible(approximation, reference, "relative_error");

  const double reference_norm = norm_train(reference);
  if (reference_norm == 0.0)
  {
    throw py::value_error(
      "relative_error is undefined when the reference train has zero norm");
  }

  return norm_train(subtract_trains(approximation, reference)) / reference_norm;
}

} // namespace

void register_algebra(py::class_<PyTensorTrain>& tensor_train_class, py::module_& module)
{
  // Tell NumPy not to try to broadcast a scalar over this type, so that expressions like
  // numpy.float64(2) * train reach __rmul__ instead of producing an object array.
  tensor_train_class.attr("__array_ufunc__") = py::none();

  tensor_train_class
    .def("__add__", &add_trains, py::arg("other"),
         "Exact sum. Interface ranks add; call round() to truncate afterwards.")
    .def("__sub__", &subtract_trains, py::arg("other"),
         "Exact difference. Interface ranks add, as they do for a sum.")
    .def("__neg__", [](PyTensorTrain const& self)
    {
      return scale_train(self, -1.0);
    },
         "Exact negation. Ranks are unchanged.")
    .def("__mul__", &multiply, py::arg("other"))
    .def("__rmul__", [](PyTensorTrain const& self, py::object const& other)
    {
      return scale_train(self, as_scalar(other, "multiplication"));
    })
    .def("__truediv__", [](PyTensorTrain const& self, py::object const& other)
    {
      const double divisor = as_scalar(other, "division");
      if (divisor == 0.0)
      {
        throw py::value_error("division by zero");
      }
      return scale_train(self, 1.0 / divisor);
    })
    .def("hadamard", &hadamard_trains, py::arg("other"),
         R"doc(
Exact elementwise product, the same operation as ``self * other``.

Interface rank ``d`` of the result is the product of the two inputs' rank ``d``, so
repeated products without an intervening :meth:`round` grow ranks fastest of anything in
this interface. The dense tensor is never formed.
)doc")
    .def("inner", &inner_trains, py::arg("other"),
         R"doc(
Inner product of the two trains seen as flat vectors.

Contracts the cores right to left; the dense tensors are never formed.

Returns
-------
float
)doc")
    .def("norm", &norm_train,
         R"doc(
Frobenius norm of the tensor, equal to ``sqrt(self.inner(self))``.

Returns
-------
float
)doc");

  module.def(
    "relative_error",
    &relative_error,
    py::arg("approximation"),
    py::arg("reference"),
    R"doc(
Relative Frobenius error, ``norm(approximation - reference) / norm(reference)``.

Parameters
----------
approximation, reference : TensorTrain
    Trains of the same shape and dtype.

Returns
-------
float

Raises
------
ValueError
    If the shapes differ, or the reference train has zero norm.

Notes
-----
The difference is formed exactly, so its interface ranks are the sums of the inputs'
ranks and the subtraction is subject to cancellation: when the two trains nearly agree,
the leading digits cancel and the computed difference keeps only what accuracy the
inputs had to spare. The result is reliable as an error estimate down to roughly the
square root of the working precision, not to the last representable digit.
)doc");
}

} // namespace boba_python
