// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "errors.hpp"
#include "evaluator.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

/**
 * \file
 * \brief Host-side Python evaluators for the runtime cross sweep.
 *
 * GIL contract for this file:
 *  - The cross sweep runs with the GIL released, so no Python object may be touched
 *    outside `evaluate()`.
 *  - `evaluate()` acquires the GIL once per batch, not once per sampled entry.
 *  - A failing callback must not let a `pybind11::error_already_set` unwind through the
 *    GIL-released numerical code. The failure is stored here while the GIL is still
 *    held and re-raised by `rethrow_if_failed()` after the sweep has unwound, which the
 *    binding layer calls with the GIL held.
 *  - Nothing in this file may be captured by a device lambda; these evaluators are
 *    explicitly host-side.
 */

namespace boba_python
{

namespace py = pybind11;

/**
 * \brief Common failure plumbing for evaluators that call into Python.
 */
template <typename data_t>
class PythonEvaluator : public Evaluator<data_t>
{
public:
  /**
   * \brief Re-raises a stored callback failure as the original Python exception.
   *
   * \pre The GIL is held.
   */
  void rethrow_if_failed()
  {
    if (m_error.has_value())
    {
      m_error->restore();
      m_error.reset();
      throw py::error_already_set();
    }
  }

protected:
  /// Stores a live Python exception. \pre The GIL is held.
  [[noreturn]] void abort_with(py::error_already_set& error)
  {
    m_error.emplace(std::move(error));
    throw PythonCallbackError{};
  }

  /// Raises a new Python exception from the evaluator itself. \pre The GIL is held.
  [[noreturn]] void abort_with_new(PyObject* exception_type, std::string const& message)
  {
    PyErr_SetString(exception_type, message.c_str());
    py::error_already_set captured;
    m_error.emplace(std::move(captured));
    throw PythonCallbackError{};
  }

  /**
   * \brief Validates a batched callback result and copies it into \p values.
   *
   * Shared by every evaluator whose callback returns one value per requested entry.
   * \p description names the callback in any error message, so each caller keeps its
   * own wording.
   *
   * \pre The GIL is held.
   */
  void store_batch_result(
    py::object const& result,
    std::size_t batch_count,
    data_t* values,
    char const* description)
  {
    py::array result_array = py::array::ensure(result);
    if (!result_array)
    {
      this->abort_with_new(
        PyExc_TypeError,
        std::string(description) + " must return an array-like of values");
    }

    const char result_kind = result_array.dtype().kind();
    if (result_kind != 'f' && result_kind != 'i' && result_kind != 'u' && result_kind != 'b')
    {
      this->abort_with_new(
        PyExc_TypeError,
        std::string(description) + " returned a non-real dtype; "
        "only real floating and integer results are supported");
    }

    if (static_cast<std::size_t>(result_array.size()) != batch_count)
    {
      this->abort_with_new(
        PyExc_ValueError,
        std::string(description) + " returned " +
          std::to_string(static_cast<long long>(result_array.size())) +
          " values but " + std::to_string(static_cast<long long>(batch_count)) +
          " were requested; expected shape (" +
          std::to_string(static_cast<long long>(batch_count)) + ",)");
    }

    py::array_t<data_t, py::array::c_style | py::array::forcecast> values_array(result_array);
    std::memcpy(values, values_array.data(), batch_count * sizeof(data_t));
  }

private:
  std::optional<py::error_already_set> m_error;
};

/**
 * \brief Calls a Python function once per requested entry with a tuple index.
 *
 * The callback signature is `f(index) -> float`, where `index` is a tuple of
 * zero-based Python integers whose length is the tensor dimension.
 */
template <typename data_t>
class ScalarPythonEvaluator final : public PythonEvaluator<data_t>
{
public:
  explicit ScalarPythonEvaluator(py::object function)
      : m_function(std::move(function))
  {
  }

  void evaluate(
    std::size_t const* indices,
    std::size_t batch_count,
    std::size_t ndim,
    data_t* values) override
  {
    py::gil_scoped_acquire gil;

    try
    {
      for (std::size_t b = 0; b < batch_count; b++)
      {
        py::tuple index(ndim);
        for (std::size_t d = 0; d < ndim; d++)
        {
          index[d] = py::int_(indices[b * ndim + d]);
        }

        py::object result = m_function(index);
        values[b] = to_scalar(result);
      }
    }
    catch (py::error_already_set& error)
    {
      this->abort_with(error);
    }
  }

  /// Scalar callbacks gain nothing from large batches, but one GIL acquire per chunk helps.
  [[nodiscard]] std::size_t preferred_batch_size() const override
  {
    return 1u << 12;
  }

private:
  /**
   * \brief Converts one callback result to \p data_t. \pre The GIL is held.
   *
   * Strings and other non-numbers are rejected outright rather than parsed, so a
   * callback that accidentally returns text fails loudly instead of succeeding for
   * numeric-looking strings. `PyFloat_AsDouble` then rejects complex results, which
   * round one does not support.
   */
  data_t to_scalar(py::handle result)
  {
    if (py::isinstance<py::str>(result) || py::isinstance<py::bytes>(result) ||
        PyNumber_Check(result.ptr()) == 0)
    {
      this->abort_with_new(
        PyExc_TypeError,
        std::string("scalar cross callback must return a real number; got ") +
          py::str(py::type::handle_of(result)).cast<std::string>());
    }

    const double value = PyFloat_AsDouble(result.ptr());
    if (value == -1.0 && PyErr_Occurred() != nullptr)
    {
      throw py::error_already_set();
    }
    return static_cast<data_t>(value);
  }

  py::object m_function;
};

/**
 * \brief Calls a Python function once per batch with an `(N, ndim)` index array.
 *
 * The callback signature is `f(indices) -> array_like` where `indices` is a C-contiguous
 * `int64` array of shape `(N, ndim)` and the result holds `N` real values.
 */
template <typename data_t>
class VectorizedPythonEvaluator final : public PythonEvaluator<data_t>
{
public:
  explicit VectorizedPythonEvaluator(py::object function)
      : m_function(std::move(function))
  {
  }

  void evaluate(
    std::size_t const* indices,
    std::size_t batch_count,
    std::size_t ndim,
    data_t* values) override
  {
    py::gil_scoped_acquire gil;

    try
    {
      py::array_t<std::int64_t> index_array(
        {static_cast<py::ssize_t>(batch_count), static_cast<py::ssize_t>(ndim)});
      auto index_writer = index_array.mutable_unchecked<2>();
      for (std::size_t b = 0; b < batch_count; b++)
      {
        for (std::size_t d = 0; d < ndim; d++)
        {
          index_writer(static_cast<py::ssize_t>(b), static_cast<py::ssize_t>(d)) =
            static_cast<std::int64_t>(indices[b * ndim + d]);
        }
      }

      py::object result = m_function(index_array);
      this->store_batch_result(result, batch_count, values, "vectorized cross callback");
    }
    catch (py::error_already_set& error)
    {
      this->abort_with(error);
    }
  }

  /**
   * \brief Cap on entries per callback invocation.
   *
   * Large enough that a whole score-tensor block normally arrives in one call, small
   * enough that the staging buffers stay bounded when interface ranks grow.
   */
  [[nodiscard]] std::size_t preferred_batch_size() const override
  {
    return 1u << 20;
  }

private:
  py::object m_function;
};

} // namespace boba_python
