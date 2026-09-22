// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "python_evaluator.hpp"
#include "runtime_tensor_train.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

/**
 * \file
 * \brief Entry source for cross-approximating a pointwise function of existing trains.
 *
 * The cross sweep asks for tensor entries at the multi-indices it has selected. When the
 * tensor being approximated is `f(A, B, ...)` applied entrywise to trains that already
 * exist, each requested entry is `f(A[index], B[index], ...)`, and the two halves of
 * that expression have very different costs.
 *
 * Looking up `A[index]` is a contraction of the selected core slices: pure C++, no
 * Python, and the dominant cost when the ranks are large. Applying `f` is a scalar
 * operation the user wrote in Python.
 *
 * So the two are separated. Every entry lookup for a whole batch happens first, with the
 * GIL still released, and `f` is then called exactly once for the entire batch with one
 * array of values per input train. A batch holds up to 2^20 entries, which turns what
 * would otherwise be a million GIL acquisitions and a million Python calls into one.
 */

namespace boba_python
{

/**
 * \brief Evaluates `f(A[index], B[index], ...)` in batches.
 *
 * \tparam data_t Scalar type shared by the input trains and the result.
 *
 * The callback receives one one-dimensional NumPy array per input train, all of the same
 * length, and returns one value per entry. The arrays are copies, so the callback cannot
 * disturb the sweep by writing to them.
 */
template <typename data_t>
class TensorTrainFunctionEvaluator final : public PythonEvaluator<data_t>
{
public:
  TensorTrainFunctionEvaluator(
    py::object function,
    std::vector<RuntimeTensorTrain<data_t> const*> inputs)
      : m_function(std::move(function)),
        m_inputs(std::move(inputs))
  {
  }

  void evaluate(
    std::size_t const* indices,
    std::size_t batch_count,
    std::size_t ndim,
    data_t* values) override
  {
    const std::size_t input_count = m_inputs.size();

    // Phase one: no Python whatsoever. This is the expensive half, and it runs with the
    // GIL still released by the sweep.
    m_scratch.resize(input_count * batch_count);
    for (std::size_t k = 0; k < input_count; k++)
    {
      data_t* column = m_scratch.data() + k * batch_count;
      for (std::size_t b = 0; b < batch_count; b++)
      {
        column[b] = m_inputs[k]->entry(
          std::span<const std::size_t>(indices + b * ndim, ndim));
      }
    }

    // Phase two: one GIL acquisition and one call for the whole batch.
    py::gil_scoped_acquire gil;

    try
    {
      py::tuple arguments(input_count);
      for (std::size_t k = 0; k < input_count; k++)
      {
        // Constructed without a base handle, so pybind11 copies: the callback holds its
        // own arrays and m_scratch stays ours to reuse on the next batch.
        arguments[k] = py::array_t<data_t>(
          static_cast<py::ssize_t>(batch_count), m_scratch.data() + k * batch_count);
      }

      py::object result = m_function(*arguments);
      this->store_batch_result(result, batch_count, values, "cross_function callback");
    }
    catch (py::error_already_set& error)
    {
      this->abort_with(error);
    }
  }

  /**
   * \brief Entries per callback invocation.
   *
   * Matches the vectorized evaluator: large enough that a whole score-tensor block
   * normally arrives in one call, small enough to keep the staging buffers bounded as
   * interface ranks grow.
   */
  [[nodiscard]] std::size_t preferred_batch_size() const override
  {
    return 1u << 20;
  }

private:
  py::object m_function;
  std::vector<RuntimeTensorTrain<data_t> const*> m_inputs;
  /// Reused between batches so a long sweep does not reallocate on every call.
  std::vector<data_t> m_scratch;
};

} // namespace boba_python
