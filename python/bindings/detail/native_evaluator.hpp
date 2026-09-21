// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "evaluator.hpp"

#include <cstddef>
#include <utility>

/**
 * \file
 * \brief Pure-C++ evaluator, free of any Python dependency.
 *
 * Used by the C++ parity test that compares the runtime cross sweep against native
 * `boba::DMRGCross::apply` on fixed compile-time dimensions. It also demonstrates that
 * the sampling algorithm has no structural dependence on a Python callable, which is
 * what keeps a future device-side evaluator a drop-in replacement.
 */

namespace boba_python
{

/**
 * \brief Adapts any callable of the form `data_t f(const std::size_t* index, std::size_t ndim)`.
 */
template <typename data_t, typename function_t>
class NativeEvaluator final : public Evaluator<data_t>
{
public:
  explicit NativeEvaluator(function_t function)
      : m_function(std::move(function))
  {
  }

  void evaluate(
    std::size_t const* indices,
    std::size_t batch_count,
    std::size_t ndim,
    data_t* values) override
  {
    for (std::size_t b = 0; b < batch_count; b++)
    {
      values[b] = m_function(indices + b * ndim, ndim);
    }
  }

private:
  function_t m_function;
};

template <typename data_t, typename function_t>
[[nodiscard]] NativeEvaluator<data_t, function_t> make_native_evaluator(function_t function)
{
  return NativeEvaluator<data_t, function_t>(std::move(function));
}

} // namespace boba_python
