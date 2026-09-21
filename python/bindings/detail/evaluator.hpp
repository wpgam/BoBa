// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>

/**
 * \file
 * \brief Backend-agnostic entry-evaluation interface used by the runtime cross sweep.
 *
 * The cross algorithm only ever needs "given a batch of multi-indices, give me the
 * corresponding tensor entries". Keeping that behind an interface means the sampling
 * loop has no dependency on Python at all, which is what makes a future device-side
 * evaluator (CUDA/HIP, or a native array source) a drop-in addition rather than an
 * API change.
 */

namespace boba_python
{

/**
 * \brief Evaluates batches of runtime-dimensional multi-indices.
 *
 * \tparam data_t Scalar type of the produced values.
 *
 * Implementations must be callable from the thread that drives the cross sweep and
 * are responsible for their own GIL handling if they touch Python.
 */
template <typename data_t>
class Evaluator
{
public:
  virtual ~Evaluator() = default;

  Evaluator() = default;
  Evaluator(Evaluator const&) = delete;
  Evaluator& operator=(Evaluator const&) = delete;

  /**
   * \brief Fills \p values with the tensor entries at the requested indices.
   *
   * \param indices `batch_count * ndim` zero-based indices, one multi-index per row,
   *        stored row by row (index `d` of row `b` lives at `indices[b * ndim + d]`).
   * \param batch_count Number of multi-indices in this batch.
   * \param ndim Tensor dimension; the same for every row.
   * \param values Output buffer of `batch_count` entries.
   *
   * \note Implementations backed by Python throw `PythonCallbackError` after restoring
   *       the Python error indicator; they never let a `py::error_already_set` escape
   *       into GIL-released code.
   */
  virtual void evaluate(
    std::size_t const* indices,
    std::size_t batch_count,
    std::size_t ndim,
    data_t* values) = 0;

  /**
   * \brief Upper bound on the batch size this evaluator wants to receive at once.
   *
   * The cross sweep splits a score tensor into chunks no larger than this so the
   * index staging buffer stays bounded even when interface ranks are large.
   */
  [[nodiscard]] virtual std::size_t preferred_batch_size() const
  {
    return 1u << 16;
  }
};

} // namespace boba_python
