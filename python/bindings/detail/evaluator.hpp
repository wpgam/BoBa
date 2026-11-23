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

  /**
   * \brief Optionally fills a whole two-site score block in one call.
   *
   * The block is the 4-way tensor T(s1, i1, i2, s2) =
   * f(left[s1], i1, i2, right[s2]) for the free modes (k, k + 1), stored column-major
   * (s1 fastest). `left_indices` holds `r_left` tuples over modes 0..k-1, column-major as
   * an `r_left x k` matrix (tuple s1, mode m at `left_indices[s1 + r_left * m]`);
   * `right_indices` holds `r_right` tuples over modes k+2..ndim-1, column-major as an
   * `(ndim - k - 2) x r_right` matrix (mode m, tuple s2 at `right_indices[m + nr * s2]`).
   *
   * Evaluators that can exploit this structure (shared left/right parts) override it
   * and return true. The default returns false, and the caller falls back to
   * evaluate() on the expanded multi-indices.
   */
  virtual bool evaluate_block(
    std::size_t const* left_indices, std::size_t r_left,
    std::size_t k, std::size_t n_k, std::size_t n_k1,
    std::size_t const* right_indices, std::size_t r_right,
    std::size_t ndim, data_t* values)
  {
    (void)left_indices; (void)r_left; (void)k; (void)n_k; (void)n_k1;
    (void)right_indices; (void)r_right; (void)ndim; (void)values;
    return false;
  }
};

} // namespace boba_python
