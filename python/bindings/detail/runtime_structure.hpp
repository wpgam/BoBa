// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "runtime_tensor_train.hpp"

#include <cstddef>
#include <vector>

/**
 * \file
 * \brief Structural tensor-train operations: mode concatenation and mode slicing.
 *
 * Unlike the rest of this directory, these two have no counterpart in BoBa to adapt.
 * Both are nonetheless ordinary core-level manipulations that never form the dense
 * tensor:
 *
 *  - Concatenation along a mode is the block-diagonal core combination that
 *    `boba::add_subcore` already implements, applied with a mode-index offset on the
 *    concatenation axis so the two operands occupy disjoint ranges of that mode. Seen
 *    from the tensor, it is the sum of two trains whose supports do not overlap, which
 *    is exactly what concatenation is.
 *
 *  - Slicing a mode copies a sub-range of one core's middle index. Interface ranks are
 *    untouched, because restricting a mode cannot couple the factors.
 *
 * Note that BoBa's own subtensor path cannot be reused here: `unroll_subtensor` forwards
 * to `partial_decompress_core`, which asserts that the requested range covers the whole
 * mode (`"Unroll subtensors not currently supported, see issue 238"`). Working at the
 * core level sidesteps that limitation entirely.
 */

namespace boba_python
{

/**
 * \brief How one mode is selected by a slicing operation.
 *
 * `start`, `step` and `count` are the already-resolved form of a Python slice, as
 * `py::slice::compute` produces: negative indices and clamping have been applied, so
 * the selected source indices are `start + i * step` for `i` in `[0, count)`.
 */
struct ModeSelection
{
  /// True when a single index was given, so the mode disappears from the result.
  bool drop = false;
  std::ptrdiff_t start = 0;
  std::ptrdiff_t step = 1;
  /// Number of selected indices along this mode; always 1 when `drop` is true.
  std::size_t count = 1;
};

/**
 * \brief Concatenates two trains along mode \p mode.
 *
 * The operands must agree on every mode extent except \p mode, and the result has extent
 * `a.shape()[mode] + b.shape()[mode]` there. Interface ranks are the sums of the
 * operands' ranks, which is the exact cost of the operation; nothing is truncated and
 * the dense tensor is never formed.
 *
 * The caller validates the shapes.
 */
template <typename data_t>
[[nodiscard]] RuntimeTensorTrain<data_t> concatenate_mode(
  RuntimeTensorTrain<data_t> const& a,
  RuntimeTensorTrain<data_t> const& b,
  std::size_t mode);

/**
 * \brief Selects a sub-range of every mode, optionally dropping some entirely.
 *
 * Kept modes are restricted to the selected indices by copying a sub-range of the
 * corresponding core, leaving interface ranks unchanged. A dropped mode fixes one index,
 * turning its core into a matrix that is absorbed into a neighbouring kept core, so the
 * result has one fewer mode and, again, the same interface ranks.
 *
 * \param selections Exactly `train.ndim()` entries. At least one must be a kept mode;
 *        a selection that drops every mode denotes a scalar, which the caller handles
 *        through `entry()` instead.
 */
template <typename data_t>
[[nodiscard]] RuntimeTensorTrain<data_t> slice_modes(
  RuntimeTensorTrain<data_t> const& train,
  std::vector<ModeSelection> const& selections);

extern template RuntimeTensorTrain<float> concatenate_mode<float>(
  RuntimeTensorTrain<float> const&, RuntimeTensorTrain<float> const&, std::size_t);
extern template RuntimeTensorTrain<double> concatenate_mode<double>(
  RuntimeTensorTrain<double> const&, RuntimeTensorTrain<double> const&, std::size_t);

extern template RuntimeTensorTrain<float> slice_modes<float>(
  RuntimeTensorTrain<float> const&, std::vector<ModeSelection> const&);
extern template RuntimeTensorTrain<double> slice_modes<double>(
  RuntimeTensorTrain<double> const&, std::vector<ModeSelection> const&);

} // namespace boba_python
