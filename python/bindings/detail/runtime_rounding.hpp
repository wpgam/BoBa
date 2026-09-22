// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "runtime_tensor_train.hpp"

#include <cstddef>
#include <vector>

/**
 * \file
 * \brief Runtime-dimensional orthogonalization and rounding for the Python extension.
 *
 * Adapted from `boba::TensorTrain::orthogonalize` and `boba::TensorTrain::round`. Both
 * native routines are sweeps whose bodies touch one or two cores at a time and mention
 * `dimension` only as a loop bound, so only the loop is restated here. Every numerical
 * step -- `boba::QR`, `boba::SVD`, the unfold/fold helpers and
 * `boba::apply_as_diagonal_right_in_place` -- is the same BoBa primitive the native
 * sweep calls.
 */

namespace boba_python
{

/**
 * \brief Left-orthogonalizes \p train in place by a left-to-right QR sweep.
 *
 * Afterwards every core but the last is left-orthogonal and the represented tensor is
 * unchanged up to roundoff. Mirrors `boba::TensorTrain::orthogonalize`.
 */
template <typename data_t>
void orthogonalize(RuntimeTensorTrain<data_t>& train);

/**
 * \brief Rounds \p train in place: a QR sweep, then a right-to-left truncated SVD sweep.
 *
 * Mirrors `boba::TensorTrain::round`, including its collapse-to-zero behavior when an
 * interface retains no significant singular values. The result is right-orthogonal with
 * the first core as the orthogonality center.
 *
 * \param relative_tolerance Maps to `boba::SVD::tolerance_relative`.
 * \param absolute_tolerance Maps to `boba::SVD::tolerance_absolute`.
 * \param max_ranks One bound per interface, `train.ndim() + 1` entries, indexed the way
 *        `boba::TensorTrain::max_ranks` is: entry `d` bounds the interface between cores
 *        `d - 1` and `d`. The two boundary entries are never read.
 */
template <typename data_t>
void round(
  RuntimeTensorTrain<data_t>& train,
  double relative_tolerance,
  double absolute_tolerance,
  std::vector<std::size_t> const& max_ranks);

extern template void orthogonalize<float>(RuntimeTensorTrain<float>&);
extern template void orthogonalize<double>(RuntimeTensorTrain<double>&);

extern template void round<float>(
  RuntimeTensorTrain<float>&, double, double, std::vector<std::size_t> const&);
extern template void round<double>(
  RuntimeTensorTrain<double>&, double, double, std::vector<std::size_t> const&);

} // namespace boba_python
