// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "evaluator.hpp"
#include "runtime_tensor_train.hpp"

#include <cstddef>

/**
 * \file
 * \brief Runtime-dimensional DMRG cross, adapted from `boba::DMRGCross::apply`.
 *
 * `boba::DMRGCross<data_t>::apply` takes the tensor dimension as a template parameter
 * and uses `boba::Array<..., dimension + 1>` internally. The Python interface needs a
 * dimension that is only known at run time, so the sweep is re-expressed here with
 * `std::vector` containers. Everything else is unchanged: the same BoBa QR, SVD, LU,
 * MAXVOL, DEIM, backsolve and matrix-product primitives run in the same order, and the
 * dimension-independent helpers (`select_indices`, `maxvol2`, `reort`) are called on a
 * real `boba::DMRGCross` instance rather than duplicated.
 */

namespace boba_python
{

/// Submatrix selection strategy, mirroring `boba::DMRGCross::SubmatrixSelectionType`.
enum class SubmatrixSelection
{
  MAXVOL,
  DEIM
};

/**
 * \brief Controls for the runtime cross sweep.
 *
 * Defaults intentionally match the member defaults of `boba::DMRGCross`.
 */
struct CrossOptions
{
  double tolerance = 1.0e-7;
  std::size_t n_sweeps = 10;
  std::size_t kick_rank = 2;
  SubmatrixSelection selection = SubmatrixSelection::MAXVOL;
  bool verbose = false;
};

/**
 * \brief Approximates the tensor sampled by \p evaluator with a tensor train.
 *
 * \param initial_guess Starting train; its ranks and mode sizes drive the first sweep.
 * \param evaluator Entry source, queried only at the indices the sweep selects.
 * \param options Sweep controls.
 * \return Host-space tensor train approximation.
 *
 * \note Requires `initial_guess.ndim() >= 2`, the same restriction the native two-site
 *       sweep has. Callers validate this at the Python boundary.
 * \note Does not touch Python. Any Python interaction happens inside \p evaluator.
 */
template <typename data_t>
[[nodiscard]] RuntimeTensorTrain<data_t> runtime_dmrg_cross(
  RuntimeTensorTrain<data_t> const& initial_guess,
  Evaluator<data_t>& evaluator,
  CrossOptions const& options);

} // namespace boba_python
