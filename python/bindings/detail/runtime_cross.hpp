// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

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
  /// Relative truncation tolerance of each two-site SVD. <= 0 keeps the native default
  /// (DMRGCross::two_site_svd_tolerance_relative, 1e-7), independent of `tolerance`.
  double svd_tolerance = 0.0;
  /// Cap on every interface rank inside the sweep (the two-site SVD and the kick-rank
  /// enrichment). 0 means uncapped, the native behaviour.
  std::size_t max_rank = 0;
  /// Size of a fixed random validation set of entries. > 0: after every full sweep the
  /// relative L2 error of the current train on it is computed, and the sweep stops once it
  /// is below `tolerance`. 0 keeps the native stopping test (largest local two-site
  /// interpolation error below `tolerance`).
  std::size_t val_size = 0;
  /// Seed of the validation set.
  std::uint64_t val_seed = 1234;
  /// With a validation set, also stop once a sweep improves the validation error by less
  /// than this fraction (the error has stalled at the level the two-site SVD tolerance allows).
  /// 0 disables the stall test.
  double val_stall = 0.1;
  /// If set, reseeds the calling thread's generator before the random starting guess, so the
  /// result depends only on this seed, not on earlier random fills or on other threads.
  std::optional<std::uint32_t> seed;
};

/// What a cross run reports back (filled when a pointer is passed).
struct CrossInfo
{
  /// Relative L2 error on the validation set after each full sweep (empty if val_size = 0).
  std::vector<double> val_history;
  /// Final validation error (or the last largest local error if val_size = 0).
  double final_error = 0.0;
  /// Full sweeps performed.
  std::size_t sweeps = 0;
  /// Entries requested from the evaluator, including the validation set.
  std::size_t n_evals = 0;
  /// Whether the stopping test was met before the sweep budget ran out.
  bool converged = false;
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
  CrossOptions const& options,
  CrossInfo* info = nullptr);

} // namespace boba_python
