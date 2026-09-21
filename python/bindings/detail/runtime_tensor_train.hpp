// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <cstddef>
#include <span>
#include <vector>

/**
 * \file
 * \brief Runtime-dimensional tensor train used only by the BoBa Python extension.
 *
 * BoBa's public `boba::TensorTrain<dimension, space, data_t>` encodes the tensor
 * dimension as a template parameter. The Python interface must accept a dimension
 * that is only known at run time, so this header defines a private tensor-train
 * representation whose core count is a `std::vector` size rather than a template
 * argument. It is deliberately *not* part of BoBa's public C++ API and must not be
 * installed under `include/BOBA/`.
 *
 * Only the container is re-implemented. Every numerical kernel (SVD, QR, matrix
 * products, folds) is a BoBa primitive.
 */

namespace boba_python
{

/// Execution space used by the round-one Python bindings. Host only.
inline constexpr ::boba::execution_space python_space = ::boba::host_space;

/**
 * \brief Tensor train whose dimension is a runtime value.
 *
 * Cores follow BoBa's own core convention, `Tensor<3>` with extents
 * `(left_rank, mode_size, right_rank)` and BoBa's first-index-fastest
 * (column-major) storage order.
 *
 * \tparam data_t Scalar type, `float` or `double` in round one.
 */
template <typename data_t>
class RuntimeTensorTrain
{
public:
  using core_type = ::boba::Tensor<3, python_space, data_t>;

  RuntimeTensorTrain() = default;

  explicit RuntimeTensorTrain(std::vector<core_type> cores)
      : m_cores(std::move(cores))
  {
  }

  [[nodiscard]] std::size_t ndim() const noexcept
  {
    return m_cores.size();
  }

  /// Mode extents, one per core.
  [[nodiscard]] std::vector<std::size_t> shape() const;

  /// All `ndim() + 1` interface ranks, including the boundary ranks `(1, ..., 1)`.
  [[nodiscard]] std::vector<std::size_t> ranks() const;

  [[nodiscard]] std::vector<core_type>& cores() noexcept
  {
    return m_cores;
  }

  [[nodiscard]] std::vector<core_type> const& cores() const noexcept
  {
    return m_cores;
  }

  [[nodiscard]] core_type& core(std::size_t d) noexcept
  {
    return m_cores[d];
  }

  [[nodiscard]] core_type const& core(std::size_t d) const noexcept
  {
    return m_cores[d];
  }

  /// Number of entries of the represented dense tensor. Throws on overflow.
  [[nodiscard]] std::size_t dense_size() const;

  /**
   * \brief Evaluates one entry by a left-to-right contraction of the selected core slices.
   *
   * Mirrors `boba::TensorTrain::unroll_value` but takes a runtime-length index.
   * Never materializes the dense tensor.
   */
  [[nodiscard]] data_t entry(std::span<const std::size_t> indices) const;

  /**
   * \brief Contracts the whole train into a single flat core.
   *
   * Mirrors `boba::TensorTrain::unroll_subtensor` for the full range: repeated
   * `boba::partial_decompress_core` calls, which are dimension-free. The result is
   * a `(1, dense_size(), 1)` core whose payload is the dense tensor in BoBa's
   * column-major order over `shape()`.
   */
  [[nodiscard]] core_type reconstruct() const;

private:
  std::vector<core_type> m_cores;
};

/**
 * \brief Runtime-dimensional TT-SVD compression.
 *
 * Adapted from `boba::TensorTrain<dimension, space, data_t>::compress`. The native
 * routine folds the input `Tensor<dimension>` into a `(1, N)` matrix and then never
 * touches the dimension again, so the only change needed here is to take the already
 * flattened, column-major payload directly.
 *
 * \param column_major_data Dense payload in BoBa order (first index fastest).
 * \param shape Mode extents.
 * \param relative_tolerance Maps to `boba::SVD::tolerance_relative`.
 * \param absolute_tolerance Maps to `boba::SVD::tolerance_absolute`.
 * \param max_rank Maps to `boba::SVD::max_kept_singular_values`.
 */
template <typename data_t>
[[nodiscard]] RuntimeTensorTrain<data_t> compress_dense(
  data_t const* column_major_data,
  std::vector<std::size_t> const& shape,
  double relative_tolerance,
  double absolute_tolerance,
  std::size_t max_rank);

/**
 * \brief Builds the random initial guess consumed by the runtime cross sweep.
 *
 * Follows the convention used by `examples/tests/test_cross.cpp`: cores sized
 * `(r_d, n_d, r_{d+1})` with unit boundary ranks and `fill_with_random()` contents.
 *
 * \param shape Mode extents.
 * \param internal_ranks Exactly `shape.size() - 1` interior ranks.
 */
template <typename data_t>
[[nodiscard]] RuntimeTensorTrain<data_t> make_random_initial_guess(
  std::vector<std::size_t> const& shape,
  std::vector<std::size_t> const& internal_ranks);

extern template class RuntimeTensorTrain<float>;
extern template class RuntimeTensorTrain<double>;

} // namespace boba_python
