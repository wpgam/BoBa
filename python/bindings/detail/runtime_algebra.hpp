// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "runtime_tensor_train.hpp"

#include <cstddef>

/**
 * \file
 * \brief Runtime-dimensional exact tensor-train arithmetic for the Python extension.
 *
 * Every operation here is exact: no truncation happens, and the interface ranks grow
 * accordingly -- addition concatenates them, the Hadamard product multiplies them. That
 * is the defining property of the TT format, not a shortcoming, and it is why
 * `runtime_rounding.hpp` exists. Callers are expected to round when the growth stops
 * being worth its cost.
 *
 * Adapted from `boba::TensorTrain::operator+`, `::multiply_scalar`, `::inner_product`,
 * `boba::elementwise_product` and `boba::norm_frobenius`. Addition delegates to
 * `boba::add_subcore`, which already takes the core index and the train dimension as
 * ordinary runtime arguments, so nothing about it needed restating.
 */

namespace boba_python
{

/**
 * \brief Exact sum, `a + b`.
 *
 * The cores are combined block-diagonally by `boba::add_subcore`, so the interface ranks
 * of the result are the sums of the inputs' ranks. The shapes must match; the caller
 * checks that before calling.
 */
template <typename data_t>
[[nodiscard]] RuntimeTensorTrain<data_t> add(
  RuntimeTensorTrain<data_t> const& a,
  RuntimeTensorTrain<data_t> const& b);

/**
 * \brief Exact scalar multiple, `scalar * a`.
 *
 * Absorbed into the first core, as `boba::TensorTrain::multiply_scalar` does. Ranks are
 * unchanged.
 */
template <typename data_t>
[[nodiscard]] RuntimeTensorTrain<data_t> scale(
  RuntimeTensorTrain<data_t> const& a,
  data_t scalar);

/**
 * \brief Exact elementwise (Hadamard) product.
 *
 * Interface rank `d` of the result is the product of the inputs' rank `d`, which is why
 * repeated Hadamard products without rounding are the fastest way to exhaust memory in
 * this format.
 */
template <typename data_t>
[[nodiscard]] RuntimeTensorTrain<data_t> hadamard(
  RuntimeTensorTrain<data_t> const& a,
  RuntimeTensorTrain<data_t> const& b);

/**
 * \brief Inner product of the two trains seen as vectors.
 *
 * Contracts right to left, never forming the dense tensor. Mirrors
 * `boba::TensorTrain::inner_product`.
 */
template <typename data_t>
[[nodiscard]] data_t inner_product(
  RuntimeTensorTrain<data_t> const& a,
  RuntimeTensorTrain<data_t> const& b);

/// Frobenius norm, `sqrt(<a, a>)`. Mirrors `boba::norm_frobenius`.
template <typename data_t>
[[nodiscard]] data_t norm_frobenius(RuntimeTensorTrain<data_t> const& a);

extern template RuntimeTensorTrain<float> add<float>(
  RuntimeTensorTrain<float> const&, RuntimeTensorTrain<float> const&);
extern template RuntimeTensorTrain<double> add<double>(
  RuntimeTensorTrain<double> const&, RuntimeTensorTrain<double> const&);

extern template RuntimeTensorTrain<float> scale<float>(RuntimeTensorTrain<float> const&, float);
extern template RuntimeTensorTrain<double> scale<double>(RuntimeTensorTrain<double> const&, double);

extern template RuntimeTensorTrain<float> hadamard<float>(
  RuntimeTensorTrain<float> const&, RuntimeTensorTrain<float> const&);
extern template RuntimeTensorTrain<double> hadamard<double>(
  RuntimeTensorTrain<double> const&, RuntimeTensorTrain<double> const&);

extern template float inner_product<float>(
  RuntimeTensorTrain<float> const&, RuntimeTensorTrain<float> const&);
extern template double inner_product<double>(
  RuntimeTensorTrain<double> const&, RuntimeTensorTrain<double> const&);

extern template float norm_frobenius<float>(RuntimeTensorTrain<float> const&);
extern template double norm_frobenius<double>(RuntimeTensorTrain<double> const&);

} // namespace boba_python
