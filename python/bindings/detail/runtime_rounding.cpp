// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime_rounding.hpp"

namespace boba_python
{

namespace
{

/**
 * \brief Replaces every core with a unit-rank zero core.
 *
 * Native `round` calls `fill_with_zeros`, which keeps the current core extents. Dropping
 * to unit ranks represents the same tensor in the canonical form, and matches the zero
 * path already taken by `compress_dense`.
 */
template <typename data_t>
void collapse_to_zero(RuntimeTensorTrain<data_t>& train)
{
  using core_type = typename RuntimeTensorTrain<data_t>::core_type;

  auto extents = train.shape();
  for (std::size_t d = 0; d < extents.size(); d++)
  {
    train.core(d) = core_type({1, extents[d], 1});
    train.core(d).fill_with_zeros();
  }
}

} // namespace

template <typename data_t>
void orthogonalize(RuntimeTensorTrain<data_t>& train)
{
  const std::size_t ndim = train.ndim();

  // Native `dimension == 1` early return. Every train reaching this layer carries unit
  // boundary ranks -- `from_cores` enforces them and both `compress` and `cross` produce
  // them -- so a one-core train is already orthogonal and the native rank-collapsing
  // branch is unreachable here.
  if (ndim < 2)
  {
    return;
  }

  auto extents = train.shape();

  // One QR workspace reused across the sweep, as in the native routine.
  ::boba::QR<python_space, data_t> qr;

  for (std::size_t d = 0; d + 1 < ndim; d++)
  {
    // Scope the unfolding so it is released before the next core is touched.
    {
      auto unfold_left = ::boba::compute_unfold_left(train.core(d));
      qr(unfold_left);
    }

    // Q is the left unfolding of the current core.
    train.core(d) = ::boba::write_to_core_from_left_fold(qr.Q, extents[d]);

    // Absorb R into the next core.
    train.core(d + 1) = ::boba::tensor_contraction<1>(
      {"k", "l"}, qr.R, {"l", "i", "r"}, train.core(d + 1), {"k", "i", "r"});
  }
}

template <typename data_t>
void round(
  RuntimeTensorTrain<data_t>& train,
  double relative_tolerance,
  double absolute_tolerance,
  std::vector<std::size_t> const& max_ranks)
{
  const std::size_t ndim = train.ndim();

  // See the note in `orthogonalize`: a one-core train has nothing to truncate.
  if (ndim < 2)
  {
    return;
  }

  auto extents = train.shape();

  // Reuse one SVD workspace during the sweep.
  ::boba::SVD<python_space, data_t> svd;
  svd.tolerance_relative = static_cast<data_t>(relative_tolerance);
  svd.tolerance_absolute = static_cast<data_t>(absolute_tolerance);

  orthogonalize(train);

  // Right-to-left truncated SVD sweep.
  for (std::size_t d = ndim - 1; d > 0; d--)
  {
    svd.max_kept_singular_values = max_ranks[d];

    {
      auto unfold_right = ::boba::compute_unfold_right(train.core(d));
      svd(unfold_right);
    }

    if (svd.significant_singular_values == 0)
    {
      collapse_to_zero(train);
      return;
    }

    // Write V^T back into the current core, which becomes right-orthogonal.
    {
      auto unfold_right = svd.V.transpose();
      train.core(d) = ::boba::write_to_core_from_right_fold(unfold_right, extents[d]);
    }

    // Form U*S in place and absorb it into the right rank index of the preceding core.
    ::boba::apply_as_diagonal_right_in_place(svd.S, svd.U);

    train.core(d - 1) = ::boba::tensor_contraction<1>(
      {"l", "i", "k"}, train.core(d - 1), {"k", "r"}, svd.U, {"l", "i", "r"});
  }
}

template void orthogonalize<float>(RuntimeTensorTrain<float>&);
template void orthogonalize<double>(RuntimeTensorTrain<double>&);

template void round<float>(
  RuntimeTensorTrain<float>&, double, double, std::vector<std::size_t> const&);
template void round<double>(
  RuntimeTensorTrain<double>&, double, double, std::vector<std::size_t> const&);

} // namespace boba_python
