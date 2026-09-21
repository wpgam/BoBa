// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime_tensor_train.hpp"

#include "errors.hpp"

#include <cstring>
#include <limits>

namespace boba_python
{

template <typename data_t>
std::vector<std::size_t> RuntimeTensorTrain<data_t>::shape() const
{
  std::vector<std::size_t> extents(m_cores.size());
  for (std::size_t d = 0; d < m_cores.size(); d++)
  {
    extents[d] = m_cores[d].sizes(1);
  }
  return extents;
}

template <typename data_t>
std::vector<std::size_t> RuntimeTensorTrain<data_t>::ranks() const
{
  if (m_cores.empty())
  {
    return {};
  }

  std::vector<std::size_t> interface_ranks(m_cores.size() + 1);
  for (std::size_t d = 0; d < m_cores.size(); d++)
  {
    interface_ranks[d] = m_cores[d].sizes(0);
  }
  interface_ranks[m_cores.size()] = m_cores.back().sizes(2);
  return interface_ranks;
}

template <typename data_t>
std::size_t RuntimeTensorTrain<data_t>::dense_size() const
{
  return checked_product(shape(), "dense tensor");
}

template <typename data_t>
data_t RuntimeTensorTrain<data_t>::entry(std::span<const std::size_t> indices) const
{
  // Left-to-right contraction of the selected core slices, as in
  // boba::TensorTrain::unroll_value, but with a runtime-length index.
  ::boba::Matrix<python_space, data_t> accumulator({1, 1});
  accumulator.fill_with(static_cast<data_t>(1));

  for (std::size_t d = 0; d < m_cores.size(); d++)
  {
    auto const& current_core = m_cores[d];
    const std::size_t left_rank = current_core.sizes(0);
    const std::size_t right_rank = current_core.sizes(2);
    const std::size_t mode_index = indices[d];

    ::boba::Matrix<python_space, data_t> slice({left_rank, right_rank});
    auto slice_view = slice.view();
    auto core_view = current_core.const_view();

    ::boba::loop<python_space, 2>({left_rank, right_rank},
                                  [=] __boba_host_device__(::boba::Array<::boba::index_t, 2> lr)
    {
      slice_view(lr) = core_view({lr[0], mode_index, lr[1]});
    });

    auto contracted = accumulator * slice;
    accumulator = contracted;
  }

  return accumulator({0, 0});
}

template <typename data_t>
typename RuntimeTensorTrain<data_t>::core_type RuntimeTensorTrain<data_t>::reconstruct() const
{
  // Reject impossible allocations before BoBa is asked for the memory.
  const std::size_t total = dense_size();
  const std::size_t byte_limit = std::numeric_limits<std::size_t>::max() / sizeof(data_t);
  if (total > byte_limit)
  {
    throw MemoryErrorException("dense reconstruction: byte count overflows std::size_t");
  }

  core_type accumulated_core({1, 1, 1});
  accumulated_core.fill_with(static_cast<data_t>(1));

  // Same right-to-left core merge as boba::TensorTrain::unroll_subtensor. Each step
  // merges one more mode into the middle index of a Tensor<3>, so nothing here is
  // dimension-templated.
  for (std::size_t d = m_cores.size(); d > 0; d--)
  {
    auto merged = ::boba::partial_decompress_core(m_cores[d - 1], accumulated_core);
    accumulated_core = merged;
  }

  return accumulated_core;
}

template <typename data_t>
RuntimeTensorTrain<data_t> compress_dense(
  data_t const* column_major_data,
  std::vector<std::size_t> const& shape,
  double relative_tolerance,
  double absolute_tolerance,
  std::size_t max_rank)
{
  using core_type = typename RuntimeTensorTrain<data_t>::core_type;

  const std::size_t ndim = shape.size();
  const std::size_t total = checked_product(shape, "dense tensor");

  std::vector<core_type> cores(ndim);

  // Mirrors the `dimension == 1` early return in boba::TensorTrain::compress.
  if (ndim == 1)
  {
    cores[0] = core_type({1, shape[0], 1});
    std::memcpy(cores[0].data(), column_major_data, total * sizeof(data_t));
    return RuntimeTensorTrain<data_t>(std::move(cores));
  }

  // The native routine starts from `temp_fold` reshaped out of the input tensor; the
  // payload it reshapes is exactly this column-major buffer.
  ::boba::Matrix<python_space, data_t> temp_fold({1, total});
  temp_fold.rename("folding_matrix");
  std::memcpy(temp_fold.data(), column_major_data, total * sizeof(data_t));

  std::size_t ranks = 1;
  for (std::size_t d = 0; d + 1 < ndim; d++)
  {
    const std::size_t rows = shape[d] * ranks;
    const std::size_t cols = temp_fold.size() / rows;
    temp_fold.reshape({rows, cols});

    // A fresh SVD per interface, matching the native Array<SVD, dimension - 1>.
    ::boba::SVD<python_space, data_t> svd;
    svd.tolerance_relative = static_cast<data_t>(relative_tolerance);
    svd.tolerance_absolute = static_cast<data_t>(absolute_tolerance);
    svd.max_kept_singular_values = max_rank;

    svd(temp_fold);
    ::boba::apply_as_diagonal_right_in_place(svd.S, svd.U);

    ranks = svd.significant_singular_values;
    if (ranks == 0)
    {
      // Native behavior: the whole train collapses to zero.
      for (std::size_t zero_d = 0; zero_d < ndim; zero_d++)
      {
        cores[zero_d] = core_type({1, shape[zero_d], 1});
        cores[zero_d].fill_with_zeros();
      }
      return RuntimeTensorTrain<data_t>(std::move(cores));
    }

    cores[d] = ::boba::write_to_core_from_left_fold(svd.U, shape[d]);
    temp_fold = svd.V.transpose();
  }

  cores[ndim - 1] = ::boba::write_to_core_from_right_fold(temp_fold, shape[ndim - 1]);
  return RuntimeTensorTrain<data_t>(std::move(cores));
}

template <typename data_t>
RuntimeTensorTrain<data_t> make_random_initial_guess(
  std::vector<std::size_t> const& shape,
  std::vector<std::size_t> const& internal_ranks)
{
  using core_type = typename RuntimeTensorTrain<data_t>::core_type;

  const std::size_t ndim = shape.size();
  std::vector<core_type> cores(ndim);

  for (std::size_t d = 0; d < ndim; d++)
  {
    const std::size_t left_rank = (d == 0) ? 1 : internal_ranks[d - 1];
    const std::size_t right_rank = (d + 1 == ndim) ? 1 : internal_ranks[d];

    cores[d] = core_type({left_rank, shape[d], right_rank});
    // BoBa seeds fill_with_random() from std::random_device; no seed hook is exposed
    // and this code deliberately does not introduce one.
    cores[d].fill_with_random();
  }

  return RuntimeTensorTrain<data_t>(std::move(cores));
}

template class RuntimeTensorTrain<float>;
template class RuntimeTensorTrain<double>;

template RuntimeTensorTrain<float> compress_dense<float>(
  float const*, std::vector<std::size_t> const&, double, double, std::size_t);
template RuntimeTensorTrain<double> compress_dense<double>(
  double const*, std::vector<std::size_t> const&, double, double, std::size_t);

template RuntimeTensorTrain<float> make_random_initial_guess<float>(
  std::vector<std::size_t> const&, std::vector<std::size_t> const&);
template RuntimeTensorTrain<double> make_random_initial_guess<double>(
  std::vector<std::size_t> const&, std::vector<std::size_t> const&);

} // namespace boba_python
