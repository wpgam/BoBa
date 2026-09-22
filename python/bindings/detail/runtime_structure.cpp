// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime_structure.hpp"

#include "errors.hpp"

namespace boba_python
{

namespace
{

/// Copies the selected sub-range of a core's mode index. Ranks are carried over as is.
template <typename data_t>
typename RuntimeTensorTrain<data_t>::core_type select_mode_range(
  typename RuntimeTensorTrain<data_t>::core_type const& core,
  ModeSelection const& selection)
{
  using core_type = typename RuntimeTensorTrain<data_t>::core_type;

  const std::size_t left_rank = core.sizes(0);
  const std::size_t right_rank = core.sizes(2);

  core_type selected({left_rank, selection.count, right_rank});

  auto source_view = core.const_view();
  auto target_view = selected.view();

  const std::ptrdiff_t start = selection.start;
  const std::ptrdiff_t step = selection.step;

  ::boba::loop<python_space, 3>({left_rank, selection.count, right_rank},
                                [=] __boba_host_device__(::boba::Array<std::size_t, 3> ijk)
  {
    auto [rank_left, index, rank_right] = ijk;
    const auto source = static_cast<std::size_t>(
      start + static_cast<std::ptrdiff_t>(index) * step);
    target_view({rank_left, index, rank_right}) = source_view({rank_left, source, rank_right});
  });

  return selected;
}

/// The (left_rank, right_rank) matrix obtained by fixing a core's mode index.
template <typename data_t>
::boba::Matrix<python_space, data_t> fix_mode_index(
  typename RuntimeTensorTrain<data_t>::core_type const& core,
  std::size_t index)
{
  const std::size_t left_rank = core.sizes(0);
  const std::size_t right_rank = core.sizes(2);

  ::boba::Matrix<python_space, data_t> fixed({left_rank, right_rank});

  auto source_view = core.const_view();
  auto target_view = fixed.view();

  ::boba::loop<python_space, 2>({left_rank, right_rank},
                                [=] __boba_host_device__(::boba::Array<::boba::index_t, 2> lr)
  {
    target_view(lr) = source_view({lr[0], index, lr[1]});
  });

  return fixed;
}

} // namespace

template <typename data_t>
RuntimeTensorTrain<data_t> concatenate_mode(
  RuntimeTensorTrain<data_t> const& a,
  RuntimeTensorTrain<data_t> const& b,
  std::size_t mode)
{
  using core_type = typename RuntimeTensorTrain<data_t>::core_type;

  const std::size_t ndim = a.ndim();
  const std::size_t a_extent = a.core(mode).sizes(1);
  const std::size_t b_extent = b.core(mode).sizes(1);

  const std::size_t joined_extent =
    checked_product({a_extent + b_extent, 1}, "concatenated mode");

  std::vector<core_type> cores(ndim);

  // Start from a, with the concatenation axis widened to hold both operands. a keeps
  // the leading range of that mode; the trailing range is zero and is where b lands.
  for (std::size_t d = 0; d < ndim; d++)
  {
    if (d != mode)
    {
      cores[d] = a.core(d);
      continue;
    }

    auto const& source = a.core(d);
    cores[d] = core_type({source.sizes(0), joined_extent, source.sizes(2)});
    cores[d].fill_with_zeros();

    auto source_view = source.const_view();
    auto target_view = cores[d].view();

    ::boba::loop<python_space, 3>({source.sizes(0), a_extent, source.sizes(2)},
                                  [=] __boba_host_device__(::boba::Array<std::size_t, 3> ijk)
    {
      target_view(ijk) = source_view(ijk);
    });
  }

  // Add b block-diagonally in the rank indices, offset past a along the concatenation
  // axis. Because the two mode ranges are disjoint, no index of the result sees a
  // contribution from both operands, which is what makes this concatenation rather than
  // a sum.
  for (std::size_t d = 0; d < ndim; d++)
  {
    const std::size_t mode_offset = (d == mode) ? a_extent : 0;
    ::boba::add_subcore(cores[d], b.core(d), d, ndim, mode_offset);
  }

  return RuntimeTensorTrain<data_t>(std::move(cores));
}

template <typename data_t>
RuntimeTensorTrain<data_t> slice_modes(
  RuntimeTensorTrain<data_t> const& train,
  std::vector<ModeSelection> const& selections)
{
  using core_type = typename RuntimeTensorTrain<data_t>::core_type;

  const std::size_t ndim = train.ndim();

  std::vector<core_type> kept;
  kept.reserve(ndim);

  // A dropped mode leaves behind a rank-to-rank matrix. It is carried forward until a
  // kept core can absorb it, which keeps the interface ranks of the result equal to the
  // interface ranks of the input.
  ::boba::Matrix<python_space, data_t> carried;
  bool carrying = false;

  for (std::size_t d = 0; d < ndim; d++)
  {
    auto const& selection = selections[d];

    if (selection.drop)
    {
      auto fixed = fix_mode_index<data_t>(train.core(d), static_cast<std::size_t>(selection.start));
      carried = carrying ? ::boba::Matrix<python_space, data_t>(carried * fixed) : fixed;
      carrying = true;
      continue;
    }

    auto selected = select_mode_range<data_t>(train.core(d), selection);

    if (carrying)
    {
      // Absorb from the left into this core's left rank index.
      selected = ::boba::tensor_contraction<1>(
        {"k", "l"}, carried, {"l", "i", "r"}, selected, {"k", "i", "r"});
      carrying = false;
    }

    kept.push_back(std::move(selected));
  }

  // Dropped trailing modes have nothing to their right, so the carry folds into the last
  // kept core from the other side.
  if (carrying && !kept.empty())
  {
    kept.back() = ::boba::tensor_contraction<1>(
      {"l", "i", "k"}, kept.back(), {"k", "r"}, carried, {"l", "i", "r"});
  }

  return RuntimeTensorTrain<data_t>(std::move(kept));
}

template RuntimeTensorTrain<float> concatenate_mode<float>(
  RuntimeTensorTrain<float> const&, RuntimeTensorTrain<float> const&, std::size_t);
template RuntimeTensorTrain<double> concatenate_mode<double>(
  RuntimeTensorTrain<double> const&, RuntimeTensorTrain<double> const&, std::size_t);

template RuntimeTensorTrain<float> slice_modes<float>(
  RuntimeTensorTrain<float> const&, std::vector<ModeSelection> const&);
template RuntimeTensorTrain<double> slice_modes<double>(
  RuntimeTensorTrain<double> const&, std::vector<ModeSelection> const&);

} // namespace boba_python
