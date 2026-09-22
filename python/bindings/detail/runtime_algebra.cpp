// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime_algebra.hpp"

#include "errors.hpp"

namespace boba_python
{

template <typename data_t>
RuntimeTensorTrain<data_t> add(
  RuntimeTensorTrain<data_t> const& a,
  RuntimeTensorTrain<data_t> const& b)
{
  // BoBa's Tensor copy constructor allocates and copies, so this shares nothing with a.
  RuntimeTensorTrain<data_t> result = a;

  const std::size_t ndim = result.ndim();
  for (std::size_t d = 0; d < ndim; d++)
  {
    // add_subcore already takes the core index and the train dimension as runtime
    // arguments, and handles the first and last cores, where only one rank index grows.
    ::boba::add_subcore(result.core(d), b.core(d), d, ndim, 0);
  }

  return result;
}

template <typename data_t>
RuntimeTensorTrain<data_t> scale(RuntimeTensorTrain<data_t> const& a, data_t scalar)
{
  RuntimeTensorTrain<data_t> result = a;
  result.core(0) *= scalar;
  return result;
}

template <typename data_t>
RuntimeTensorTrain<data_t> hadamard(
  RuntimeTensorTrain<data_t> const& a,
  RuntimeTensorTrain<data_t> const& b)
{
  using core_type = typename RuntimeTensorTrain<data_t>::core_type;

  const std::size_t ndim = a.ndim();
  auto extents = a.shape();

  std::vector<core_type> cores(ndim);
  for (std::size_t d = 0; d < ndim; d++)
  {
    auto const& a_core = a.core(d);
    auto const& b_core = b.core(d);

    // Each rank index of the result enumerates a pair of input rank indices, so the
    // result rank is the product. Reject an overflowing product before allocating.
    const std::size_t left_rank =
      checked_product({a_core.sizes(0), b_core.sizes(0)}, "elementwise product rank");
    const std::size_t right_rank =
      checked_product({a_core.sizes(2), b_core.sizes(2)}, "elementwise product rank");

    auto rank_left_indexer = ::boba::Multiindexer<2>({a_core.sizes(0), b_core.sizes(0)});
    auto rank_right_indexer = ::boba::Multiindexer<2>({a_core.sizes(2), b_core.sizes(2)});

    auto a_view = a_core.const_view();
    auto b_view = b_core.const_view();

    cores[d] = core_type({left_rank, extents[d], right_rank});
    auto out_view = cores[d].view();

    ::boba::loop<python_space, 3>(out_view.sizes(),
                                  [=] __boba_host_device__(::boba::Array<std::size_t, 3> ijk)
    {
      auto [rank_left, index, rank_right] = ijk;
      auto [a_rank_left, b_rank_left] = rank_left_indexer.multiindex(rank_left);
      auto [a_rank_right, b_rank_right] = rank_right_indexer.multiindex(rank_right);

      out_view({rank_left, index, rank_right}) =
        a_view({a_rank_left, index, a_rank_right}) * b_view({b_rank_left, index, b_rank_right});
    });
  }

  return RuntimeTensorTrain<data_t>(std::move(cores));
}

template <typename data_t>
data_t inner_product(
  RuntimeTensorTrain<data_t> const& a,
  RuntimeTensorTrain<data_t> const& b)
{
  // Right-to-left sweep carrying a single interface vector, as in the native routine.
  ::boba::Vector<python_space, data_t> carried({1});
  carried.fill_with(static_cast<data_t>(1));
  ::boba::Vector<python_space, data_t> next;

  for (std::size_t d = a.ndim(); d > 0; d--)
  {
    auto const& a_core = a.core(d - 1);
    auto const& b_core = b.core(d - 1);

    auto contracted = ::boba::tensor_contraction<1>(
      {"l1", "i", "r1"}, a_core, {"l2", "i", "r2"}, b_core, {"l1", "l2", "r1", "r2"});

    ::boba::Matrix<python_space, data_t> interface(
      {a_core.sizes(0) * b_core.sizes(0), a_core.sizes(2) * b_core.sizes(2)});
    interface.reshape(contracted);

    next = interface * carried;
    carried = next;
  }

  return next.sum_reduce();
}

template <typename data_t>
data_t norm_frobenius(RuntimeTensorTrain<data_t> const& a)
{
  return ::boba::sqrt(::boba::abs(inner_product(a, a)));
}

template RuntimeTensorTrain<float> add<float>(
  RuntimeTensorTrain<float> const&, RuntimeTensorTrain<float> const&);
template RuntimeTensorTrain<double> add<double>(
  RuntimeTensorTrain<double> const&, RuntimeTensorTrain<double> const&);

template RuntimeTensorTrain<float> scale<float>(RuntimeTensorTrain<float> const&, float);
template RuntimeTensorTrain<double> scale<double>(RuntimeTensorTrain<double> const&, double);

template RuntimeTensorTrain<float> hadamard<float>(
  RuntimeTensorTrain<float> const&, RuntimeTensorTrain<float> const&);
template RuntimeTensorTrain<double> hadamard<double>(
  RuntimeTensorTrain<double> const&, RuntimeTensorTrain<double> const&);

template float inner_product<float>(
  RuntimeTensorTrain<float> const&, RuntimeTensorTrain<float> const&);
template double inner_product<double>(
  RuntimeTensorTrain<double> const&, RuntimeTensorTrain<double> const&);

template float norm_frobenius<float>(RuntimeTensorTrain<float> const&);
template double norm_frobenius<double>(RuntimeTensorTrain<double> const&);

} // namespace boba_python
