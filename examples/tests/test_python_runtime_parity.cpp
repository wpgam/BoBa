// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Checks that the runtime-dimensional tensor train and DMRG cross used by the Python
  extension stay behaviorally equivalent to BoBa's native, dimension-templated
  implementations.

  The runtime code lives under python/bindings/detail/ because it is a private
  implementation detail of the Python extension and must not become part of BoBa's
  public C++ API. It depends only on BoBa, never on pybind11, so it can be exercised
  here. The two translation units are included directly because BoBa tests are built as
  a single translation unit by both the Makefile and CMake.

  TT representations are gauge dependent, and DMRGCross seeds its rank-kick step from
  std::random_device, so identical cores are not a meaningful invariant. What is checked
  is numerical and behavioral parity: reconstructed values, achieved accuracy, ranks and
  selection-strategy behavior.
*/

#include "../../python/bindings/detail/native_evaluator.hpp"
#include "../../python/bindings/detail/runtime_cross.cpp"
#include "../../python/bindings/detail/runtime_tensor_train.cpp"

constexpr boba::execution_space host_space = ::boba::host_space;

using boba_python::CrossOptions;
using boba_python::RuntimeTensorTrain;
using boba_python::SubmatrixSelection;

/// The target tensor: exactly TT-rank 2, so both implementations should nail it.
template <size_t dimension>
double target_function(::boba::Array<size_t, dimension> index)
{
  return 1.0 + static_cast<double>(::boba::sum(index));
}

double target_function_runtime(const size_t* index, size_t dimension)
{
  double total = 0.0;
  for (size_t d = 0; d < dimension; d++)
  {
    total += static_cast<double>(index[d]);
  }
  return 1.0 + total;
}

/// Copies a native train into the runtime representation so both start from the same guess.
template <size_t dimension>
RuntimeTensorTrain<double> to_runtime(
  ::boba::TensorTrain<dimension, host_space, double> const& native)
{
  std::vector<::boba::Tensor<3, host_space, double>> cores;
  cores.reserve(dimension);
  for (size_t d = 0; d < dimension; d++)
  {
    cores.push_back(native.cores[d]);
  }
  return RuntimeTensorTrain<double>(std::move(cores));
}

/// Largest absolute difference between the runtime train and the target over all entries.
template <size_t dimension>
double runtime_max_error(
  RuntimeTensorTrain<double> const& train,
  ::boba::Array<size_t, dimension> sizes)
{
  ::boba::Multiindexer<dimension> indexer(sizes);
  double worst = 0.0;
  std::vector<size_t> index(dimension);

  for (size_t flat = 0; flat < indexer.size(); flat++)
  {
    auto multi = indexer.multiindex(flat);
    for (size_t d = 0; d < dimension; d++)
    {
      index[d] = multi[d];
    }
    const double expected = target_function<dimension>(multi);
    worst = ::boba::max(worst, ::boba::abs(train.entry(index) - expected));
  }
  return worst;
}

/// Largest absolute difference between a native train and the target over all entries.
template <size_t dimension>
double native_max_error(
  ::boba::TensorTrain<dimension, host_space, double> const& train,
  ::boba::Array<size_t, dimension> sizes)
{
  ::boba::Multiindexer<dimension> indexer(sizes);
  double worst = 0.0;

  for (size_t flat = 0; flat < indexer.size(); flat++)
  {
    auto multi = indexer.multiindex(flat);
    const double expected = target_function<dimension>(multi);
    worst = ::boba::max(worst, ::boba::abs(train.unroll_value(multi) - expected));
  }
  return worst;
}

// ---------------------------------------------------------------------------
// Compression parity
// ---------------------------------------------------------------------------

/*
  TT-SVD is deterministic, so native compress and the runtime adaptation should agree to
  roundoff, core by core.
*/
template <size_t dimension>
bool check_compression_parity(::boba::Array<size_t, dimension> sizes)
{
  bool check = true;
  std::cout << "\n=== Compression parity, dimension " << dimension << " ===" << std::endl;

  ::boba::Tensor<dimension, host_space, double> dense(sizes);
  {
    auto dense_view = dense.view();
    ::boba::loop<host_space, 1>(dense_view.size(), [=] __boba_host_device__(size_t flat)
    {
      dense_view(flat) = target_function<dimension>(dense_view.multiindex(flat));
    });
  }

  ::boba::TensorTrain<dimension, host_space, double> native(sizes);
  native.svd_tolerance_relative = 1.0e-12;
  native.svd_tolerance_absolute = 1.0e-12;
  native.compress(dense);

  std::vector<size_t> runtime_sizes(dimension);
  for (size_t d = 0; d < dimension; d++)
  {
    runtime_sizes[d] = sizes[d];
  }

  // The runtime routine consumes the same flat, column-major payload the native routine
  // folds its input tensor into.
  auto runtime = boba_python::compress_dense<double>(
    dense.const_data(), runtime_sizes, 1.0e-12, 1.0e-12, ::boba::highest_value<size_t>());

  auto native_ranks = native.ranks();
  auto runtime_ranks = runtime.ranks();

  bool ranks_match = (runtime_ranks.size() == dimension + 1);
  for (size_t d = 0; ranks_match && d < dimension + 1; d++)
  {
    ranks_match = ranks_match && (native_ranks[d] == runtime_ranks[d]);
  }
  pass_or_fail_bool(check, ranks_match);

  // Deterministic algorithm, so compare the cores entrywise, not just the values.
  double worst_core_difference = 0.0;
  for (size_t d = 0; d < dimension; d++)
  {
    auto const& native_core = native.cores[d];
    auto const& runtime_core = runtime.core(d);
    if (native_core.sizes() != runtime_core.sizes())
    {
      worst_core_difference = ::boba::highest_value<double>();
      break;
    }
    for (size_t flat = 0; flat < native_core.size(); flat++)
    {
      worst_core_difference = ::boba::max(
        worst_core_difference,
        ::boba::abs(native_core.const_data()[flat] - runtime_core.const_data()[flat]));
    }
  }

  const double core_tolerance = 1.0e-10;
  pass_or_fail(check, worst_core_difference, core_tolerance);

  const double reconstruction_tolerance = 1.0e-9;
  const double runtime_error = runtime_max_error<dimension>(runtime, sizes);
  pass_or_fail(check, runtime_error, reconstruction_tolerance);

  return check;
}

// ---------------------------------------------------------------------------
// Cross parity
// ---------------------------------------------------------------------------

template <size_t dimension>
bool check_cross_parity(
  ::boba::Array<size_t, dimension> sizes,
  size_t initial_rank,
  SubmatrixSelection selection)
{
  bool check = true;
  std::cout << "\n=== Cross parity, dimension " << dimension
            << ", selection "
            << ((selection == SubmatrixSelection::DEIM) ? "DEIM" : "MAXVOL")
            << " ===" << std::endl;

  // One initial guess, shared by both implementations.
  ::boba::TensorTrain<dimension, host_space, double> initial_guess(sizes);
  for (size_t d = 0; d < dimension; d++)
  {
    const size_t left_rank = (d == 0) ? 1 : initial_rank;
    const size_t right_rank = (d + 1 == dimension) ? 1 : initial_rank;
    initial_guess.cores[d].resize({left_rank, sizes[d], right_rank});
    initial_guess.cores[d].fill_with_random();
  }

  auto runtime_guess = to_runtime<dimension>(initial_guess);

  boba::DMRGCross<double> native_cross;
  native_cross.convergence_tolerance = 1.0e-10;
  native_cross.max_sweeps = 10;
  native_cross.kickrank = 2;
  native_cross.submatrix_selection_type =
    (selection == SubmatrixSelection::DEIM)
      ? boba::DMRGCross<double>::SubmatrixSelectionType::DEIM
      : boba::DMRGCross<double>::SubmatrixSelectionType::MAXVOL;

  auto native_result = native_cross.apply(initial_guess, [=](::boba::Array<size_t, dimension> index)
  {
    return target_function<dimension>(index);
  });

  CrossOptions options;
  options.tolerance = 1.0e-10;
  options.n_sweeps = 10;
  options.kick_rank = 2;
  options.selection = selection;

  auto evaluator = boba_python::make_native_evaluator<double>(&target_function_runtime);
  auto runtime_result = boba_python::runtime_dmrg_cross<double>(runtime_guess, evaluator, options);

  // Both must reach the same approximation quality on the same target.
  const double accuracy_tolerance = 1.0e-8;
  const double native_error = native_max_error<dimension>(native_result, sizes);
  const double runtime_error = runtime_max_error<dimension>(runtime_result, sizes);
  pass_or_fail(check, native_error, accuracy_tolerance);
  pass_or_fail(check, runtime_error, accuracy_tolerance);

  // Ranks are deterministic for this exactly-low-rank target once both converge.
  auto native_ranks = native_result.ranks();
  auto runtime_ranks = runtime_result.ranks();
  bool ranks_match = (runtime_ranks.size() == dimension + 1);
  for (size_t d = 0; ranks_match && d < dimension + 1; d++)
  {
    ranks_match = ranks_match && (native_ranks[d] == runtime_ranks[d]);
  }
  pass_or_fail_bool(check, ranks_match);

  // The two trains must agree with each other, not merely with the target.
  double worst_pairwise = 0.0;
  {
    ::boba::Multiindexer<dimension> indexer(sizes);
    std::vector<size_t> index(dimension);
    for (size_t flat = 0; flat < indexer.size(); flat++)
    {
      auto multi = indexer.multiindex(flat);
      for (size_t d = 0; d < dimension; d++)
      {
        index[d] = multi[d];
      }
      worst_pairwise = ::boba::max(
        worst_pairwise,
        ::boba::abs(native_result.unroll_value(multi) - runtime_result.entry(index)));
    }
  }
  pass_or_fail(check, worst_pairwise, accuracy_tolerance);

  return check;
}

int main(int argc, char* argv[])
{
  boba::detail::ignore(argc);
  boba::detail::ignore(argv);
  boba::splash();
  boba::init();
  boba_print("Parity between BoBa's native tensor train / DMRG cross and the "
             "runtime-dimensional versions used by the Python bindings");

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;

  check = check_compression_parity<2>({5, 7}) && check;
  check = check_compression_parity<3>({5, 7, 4}) && check;
  check = check_compression_parity<4>({3, 5, 4, 6}) && check;

  check = check_cross_parity<2>({6, 7}, 2, SubmatrixSelection::MAXVOL) && check;
  check = check_cross_parity<3>({5, 7, 4}, 2, SubmatrixSelection::MAXVOL) && check;
  check = check_cross_parity<4>({4, 5, 3, 6}, 2, SubmatrixSelection::MAXVOL) && check;
  check = check_cross_parity<3>({5, 7, 4}, 2, SubmatrixSelection::DEIM) && check;

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Runtime-dimensional compression and cross match native BoBa" << std::endl;

  boba::finalize();
  return final_check(check);
}
