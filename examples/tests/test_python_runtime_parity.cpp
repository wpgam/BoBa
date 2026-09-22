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
#include "../../python/bindings/detail/runtime_algebra.cpp"
#include "../../python/bindings/detail/runtime_cross.cpp"
#include "../../python/bindings/detail/runtime_structure.cpp"
#include "../../python/bindings/detail/runtime_rounding.cpp"
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

// ---------------------------------------------------------------------------
// Rounding parity
// ---------------------------------------------------------------------------

/*
  Builds a train with deliberately redundant ranks by adding a compressed train to a
  copy of itself. TT addition is exact and concatenates ranks, so the result represents
  2*T at double the ranks -- exactly the situation rounding exists to clean up, and the
  situation every later exact-arithmetic operation will produce.
*/
template <size_t dimension, typename function_t>
::boba::TensorTrain<dimension, host_space, double> compress_native(
  ::boba::Array<size_t, dimension> sizes,
  function_t function)
{
  ::boba::Tensor<dimension, host_space, double> dense(sizes);
  {
    auto dense_view = dense.view();
    ::boba::loop<host_space, 1>(dense_view.size(), [=] __boba_host_device__(size_t flat)
    {
      dense_view(flat) = function(dense_view.multiindex(flat));
    });
  }

  ::boba::TensorTrain<dimension, host_space, double> train(sizes);
  train.svd_tolerance_relative = 1.0e-12;
  train.svd_tolerance_absolute = 1.0e-12;
  train.compress(dense);
  return train;
}

template <size_t dimension>
::boba::TensorTrain<dimension, host_space, double> make_rank_inflated_train(
  ::boba::Array<size_t, dimension> sizes)
{
  auto train = compress_native<dimension>(sizes, [] __boba_host_device__(
    ::boba::Array<size_t, dimension> index)
  {
    return target_function<dimension>(index);
  });

  ::boba::TensorTrain<dimension, host_space, double> addend(train);
  train += addend;
  return train;
}

/// Largest entrywise difference between a native train's cores and a runtime train's.
template <size_t dimension>
double worst_core_difference(
  ::boba::TensorTrain<dimension, host_space, double> const& native,
  RuntimeTensorTrain<double> const& runtime)
{
  double worst = 0.0;
  for (size_t d = 0; d < dimension; d++)
  {
    auto const& native_core = native.cores[d];
    auto const& runtime_core = runtime.core(d);
    if (native_core.sizes() != runtime_core.sizes())
    {
      return ::boba::highest_value<double>();
    }
    for (size_t flat = 0; flat < native_core.size(); flat++)
    {
      worst = ::boba::max(
        worst,
        ::boba::abs(native_core.const_data()[flat] - runtime_core.const_data()[flat]));
    }
  }
  return worst;
}

/*
  The QR sweep is deterministic, so native orthogonalize and the runtime adaptation
  should agree core by core, not merely represent the same tensor.
*/
template <size_t dimension>
bool check_orthogonalization_parity(::boba::Array<size_t, dimension> sizes)
{
  bool check = true;
  std::cout << "\n=== Orthogonalization parity, dimension " << dimension << " ===" << std::endl;

  auto native = make_rank_inflated_train<dimension>(sizes);
  auto runtime = to_runtime<dimension>(native);

  native.orthogonalize();
  boba_python::orthogonalize(runtime);

  pass_or_fail(check, worst_core_difference<dimension>(native, runtime), 1.0e-10);
  return check;
}

/*
  Rounding is a QR sweep followed by a truncated SVD sweep, both deterministic. Starting
  from the same rank-inflated train, the two implementations should produce the same
  ranks and the same cores.
*/
template <size_t dimension>
bool check_rounding_parity(::boba::Array<size_t, dimension> sizes, size_t max_rank)
{
  bool check = true;
  std::cout << "\n=== Rounding parity, dimension " << dimension << ", max_rank "
            << max_rank << " ===" << std::endl;

  auto native = make_rank_inflated_train<dimension>(sizes);
  auto runtime = to_runtime<dimension>(native);

  const double tolerance = 1.0e-12;

  native.svd_tolerance_relative = tolerance;
  native.svd_tolerance_absolute = tolerance;
  native.max_ranks = ::boba::filled_array<dimension + 1>(max_rank);
  native.round();

  std::vector<size_t> bounds(dimension + 1, max_rank);
  boba_python::round(runtime, tolerance, tolerance, bounds);

  auto native_ranks = native.ranks();
  auto runtime_ranks = runtime.ranks();

  bool ranks_match = (runtime_ranks.size() == dimension + 1);
  for (size_t d = 0; ranks_match && d < dimension + 1; d++)
  {
    ranks_match = ranks_match && (native_ranks[d] == runtime_ranks[d]);
  }
  pass_or_fail_bool(check, ranks_match);

  pass_or_fail(check, worst_core_difference<dimension>(native, runtime), 1.0e-10);

  // The rounded train still represents 2 * target, so check the values it stands for as
  // well as the factorization it stores.
  double worst_value = 0.0;
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
      const double expected = 2.0 * target_function<dimension>(multi);
      worst_value = ::boba::max(worst_value, ::boba::abs(runtime.entry(index) - expected));
    }
  }
  pass_or_fail(check, worst_value, 1.0e-9);

  return check;
}

// ---------------------------------------------------------------------------
// Exact arithmetic parity
// ---------------------------------------------------------------------------

/// A second low-rank target, independent of target_function, for two-operand checks.
template <size_t dimension>
double other_target_function(::boba::Array<size_t, dimension> index)
{
  double total = 1.0;
  for (size_t d = 0; d < dimension; d++)
  {
    total *= (2.0 + static_cast<double>(index[d]) / (1.0 + static_cast<double>(d)));
  }
  return total;
}

/// Relative difference between two scalars, safe when the reference is zero.
double relative_difference(double reference, double candidate)
{
  const double scale = ::boba::max(::boba::abs(reference), 1.0);
  return ::boba::abs(reference - candidate) / scale;
}

/*
  Exact TT addition and the Hadamard product are deterministic and involve no truncation,
  so native BoBa and the runtime adaptation should produce identical cores, not merely
  equivalent tensors. The scalar reductions are compared relatively.
*/
template <size_t dimension>
bool check_algebra_parity(::boba::Array<size_t, dimension> sizes)
{
  bool check = true;
  std::cout << "\n=== Exact arithmetic parity, dimension " << dimension << " ===" << std::endl;

  auto native_a = compress_native<dimension>(sizes, [] __boba_host_device__(
    ::boba::Array<size_t, dimension> index)
  {
    return target_function<dimension>(index);
  });
  auto native_b = compress_native<dimension>(sizes, [] __boba_host_device__(
    ::boba::Array<size_t, dimension> index)
  {
    return other_target_function<dimension>(index);
  });

  auto runtime_a = to_runtime<dimension>(native_a);
  auto runtime_b = to_runtime<dimension>(native_b);

  // Addition: ranks concatenate, so the result ranks must match as well as the cores.
  {
    auto native_sum = native_a + native_b;
    auto runtime_sum = boba_python::add(runtime_a, runtime_b);

    auto native_ranks = native_sum.ranks();
    auto runtime_ranks = runtime_sum.ranks();
    bool ranks_match = (runtime_ranks.size() == dimension + 1);
    for (size_t d = 0; ranks_match && d < dimension + 1; d++)
    {
      ranks_match = ranks_match && (native_ranks[d] == runtime_ranks[d]);
    }
    pass_or_fail_bool(check, ranks_match);
    pass_or_fail(check, worst_core_difference<dimension>(native_sum, runtime_sum), 1.0e-12);
  }

  // Hadamard product: ranks multiply.
  {
    auto native_product = ::boba::elementwise_product(native_a, native_b);
    auto runtime_product = boba_python::hadamard(runtime_a, runtime_b);

    auto native_ranks = native_product.ranks();
    auto runtime_ranks = runtime_product.ranks();
    bool ranks_match = (runtime_ranks.size() == dimension + 1);
    for (size_t d = 0; ranks_match && d < dimension + 1; d++)
    {
      ranks_match = ranks_match && (native_ranks[d] == runtime_ranks[d]);
    }
    pass_or_fail_bool(check, ranks_match);
    pass_or_fail(
      check, worst_core_difference<dimension>(native_product, runtime_product), 1.0e-12);
  }

  // Scalar multiplication is absorbed into the first core by both implementations.
  {
    auto native_scaled = native_a * (-2.5);
    auto runtime_scaled = boba_python::scale(runtime_a, -2.5);
    pass_or_fail(
      check, worst_core_difference<dimension>(native_scaled, runtime_scaled), 1.0e-12);
  }

  // Reductions.
  {
    const double native_inner = native_a.inner_product(native_b);
    const double runtime_inner = boba_python::inner_product(runtime_a, runtime_b);
    pass_or_fail(check, relative_difference(native_inner, runtime_inner), 1.0e-12);

    const double native_norm = ::boba::norm_frobenius(native_a);
    const double runtime_norm = boba_python::norm_frobenius(runtime_a);
    pass_or_fail(check, relative_difference(native_norm, runtime_norm), 1.0e-12);
  }

  return check;
}

// ---------------------------------------------------------------------------
// Structural parity
// ---------------------------------------------------------------------------

/*
  Concatenation and slicing have no native counterpart to compare against core for core,
  so they are checked against independent native constructions instead.

  For concatenation the reference is built the way BoBa itself would express a join: a
  zero train of the enlarged shape, with each operand added into a disjoint mode range by
  add_subtrain. That produces the same tensor by a different route, through BoBa's own
  subtrain arithmetic. It is not how concatenate_mode is implemented, and deliberately
  so: seeding from a rank-one zero train leaves one spurious rank behind, which is the
  reason the runtime version builds the cores directly. The check therefore compares
  values, and separately asserts that the runtime ranks are exactly the sums of the
  operands ranks, with no spurious extra.

  For slicing the reference is the native entry evaluator at shifted indices, since
  BoBa cannot extract a dense subtensor at all: unroll_subtensor forwards to
  partial_decompress_core, which asserts on any partial range (issue 238).
*/
template <size_t dimension>
bool check_structure_parity(::boba::Array<size_t, dimension> sizes, size_t mode)
{
  bool check = true;
  std::cout << "\n=== Structural parity, dimension " << dimension << ", mode " << mode
            << " ===" << std::endl;

  auto native_a = compress_native<dimension>(sizes, [] __boba_host_device__(
    ::boba::Array<size_t, dimension> index)
  {
    return target_function<dimension>(index);
  });
  auto native_b = compress_native<dimension>(sizes, [] __boba_host_device__(
    ::boba::Array<size_t, dimension> index)
  {
    return other_target_function<dimension>(index);
  });

  auto runtime_a = to_runtime<dimension>(native_a);
  auto runtime_b = to_runtime<dimension>(native_b);

  // ---- concatenation ----
  {
    ::boba::Array<size_t, dimension> enlarged_sizes = sizes;
    enlarged_sizes[mode] = sizes[mode] * 2;

    ::boba::TensorTrain<dimension, host_space, double> native_joined(enlarged_sizes);
    native_joined.fill_with_zeros();

    auto offsets = ::boba::filled_array<dimension>(static_cast<size_t>(0));
    native_joined.add_subtrain(native_a, offsets);
    offsets[mode] = sizes[mode];
    native_joined.add_subtrain(native_b, offsets);

    auto runtime_joined = boba_python::concatenate_mode(runtime_a, runtime_b, mode);

    // Ranks must be exactly the sums, with no seed rank carried along.
    auto a_ranks = native_a.ranks();
    auto b_ranks = native_b.ranks();
    auto joined_ranks = runtime_joined.ranks();

    bool ranks_exact = (joined_ranks.size() == dimension + 1);
    ranks_exact = ranks_exact && (joined_ranks.front() == 1) && (joined_ranks.back() == 1);
    for (size_t d = 1; ranks_exact && d < dimension; d++)
    {
      ranks_exact = ranks_exact && (joined_ranks[d] == a_ranks[d] + b_ranks[d]);
    }
    pass_or_fail_bool(check, ranks_exact);

    double worst_joined = 0.0;
    {
      ::boba::Multiindexer<dimension> indexer(enlarged_sizes);
      std::vector<size_t> index(dimension);
      for (size_t flat = 0; flat < indexer.size(); flat++)
      {
        auto multi = indexer.multiindex(flat);
        for (size_t d = 0; d < dimension; d++)
        {
          index[d] = multi[d];
        }
        worst_joined = ::boba::max(
          worst_joined,
          ::boba::abs(native_joined.unroll_value(multi) - runtime_joined.entry(index)));
      }
    }
    pass_or_fail(check, worst_joined, 1.0e-10);
  }

  // ---- slicing ----
  {
    // Drop the first index of the chosen mode, keep everything else whole.
    std::vector<boba_python::ModeSelection> selections(dimension);
    for (size_t d = 0; d < dimension; d++)
    {
      selections[d].drop = false;
      selections[d].start = (d == mode) ? 1 : 0;
      selections[d].step = 1;
      selections[d].count = (d == mode) ? sizes[d] - 1 : sizes[d];
    }

    auto runtime_sliced = boba_python::slice_modes(runtime_a, selections);

    // Restricting a mode cannot couple the factors, so the ranks must be untouched.
    auto original_ranks = runtime_a.ranks();
    auto sliced_ranks = runtime_sliced.ranks();
    bool ranks_unchanged = (sliced_ranks.size() == original_ranks.size());
    for (size_t d = 0; ranks_unchanged && d < sliced_ranks.size(); d++)
    {
      ranks_unchanged = ranks_unchanged && (sliced_ranks[d] == original_ranks[d]);
    }
    pass_or_fail_bool(check, ranks_unchanged);

    ::boba::Array<size_t, dimension> sliced_sizes = sizes;
    sliced_sizes[mode] = sizes[mode] - 1;

    double worst_sliced = 0.0;
    {
      ::boba::Multiindexer<dimension> indexer(sliced_sizes);
      std::vector<size_t> index(dimension);
      for (size_t flat = 0; flat < indexer.size(); flat++)
      {
        auto multi = indexer.multiindex(flat);
        for (size_t d = 0; d < dimension; d++)
        {
          index[d] = multi[d];
        }

        auto source = multi;
        source[mode] = multi[mode] + 1;

        worst_sliced = ::boba::max(
          worst_sliced,
          ::boba::abs(native_a.unroll_value(source) - runtime_sliced.entry(index)));
      }
    }
    pass_or_fail(check, worst_sliced, 1.0e-10);
  }

  return check;
}

// ---------------------------------------------------------------------------
// Cross over a pointwise function of existing trains
// ---------------------------------------------------------------------------

/*
  The Python interface can cross-approximate f(A, B, ...) for trains that already exist.
  The entry source behind that feature looks each input train up at the requested index
  and then applies f; only the application of f involves Python, and this check exercises
  everything else.

  The target is chosen so the right answer is known in closed form: for f(x, y) = x * y
  the result is the Hadamard product, which the exact arithmetic already computes. So the
  cross approximation is compared against an independently computed exact train rather
  than against a tolerance on itself. That also cross-checks the two features against each
  other, since an error in either would break the agreement.
*/
template <size_t dimension>
bool check_function_cross_parity(
  ::boba::Array<size_t, dimension> sizes,
  std::vector<size_t> internal_ranks)
{
  bool check = true;
  std::cout << "\n=== Cross over a function of trains, dimension " << dimension
            << " ===" << std::endl;

  auto native_a = compress_native<dimension>(sizes, [] __boba_host_device__(
    ::boba::Array<size_t, dimension> index)
  {
    return target_function<dimension>(index);
  });
  auto native_b = compress_native<dimension>(sizes, [] __boba_host_device__(
    ::boba::Array<size_t, dimension> index)
  {
    return other_target_function<dimension>(index);
  });

  auto runtime_a = to_runtime<dimension>(native_a);
  auto runtime_b = to_runtime<dimension>(native_b);

  // The exact answer for an entrywise product.
  auto exact = boba_python::hadamard(runtime_a, runtime_b);

  // The entry source used by cross_function, with the Python callback replaced by the
  // same arithmetic in C++.
  auto evaluator = boba_python::make_native_evaluator<double>(
    [&runtime_a, &runtime_b](const size_t* index, size_t ndim)
  {
    std::span<const size_t> multi(index, ndim);
    return runtime_a.entry(multi) * runtime_b.entry(multi);
  });

  std::vector<size_t> shape(dimension);
  for (size_t d = 0; d < dimension; d++)
  {
    shape[d] = sizes[d];
  }

  auto initial_guess = boba_python::make_random_initial_guess<double>(shape, internal_ranks);

  CrossOptions options;
  options.tolerance = 1.0e-12;
  options.n_sweeps = 20;
  options.kick_rank = 2;
  options.selection = SubmatrixSelection::MAXVOL;

  auto approximated = boba_python::runtime_dmrg_cross<double>(initial_guess, evaluator, options);

  double worst = 0.0;
  double scale = 0.0;
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
      const double expected = exact.entry(index);
      scale = ::boba::max(scale, ::boba::abs(expected));
      worst = ::boba::max(worst, ::boba::abs(expected - approximated.entry(index)));
    }
  }

  pass_or_fail(check, worst / ::boba::max(scale, 1.0), 1.0e-8);

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

  check = check_orthogonalization_parity<2>({5, 7}) && check;
  check = check_orthogonalization_parity<3>({5, 7, 4}) && check;
  check = check_orthogonalization_parity<4>({3, 5, 4, 6}) && check;

  // Unbounded, so only the tolerance truncates; then a hard rank cap on the same input.
  check = check_rounding_parity<2>({5, 7}, ::boba::highest_value<size_t>()) && check;
  check = check_rounding_parity<3>({5, 7, 4}, ::boba::highest_value<size_t>()) && check;
  check = check_rounding_parity<4>({3, 5, 4, 6}, ::boba::highest_value<size_t>()) && check;
  check = check_rounding_parity<3>({5, 7, 4}, 2) && check;

  check = check_algebra_parity<2>({5, 7}) && check;
  check = check_algebra_parity<3>({5, 7, 4}) && check;
  check = check_algebra_parity<4>({3, 5, 4, 6}) && check;

  check = check_structure_parity<2>({5, 7}, 0) && check;
  check = check_structure_parity<2>({5, 7}, 1) && check;
  check = check_structure_parity<3>({5, 7, 4}, 1) && check;
  check = check_structure_parity<4>({3, 5, 4, 6}, 2) && check;

  // Interior ranks stay within what each interface can attain for the given shape.
  check = check_function_cross_parity<2>({5, 7}, {4}) && check;
  check = check_function_cross_parity<3>({5, 7, 4}, {4, 4}) && check;
  check = check_function_cross_parity<4>({3, 5, 4, 6}, {3, 4, 4}) && check;

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Runtime-dimensional compression, cross, orthogonalization, rounding and "
               "exact arithmetic match native BoBa; concatenation, slicing and "
               "cross over functions of trains agree with independent constructions" << std::endl;

  boba::finalize();
  return final_check(check);
}
