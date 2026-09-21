// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "py_tensor_train.hpp"

#include "detail/errors.hpp"
#include "detail/numpy_interop.hpp"
#include "detail/python_evaluator.hpp"
#include "detail/runtime_cross.hpp"
#include "detail/runtime_tensor_train.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace boba_python
{

namespace
{

/**
 * \brief Largest interface rank a tensor train of \p shape can carry at interface
 *        \p interface_id, counting interfaces from 1.
 *
 * Interface `i` separates modes `[0, i)` from `[i, d)`, so its rank cannot exceed the
 * smaller of the two products.
 */
std::size_t maximum_interface_rank(
  std::vector<std::size_t> const& shape,
  std::size_t interface_id)
{
  const std::size_t left = saturating_product(shape, 0, interface_id);
  const std::size_t right = saturating_product(shape, interface_id, shape.size());
  return std::min(left, right);
}

/**
 * \brief Turns the `initial_rank` argument into the `ndim - 1` interior ranks.
 *
 * Accepts either a single positive integer, applied to every interior interface, or a
 * sequence of exactly `ndim - 1` positive integers. Ranks larger than the
 * mathematically attainable interface dimension are rejected rather than silently
 * truncated, so an impossible request never turns into a surprising result.
 */
std::vector<std::size_t> parse_initial_ranks(
  py::handle initial_rank,
  std::vector<std::size_t> const& shape)
{
  const std::size_t ndim = shape.size();
  const std::size_t internal_count = ndim - 1;

  std::vector<std::size_t> ranks;

  const bool is_sequence =
    py::isinstance<py::sequence>(initial_rank) && !py::isinstance<py::str>(initial_rank);

  if (is_sequence)
  {
    auto sequence = py::reinterpret_borrow<py::sequence>(initial_rank);
    if (py::len(sequence) != internal_count)
    {
      throw py::value_error(
        "initial_rank sequence must give the " + std::to_string(internal_count) +
        " interior ranks for a " + std::to_string(ndim) + "-dimensional shape; got " +
        std::to_string(py::len(sequence)));
    }
    ranks.resize(internal_count);
    for (std::size_t i = 0; i < internal_count; i++)
    {
      py::handle item = sequence[i];
      if (!PyIndex_Check(item.ptr()))
      {
        throw py::type_error("initial_rank entries must be integers");
      }
      const long long value = item.cast<long long>();
      if (value <= 0)
      {
        throw py::value_error(
          "initial_rank entries must be positive; got " + std::to_string(value) +
          " at position " + std::to_string(i));
      }
      ranks[i] = static_cast<std::size_t>(value);
    }
  }
  else
  {
    if (!PyIndex_Check(initial_rank.ptr()))
    {
      throw py::type_error(
        "initial_rank must be a positive integer or a sequence of positive integers");
    }
    const long long value = initial_rank.cast<long long>();
    if (value <= 0)
    {
      throw py::value_error("initial_rank must be positive; got " + std::to_string(value));
    }
    ranks.assign(internal_count, static_cast<std::size_t>(value));
  }

  for (std::size_t i = 0; i < internal_count; i++)
  {
    const std::size_t bound = maximum_interface_rank(shape, i + 1);
    if (ranks[i] > bound)
    {
      throw py::value_error(
        "initial_rank " + std::to_string(ranks[i]) + " at interface " +
        std::to_string(i + 1) + " exceeds the largest attainable rank " +
        std::to_string(bound) + " for shape of that split");
    }
  }

  return ranks;
}

SubmatrixSelection parse_selection(std::string const& selection)
{
  if (selection == "maxvol")
  {
    return SubmatrixSelection::MAXVOL;
  }
  if (selection == "deim")
  {
    return SubmatrixSelection::DEIM;
  }
  throw py::value_error(
    "unknown selection '" + selection + "'; expected 'maxvol' or 'deim'");
}

/**
 * \brief Exact one-dimensional case.
 *
 * The native two-site DMRG sweep needs at least two cores. For a single mode the exact
 * tensor train is just the sampled vector, so it is built directly rather than refused.
 */
template <typename data_t>
RuntimeTensorTrain<data_t> cross_one_dimensional(
  std::size_t extent,
  Evaluator<data_t>& evaluator)
{
  ::boba::Tensor<3, python_space, data_t> core({1, extent, 1});

  const std::size_t chunk = std::min<std::size_t>(
    std::max<std::size_t>(evaluator.preferred_batch_size(), 1), extent);

  std::vector<std::size_t> index_batch(chunk);
  std::vector<data_t> value_batch(chunk);

  for (std::size_t base = 0; base < extent; base += chunk)
  {
    const std::size_t batch_count = std::min(chunk, extent - base);
    for (std::size_t b = 0; b < batch_count; b++)
    {
      index_batch[b] = base + b;
    }
    evaluator.evaluate(index_batch.data(), batch_count, 1, value_batch.data());
    for (std::size_t b = 0; b < batch_count; b++)
    {
      core({0, base + b, 0}) = value_batch[b];
    }
  }

  std::vector<::boba::Tensor<3, python_space, data_t>> cores;
  cores.push_back(std::move(core));
  return RuntimeTensorTrain<data_t>(std::move(cores));
}

template <typename data_t>
PyTensorTrain cross_typed(
  std::vector<std::size_t> const& shape,
  std::vector<std::size_t> const& internal_ranks,
  py::object const& function,
  bool vectorized,
  CrossOptions const& options)
{
  // vectorized is an explicit switch: the callback is never probed to guess its shape.
  std::unique_ptr<PythonEvaluator<data_t>> evaluator;
  if (vectorized)
  {
    evaluator = std::make_unique<VectorizedPythonEvaluator<data_t>>(function);
  }
  else
  {
    evaluator = std::make_unique<ScalarPythonEvaluator<data_t>>(function);
  }

  RuntimeTensorTrain<data_t> result;

  try
  {
    // The sweep is pure C++ and may run for a long time, so the GIL is dropped for its
    // duration. The evaluators reacquire it for the duration of each callback batch.
    py::gil_scoped_release release;

    if (shape.size() == 1)
    {
      result = cross_one_dimensional<data_t>(shape[0], *evaluator);
    }
    else
    {
      auto initial_guess = make_random_initial_guess<data_t>(shape, internal_ranks);
      result = runtime_dmrg_cross<data_t>(initial_guess, *evaluator, options);
    }
  }
  catch (PythonCallbackError const&)
  {
    // Back under the GIL: restore the user's original exception.
    evaluator->rethrow_if_failed();
    throw py::value_error("cross(): evaluator aborted without a Python exception");
  }

  return PyTensorTrain(std::move(result));
}

PyTensorTrain cross(
  py::object const& shape_like,
  py::object const& function,
  py::object const& initial_rank,
  double tolerance,
  std::size_t n_sweeps,
  std::size_t kick_rank,
  std::string const& selection,
  bool vectorized,
  py::object const& dtype_like,
  bool verbose)
{
  auto shape = parse_shape(shape_like);

  if (!py::isinstance<py::function>(function) && !PyCallable_Check(function.ptr()))
  {
    throw py::type_error("cross() requires a callable function");
  }

  auto internal_ranks = parse_initial_ranks(initial_rank, shape);

  if (n_sweeps == 0)
  {
    throw py::value_error("n_sweeps must be at least 1");
  }
  if (!(tolerance > 0.0))
  {
    throw py::value_error("tolerance must be positive");
  }

  CrossOptions options;
  options.tolerance = tolerance;
  options.n_sweeps = n_sweeps;
  options.kick_rank = kick_rank;
  options.selection = parse_selection(selection);
  options.verbose = verbose;

  // dtype defaults to float64 because cross has no input array to infer from.
  const ScalarKind kind =
    dtype_like.is_none() ? ScalarKind::Float64 : scalar_kind_from_dtype(dtype_like);
  if (kind == ScalarKind::Float32)
  {
    return cross_typed<float>(shape, internal_ranks, function, vectorized, options);
  }
  return cross_typed<double>(shape, internal_ranks, function, vectorized, options);
}

} // namespace

void register_cross(py::module_& module)
{
  module.def(
    "cross",
    &cross,
    py::arg("shape"),
    py::arg("function"),
    py::kw_only(),
    py::arg("initial_rank") = 2,
    py::arg("tolerance") = 1.0e-7,
    py::arg("n_sweeps") = 10,
    py::arg("kick_rank") = 2,
    py::arg("selection") = "maxvol",
    py::arg("vectorized") = false,
    py::arg("dtype") = py::dtype::of<double>(),
    py::arg("verbose") = false,
    R"doc(
Approximate a callable tensor with DMRG cross approximation.

Adapts BoBa's ``DMRGCross`` sweep to a dimension that is only known at run time, while
reusing BoBa's QR, SVD, MAXVOL, DEIM and backsolve kernels unchanged. The target tensor
is only ever sampled at the indices the sweep selects; it is never materialized.

Parameters
----------
shape : sequence of int
    Mode extents of the tensor to approximate.
function : callable
    Entry evaluator. With ``vectorized=False`` it is called as ``function(index)`` where
    ``index`` is a tuple of zero-based Python integers of length ``len(shape)``, and must
    return a real scalar. With ``vectorized=True`` it is called as ``function(indices)``
    where ``indices`` is a C-contiguous ``int64`` array of shape ``(N, len(shape))``, and
    must return ``N`` real values. The mode is never inferred by probing the callable.
initial_rank : int or sequence of int, optional
    Ranks of the random starting train. An integer is applied to every interior
    interface. A sequence must give exactly ``len(shape) - 1`` interior ranks; the unit
    boundary ranks are implicit. Ranks above the attainable interface dimension raise
    ``ValueError``.
tolerance : float, optional
    Sweep convergence threshold on the maximum local relative error. Native default.
n_sweeps : int, optional
    Sweep budget, mapped to ``DMRGCross::max_sweeps``. Note that the native loop
    condition is ``sweep_count < max_sweeps`` starting from 1, so ``n_sweeps=1``
    performs setup only.
kick_rank : int, optional
    Random rank enrichment per step, mapped to ``DMRGCross::kickrank``.
selection : {'maxvol', 'deim'}, optional
    Submatrix selection strategy. ``'maxvol'`` matches the native default.
vectorized : bool, optional
    Selects the batched callback protocol.
dtype : numpy dtype, optional
    ``float32`` or ``float64``. Defaults to ``float64``.
verbose : bool, optional
    Forwarded to ``DMRGCross::verbose``; prints per-step diagnostics to stdout.

Returns
-------
TensorTrain

Notes
-----
A single-mode ``shape`` is handled exactly by sampling the whole vector, since the
two-site sweep itself requires at least two cores.

Exceptions raised inside ``function`` propagate unchanged to the caller and abort the
sweep cleanly.
)doc");
}

} // namespace boba_python
