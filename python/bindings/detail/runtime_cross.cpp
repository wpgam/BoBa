// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime_cross.hpp"

#include "errors.hpp"

#include <algorithm>
#include <vector>

namespace boba_python
{

namespace
{

using ::boba::index_t;

template <typename data_t>
using matrix_t = ::boba::Matrix<python_space, data_t>;

using index_matrix_t = ::boba::Matrix<python_space, index_t>;

/**
 * \brief Runtime-dimensional replacement for
 *        boba::DMRGCross::create_score_tensor_from_index_and_function.
 *
 * The native helper builds one boba::Array<size_t, dimension> per score entry inside a
 * boba::loop and calls the user function once per entry. Here the same multi-indices
 * are staged into a flat row-major buffer and handed to the evaluator in batches, so a
 * vectorized Python callback sees many points per call. The index composition itself
 * follows the native helper exactly.
 *
 * Batches are capped at Evaluator::preferred_batch_size() rows, which bounds the
 * staging buffer by cap * dimension instead of by the score tensor size. The score
 * tensor is a local (r_left, n_d, n_d+1, r_right) block, never the dense tensor.
 */
template <typename data_t>
void fill_score_tensor(
  index_matrix_t const& left_indices,
  index_matrix_t const& right_indices,
  ::boba::Tensor<4, python_space, data_t>& score_tensor,
  std::size_t dimension,
  Evaluator<data_t>& evaluator)
{
  auto left_indices_view = left_indices.const_view();
  auto right_indices_view = right_indices.const_view();

  const std::size_t left_index_count = left_indices_view.sizes(1);
  const std::size_t first_right_index_offset = left_index_count + 2;
  const std::size_t full_index_count = first_right_index_offset + right_indices_view.sizes(0);

  auto score_tensor_view = score_tensor.view();
  const std::size_t entry_count = score_tensor_view.size();
  if (entry_count == 0)
  {
    return;
  }

  const std::size_t batch_cap = std::max<std::size_t>(evaluator.preferred_batch_size(), 1);
  const std::size_t chunk = std::min(batch_cap, entry_count);

  std::vector<std::size_t> index_batch(chunk * dimension);
  std::vector<data_t> value_batch(chunk);

  for (std::size_t base = 0; base < entry_count; base += chunk)
  {
    const std::size_t batch_count = std::min(chunk, entry_count - base);

    for (std::size_t b = 0; b < batch_count; b++)
    {
      auto [s1, i1, i2, s2] = score_tensor_view.multiindex(base + b);
      std::size_t* row = index_batch.data() + b * dimension;

      for (std::size_t left_dim = 0; left_dim < left_index_count; left_dim++)
      {
        row[left_dim] = left_indices_view({s1, left_dim});
      }
      row[left_index_count] = i1;
      row[left_index_count + 1] = i2;
      for (std::size_t right_dim = first_right_index_offset; right_dim < full_index_count; right_dim++)
      {
        row[right_dim] = right_indices_view({right_dim - first_right_index_offset, s2});
      }
    }

    evaluator.evaluate(index_batch.data(), batch_count, dimension, value_batch.data());

    for (std::size_t b = 0; b < batch_count; b++)
    {
      score_tensor_view(base + b) = value_batch[b];
    }
  }
}

} // namespace

template <typename data_t>
RuntimeTensorTrain<data_t> runtime_dmrg_cross(
  RuntimeTensorTrain<data_t> const& initial_guess,
  Evaluator<data_t>& evaluator,
  CrossOptions const& options)
{
  using native_cross_t = ::boba::DMRGCross<data_t>;

  // The dimension-independent pieces of the native algorithm are reused directly:
  // select_indices(), maxvol2(), DEIM() and reorthogonalize() are templated on execution
  // only, so this object supplies the real implementations rather than copies of them.
  native_cross_t native_cross;
  native_cross.verbose = options.verbose;
  native_cross.kickrank = options.kick_rank;
  native_cross.max_sweeps = options.n_sweeps;
  native_cross.convergence_tolerance =
    static_cast<typename native_cross_t::real_data_t>(options.tolerance);
  native_cross.submatrix_selection_type =
    (options.selection == SubmatrixSelection::DEIM)
      ? native_cross_t::SubmatrixSelectionType::DEIM
      : native_cross_t::SubmatrixSelectionType::MAXVOL;

  // Runtime stand-in for the native `dimension` template parameter.
  const std::size_t dimension = initial_guess.ndim();

  RuntimeTensorTrain<data_t> approximated_tt = initial_guess;

  auto mode_sizes = approximated_tt.shape();
  auto tt_ranks = approximated_tt.ranks();
  std::size_t sweep_count = 1;

  // Native: boba::Array<Matrix, dimension + 1>.
  std::vector<matrix_t<data_t>> transfer_matrices(dimension + 1);
  transfer_matrices[0].resize({1, 1});
  transfer_matrices[0].fill_with(static_cast<data_t>(1));
  transfer_matrices[dimension].resize({1, 1});
  transfer_matrices[dimension].fill_with(static_cast<data_t>(1));

  // Native: boba::Array<Matrix<index_t>, dimension + 1>.
  std::vector<index_matrix_t> interpolation_indices(dimension + 1);
  interpolation_indices[dimension].resize({0, tt_ranks[dimension]});
  interpolation_indices[0].resize({tt_ranks[0], 0});

  matrix_t<data_t> right_interface({1, 1});
  right_interface.fill_with(static_cast<data_t>(1));

  // ---------------------------------------------------------------------------
  // Right-to-left setup sweep (native "DMRGCross_core_setup").
  // ---------------------------------------------------------------------------
  for (std::size_t core_id = dimension - 1; core_id > 0; core_id--)
  {
    const auto mode_extent = approximated_tt.core(core_id).sizes(1);

    matrix_t<data_t> core_matrix = ::boba::reshape_to_matrix(
      approximated_tt.core(core_id),
      {tt_ranks[core_id] * mode_sizes[core_id], tt_ranks[core_id + 1]});

    auto core_times_right_interface = core_matrix * right_interface;
    core_matrix.resize({tt_ranks[core_id], mode_sizes[core_id] * tt_ranks[core_id + 1]});
    core_matrix.reshape(core_times_right_interface);
    core_matrix.transpose_in_place();

    ::boba::QR<python_space, data_t> qr;
    qr(core_matrix);

    core_matrix = qr.Q;
    auto triangular_factor = qr.R;

    auto selected_indices = native_cross.select_indices(qr.Q);

    index_matrix_t trailing_indices = interpolation_indices[core_id + 1];

    auto next_rank = ::boba::min(mode_extent * tt_ranks[core_id + 1], tt_ranks[core_id]);

    // Native allocates Matrix<index_t>({dimension - core_id, next_rank}); `dimension`
    // is simply a runtime value here.
    index_matrix_t leading_indices({dimension - core_id, next_rank});
    leading_indices.fill_with_zeros();

    for (std::size_t selected_column = 0; selected_column < next_rank; selected_column++)
    {
      index_t flattened_index = selected_indices({selected_column});
      ::boba::Multiindexer<2> unfolded_core_indices({tt_ranks[core_id + 1], mode_extent});
      auto [right_rank_index, mode_index] = unfolded_core_indices.multiindex(flattened_index);

      auto trailing_indices_view = trailing_indices.view();
      auto leading_indices_view = leading_indices.view();

      ::boba::loop<python_space, 1>(leading_indices.rows(),
                                    [=] __boba_host_device__(index_t i)
      {
        index_t value = 0;
        if (i == 0)
        {
          value = mode_index;
        }
        else
        {
          value = trailing_indices_view({i - 1, right_rank_index});
        }
        leading_indices_view({i, selected_column}) = value;
      });
    }

    interpolation_indices[core_id] = leading_indices;

    right_interface = core_matrix.extract_rows(selected_indices);

    {
      auto temp = ::boba::right_backsolve(right_interface, core_matrix);
      core_matrix = temp;
    }
    {
      auto temp = right_interface * triangular_factor;
      right_interface = temp;
    }
    right_interface.transpose_in_place();

    core_matrix.transpose_in_place();

    approximated_tt.core(core_id).resize(
      {tt_ranks[core_id], mode_sizes[core_id], tt_ranks[core_id + 1]});
    approximated_tt.core(core_id).reshape(core_matrix);

    core_matrix.reshape({tt_ranks[core_id] * mode_sizes[core_id], tt_ranks[core_id + 1]});
    {
      auto temp = core_matrix * transfer_matrices[core_id + 1];
      core_matrix = temp;
    }

    core_matrix.reshape({tt_ranks[core_id], mode_sizes[core_id] * tt_ranks[core_id + 1]});
    core_matrix.transpose_in_place();

    ::boba::QR<python_space, data_t> QR2;
    QR2(core_matrix);
    triangular_factor = QR2.R;
    transfer_matrices[core_id] = triangular_factor;
  }

  matrix_t<data_t> first_core_matrix({tt_ranks[0] * mode_sizes[0], tt_ranks[1]});
  {
    auto temp = approximated_tt.core(0);
    first_core_matrix.reshape(temp);
  }

  {
    ::boba::Tensor<3, python_space, data_t> temp({tt_ranks[0], mode_sizes[0], tt_ranks[1]});
    temp.reshape(first_core_matrix * right_interface);
    approximated_tt.core(0) = temp;
  }

  bool not_converged = true;
  bool sweep_right = true;

  std::size_t core_id = 0;
  auto max_relative_error = 0.0;

  // ---------------------------------------------------------------------------
  // Two-site DMRG sweep (native "DMRGCross_sweep").
  // ---------------------------------------------------------------------------
  while ((sweep_count < native_cross.max_sweeps) && not_converged)
  {
    auto left_core = approximated_tt.core(core_id);
    auto right_core = approximated_tt.core(core_id + 1);
    auto left_indices = interpolation_indices[core_id];
    auto right_indices = interpolation_indices[core_id + 2];

    ::boba::Tensor<4, python_space, data_t> score_tensor(
      {tt_ranks[core_id], mode_sizes[core_id], mode_sizes[core_id + 1], tt_ranks[core_id + 2]});

    fill_score_tensor(left_indices, right_indices, score_tensor, dimension, evaluator);

    matrix_t<data_t> score(
      {tt_ranks[core_id], mode_sizes[core_id] * mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});
    score.reshape(score_tensor);

    score = transfer_matrices[core_id] * score;
    tt_ranks[core_id] = score.sizes(0);

    score.reshape(
      {tt_ranks[core_id] * mode_sizes[core_id] * mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
    {
      auto temp = score * transfer_matrices[core_id + 2];
      score = temp;
    }
    tt_ranks[core_id + 2] = score.sizes(1);

    score.reshape(
      {tt_ranks[core_id] * mode_sizes[core_id], mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});

    ::boba::SVD<python_space, data_t> svd;
    // Mirrors native DMRGCross: the two-site block SVD carries its own thresholds,
    // deliberately independent of the top-level convergence tolerance. Reading them
    // from the native object keeps this in step with any future default change.
    svd.tolerance_relative = native_cross.two_site_svd_tolerance_relative;
    svd.tolerance_absolute = native_cross.two_site_svd_tolerance_absolute;
    svd(score);
    auto u = svd.U;
    auto s = svd.S;
    auto v = svd.V;
    auto r = svd.significant_singular_values;

    // for compatibility with tt-toolbox
    u *= -1.0;
    v *= -1.0;

    // Kick rank
    if (sweep_right)
    {
      ::boba::apply_as_diagonal_right_in_place(s, v);

      matrix_t<data_t> ur({u.rows(), native_cross.kickrank});
      ur.fill_with_zeros();

      matrix_t<data_t> random_enrichment({u.rows(), native_cross.kickrank});
      random_enrichment.fill_with_random();
      ur.reshape(random_enrichment);

      u = native_cross.reorthogonalize(u, ur);
      const auto rank_added = u.cols() - r;
      if (rank_added > 0)
      {
        matrix_t<data_t> vr({v.rows(), rank_added});
        vr.fill_with_zeros();
        auto v_temp = ::boba::concatenate_columns(v, vr);
        v = v_temp;
      }
      r = r + rank_added;
    }
    else
    {
      ::boba::apply_as_diagonal_right_in_place(s, u);

      matrix_t<data_t> vr({v.rows(), native_cross.kickrank});
      vr.fill_with_zeros();

      matrix_t<data_t> random_enrichment({v.rows(), native_cross.kickrank});
      random_enrichment.fill_with_random();

      vr.reshape(random_enrichment);
      v = native_cross.reorthogonalize(v, vr);

      const auto rank_added = v.cols() - r;
      if (rank_added > 0)
      {
        matrix_t<data_t> ur({u.rows(), rank_added});
        ur.fill_with_zeros();
        auto u_temp = ::boba::concatenate_columns(u, ur);
        u = u_temp;
      }
      r = r + rank_added;
    }

    v.transpose_in_place();

    matrix_t<data_t> left_core_matrix(
      {left_core.size() / tt_ranks[core_id + 1], tt_ranks[core_id + 1]});
    left_core_matrix.reshape(left_core);

    matrix_t<data_t> right_core_matrix(
      {tt_ranks[core_id + 1], right_core.size() / tt_ranks[core_id + 1]});
    right_core_matrix.reshape(right_core);

    auto approximation = left_core_matrix * right_core_matrix;
    approximation.reshape(
      {tt_ranks[core_id], mode_sizes[core_id] * mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});
    approximation = transfer_matrices[core_id] * approximation;
    approximation.reshape(
      {tt_ranks[core_id] * mode_sizes[core_id] * mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
    approximation = approximation * transfer_matrices[core_id + 2];

    auto norm_diff =
      ::boba::norm_difference_frobenius(::boba::flatten(score), ::boba::flatten(approximation));
    auto norm_score = ::boba::norm_frobenius(score);
    auto local_relative_error = norm_diff / norm_score;
    max_relative_error = ::boba::max(max_relative_error, static_cast<double>(local_relative_error));

    if (native_cross.verbose)
    {
      boba_print(sweep_count);
      boba_print(core_id);
      boba_print(r);
      boba_print(local_relative_error);
    }

    tt_ranks[core_id + 1] = r;
    u.reshape({tt_ranks[core_id], mode_sizes[core_id] * r});
    {
      auto temp = ::boba::backsolve(transfer_matrices[core_id], u);
      u = temp;
    }

    v.reshape({r * mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
    u.reshape({tt_ranks[core_id] * mode_sizes[core_id], tt_ranks[core_id + 1]});

    {
      auto temp = ::boba::right_backsolve(transfer_matrices[core_id + 2], v);
      v = temp;
    }

    v.reshape({r, mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});

    if (sweep_right)
    {
      ::boba::QR<python_space, data_t> qr;
      qr(u);
      u = qr.Q;
      auto triangular_factor = qr.R;
      auto selected_indices = native_cross.select_indices(u);

      right_interface = u.extract_rows(selected_indices);

      {
        auto temp = ::boba::right_backsolve(right_interface, u);
        u = temp;
      }

      approximated_tt.core(core_id).resize(
        {tt_ranks[core_id], mode_sizes[core_id], tt_ranks[core_id + 1]});
      approximated_tt.core(core_id).reshape(u);

      {
        auto temp = right_interface * triangular_factor;
        right_interface = temp;
      }
      v = right_interface * v;

      approximated_tt.core(core_id + 1).resize(
        {tt_ranks[core_id + 1], mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
      approximated_tt.core(core_id + 1).reshape(v);

      matrix_t<data_t> updated_left_core(
        {tt_ranks[core_id], mode_sizes[core_id] * tt_ranks[core_id + 1]});
      updated_left_core.reshape(u);
      auto updated_left_core_with_transfer = transfer_matrices[core_id] * updated_left_core;

      updated_left_core_with_transfer.reshape(
        {tt_ranks[core_id] * mode_sizes[core_id], tt_ranks[core_id + 1]});

      qr(updated_left_core_with_transfer);
      triangular_factor = qr.R;
      transfer_matrices[core_id + 1] = triangular_factor;

      auto previous_indices = interpolation_indices[core_id];
      index_matrix_t updated_indices({tt_ranks[core_id + 1], core_id + 1});
      updated_indices.fill_with_zeros();
      for (std::size_t selected_row = 0; selected_row < tt_ranks[core_id + 1]; selected_row++)
      {
        auto flattened_index = selected_indices({selected_row});
        ::boba::Multiindexer<2> unfolded_core_indices({tt_ranks[core_id], mode_sizes[core_id]});
        auto [left_rank_index, mode_index] = unfolded_core_indices.multiindex(flattened_index);

        auto previous_indices_view = previous_indices.view();
        auto updated_indices_view = updated_indices.view();

        ::boba::loop<python_space, 1>(updated_indices.cols(),
                                      [=] __boba_host_device__(index_t k)
        {
          index_t value = 0;

          if (core_id == previous_indices_view.sizes(1))
          {
            value = mode_index;
          }
          else
          {
            value = previous_indices_view({left_rank_index, k});
          }
          updated_indices_view({selected_row, k}) = value;
        });
      }

      interpolation_indices[core_id + 1] = updated_indices;
      if (core_id == dimension - 2)
      {
        sweep_right = not(sweep_right);
      }
      else
      {
        core_id += 1;
      }
    }
    else
    {
      // Reverse direction
      v.transpose_in_place();
      ::boba::QR<python_space, data_t> qr;
      qr(v);
      v = qr.Q;
      auto triangular_factor = qr.R;
      auto selected_indices = native_cross.select_indices(v);

      right_interface = v.extract_rows(selected_indices);

      {
        auto temp = ::boba::right_backsolve(right_interface, v);
        v = temp;
      }

      ::boba::Tensor<3, python_space, data_t> reshaped_right_core(
        {mode_sizes[core_id + 1], tt_ranks[core_id + 2], tt_ranks[core_id + 1]});
      reshaped_right_core.reshape(v);

      approximated_tt.core(core_id + 1) = reshaped_right_core;
      ::boba::permute({"index", "rank_right", "rank_left"},
                      approximated_tt.core(core_id + 1),
                      {"rank_left", "index", "rank_right"});

      {
        auto temp = right_interface * triangular_factor;
        right_interface = temp;
        right_interface.transpose_in_place();
        temp = u * right_interface;
        u = temp;
      }
      approximated_tt.core(core_id).resize(
        {tt_ranks[core_id], mode_sizes[core_id], tt_ranks[core_id + 1]});
      approximated_tt.core(core_id).reshape(u);

      v.transpose_in_place();
      v.reshape({tt_ranks[core_id + 1] * mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
      {
        auto temp = v * transfer_matrices[core_id + 2];
        v = temp;
      }

      v.reshape({tt_ranks[core_id + 1], mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});
      v.transpose_in_place();

      qr(v);
      triangular_factor = qr.R;

      transfer_matrices[core_id + 1] = triangular_factor;
      auto previous_indices = interpolation_indices[core_id + 2];

      // Native allocates Matrix<index_t>({dimension - core_id - 1, ...}).
      index_matrix_t updated_indices({dimension - core_id - 1, tt_ranks[core_id + 1]});
      updated_indices.fill_with_zeros();

      for (std::size_t selected_row = 0; selected_row < tt_ranks[core_id + 1]; selected_row++)
      {
        auto flattened_index = selected_indices({selected_row});
        ::boba::Multiindexer<2> unfolded_core_indices(
          {mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
        auto [mode_index, right_rank_index] = unfolded_core_indices.multiindex(flattened_index);

        auto previous_indices_view = previous_indices.view();
        auto updated_indices_view = updated_indices.view();

        ::boba::loop<python_space, 1>(updated_indices.rows(),
                                      [=] __boba_host_device__(index_t i)
        {
          index_t value = 0;
          if (i == 0)
          {
            value = mode_index;
          }
          else
          {
            value = previous_indices_view({i - 1, right_rank_index});
          }
          updated_indices_view({i, selected_row}) = value;
        });
      }

      interpolation_indices[core_id + 1] = updated_indices;

      if (core_id == 0)
      {
        sweep_right = not(sweep_right);
        sweep_count = sweep_count + 1;
        // Native compares a double accumulator against real_data_t; the cast is the
        // same promotion the native code performs implicitly, made explicit for float.
        if (max_relative_error < static_cast<double>(native_cross.convergence_tolerance))
        {
          not_converged = false;
        }
        else
        {
          max_relative_error = 0.0;
        }
      }
      else
      {
        core_id--;
      }
    }
  }

  return approximated_tt;
}

template RuntimeTensorTrain<float> runtime_dmrg_cross<float>(
  RuntimeTensorTrain<float> const&,
  Evaluator<float>&,
  CrossOptions const&);

template RuntimeTensorTrain<double> runtime_dmrg_cross<double>(
  RuntimeTensorTrain<double> const&,
  Evaluator<double>&,
  CrossOptions const&);

} // namespace boba_python
