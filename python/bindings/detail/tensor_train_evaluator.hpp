// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "python_evaluator.hpp"
#include "runtime_tensor_train.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

/**
 * \file
 * \brief Entry source for cross-approximating a pointwise function of existing trains.
 *
 * The cross sweep asks for tensor entries at the multi-indices it has selected. When the
 * tensor being approximated is `f(A, B, ...)` applied entrywise to trains that already
 * exist, each requested entry is `f(A[index], B[index], ...)`, and the two halves of
 * that expression have very different costs.
 *
 * Looking up `A[index]` is a contraction of the selected core slices: pure C++, no
 * Python, and the dominant cost when the ranks are large. Applying `f` is a scalar
 * operation the user wrote in Python.
 *
 * So the two are separated. Every entry lookup for a whole batch happens first, with the
 * GIL still released, and `f` is then called exactly once for the entire batch with one
 * array of values per input train. A batch holds up to 2^20 entries, which turns what
 * would otherwise be a million GIL acquisitions and a million Python calls into one.
 */

namespace boba_python
{

/**
 * \brief Evaluates `f(A[index], B[index], ...)` in batches.
 *
 * \tparam data_t Scalar type shared by the input trains and the result.
 *
 * The callback receives one one-dimensional NumPy array per input train, all of the same
 * length, and returns one value per entry. The arrays are copies, so the callback cannot
 * disturb the sweep by writing to them.
 */
template <typename data_t>
class TensorTrainFunctionEvaluator final : public PythonEvaluator<data_t>
{
public:
  TensorTrainFunctionEvaluator(
    py::object function,
    std::vector<RuntimeTensorTrain<data_t> const*> inputs)
      : m_function(std::move(function)),
        m_inputs(std::move(inputs))
  {
  }

  void evaluate(
    std::size_t const* indices,
    std::size_t batch_count,
    std::size_t ndim,
    data_t* values) override
  {
    const std::size_t input_count = m_inputs.size();

    // Phase one: no Python whatsoever. This is the expensive half, and it runs with the
    // GIL still released by the sweep.
    m_scratch.resize(input_count * batch_count);
    for (std::size_t k = 0; k < input_count; k++)
    {
      data_t* column = m_scratch.data() + k * batch_count;
      for (std::size_t b = 0; b < batch_count; b++)
      {
        column[b] = m_inputs[k]->entry(
          std::span<const std::size_t>(indices + b * ndim, ndim));
      }
    }

    // Phase two: one GIL acquisition and one call for the whole batch.
    py::gil_scoped_acquire gil;

    try
    {
      py::tuple arguments(input_count);
      for (std::size_t k = 0; k < input_count; k++)
      {
        // Constructed without a base handle, so pybind11 copies: the callback holds its
        // own arrays and m_scratch stays ours to reuse on the next batch.
        arguments[k] = py::array_t<data_t>(
          static_cast<py::ssize_t>(batch_count), m_scratch.data() + k * batch_count);
      }

      py::object result = m_function(*arguments);
      this->store_batch_result(result, batch_count, values, "cross_function callback");
    }
    catch (py::error_already_set& error)
    {
      this->abort_with(error);
    }
  }

  /**
   * \brief Entries per callback invocation.
   *
   * Matches the vectorized evaluator: large enough that a whole score-tensor block
   * normally arrives in one call, small enough to keep the staging buffers bounded as
   * interface ranks grow.
   */
  [[nodiscard]] std::size_t preferred_batch_size() const override
  {
    return 1u << 20;
  }

  /**
   * \brief Fills a two-site score block with matrix products instead of per-entry
   *        contractions.
   *
   * For every input train, the left chain G_0..G_{k-1} is contracted once per left tuple
   * (an r_left x r_k matrix L) and the right chain G_{k+2}..G_{D-1} once per right tuple
   * (an r_{k+2} x r_right matrix R). The block is then
   *   T(s1, i1, i2, s2) = L(s1, :) G_k(:, i1, :) G_{k+1}(:, i2, :) R(:, s2),
   * formed with two GEMMs. Entry cost drops from O(D r^2) to O(r^2 / n) amortised, and
   * the Python function is still called once on whole arrays.
   */
  bool evaluate_block(
    std::size_t const* left_indices, std::size_t r_left,
    std::size_t k, std::size_t n_k, std::size_t n_k1,
    std::size_t const* right_indices, std::size_t r_right,
    std::size_t ndim, data_t* values) override
  {
    using Mat = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
    const std::size_t input_count = m_inputs.size();
    const std::size_t block_size = r_left * n_k * n_k1 * r_right;
    const std::size_t nr = ndim - k - 2;

    m_scratch.resize(input_count * block_size);
    for (std::size_t t = 0; t < input_count; t++)
    {
      auto const& cores = m_inputs[t]->cores();
      // core d as (r_a, n, r_b), column-major: element (a, i, b) at a + r_a * (i + n * b)
      auto slice = [&](std::size_t d, std::size_t i) {
        auto const& c = cores[d];
        const std::size_t ra = c.sizes(0), n = c.sizes(1), rb = c.sizes(2);
        Mat M(ra, rb);
        data_t const* base = c.data();
        for (std::size_t b = 0; b < rb; b++)
          for (std::size_t a = 0; a < ra; a++)
            M(a, b) = base[a + ra * (i + n * b)];
        return M;
      };

      // left environments: L(s1, :) = prod_{m<k} G_m(:, l_m, :), shape r_left x r_k
      const std::size_t r_k = cores[k].sizes(0);
      Mat L = Mat::Ones(r_left, 1);
      for (std::size_t m = 0; m < k; m++)
      {
        auto const& c = cores[m];
        const std::size_t ra = c.sizes(0), n = c.sizes(1), rb = c.sizes(2);
        data_t const* base = c.data();
        Mat next(r_left, rb);
        for (std::size_t s1 = 0; s1 < r_left; s1++)
        {
          const std::size_t i = left_indices[s1 + r_left * m];
          for (std::size_t b = 0; b < rb; b++)
          {
            data_t acc = data_t(0);
            for (std::size_t a = 0; a < ra; a++)
              acc += L(s1, a) * base[a + ra * (i + n * b)];
            next(s1, b) = acc;
          }
        }
        L = std::move(next);
      }
      (void)r_k;

      // right environments: R(:, s2) = prod_{m>k+1} G_m(:, j_m, :) 1, shape r_{k+2} x r_right
      Mat R = Mat::Ones(1, r_right);
      for (std::size_t mm = nr; mm-- > 0;)
      {
        const std::size_t m = k + 2 + mm;
        auto const& c = cores[m];
        const std::size_t ra = c.sizes(0), n = c.sizes(1), rb = c.sizes(2);
        data_t const* base = c.data();
        Mat next(ra, r_right);
        for (std::size_t s2 = 0; s2 < r_right; s2++)
        {
          const std::size_t j = right_indices[mm + nr * s2];
          for (std::size_t a = 0; a < ra; a++)
          {
            data_t acc = data_t(0);
            for (std::size_t b = 0; b < rb; b++)
              acc += base[a + ra * (j + n * b)] * R(b, s2);
            next(a, s2) = acc;
          }
        }
        R = std::move(next);
      }

      // free pair: A(s1, i1, :) = L G_k(:, i1, :)  -> (r_left * n_k) x r_mid, s1 fastest
      auto const& ck = cores[k];
      const std::size_t rka = ck.sizes(0), rmid = ck.sizes(2);
      Mat Gk(rka, n_k * rmid); // column (i1 + n_k * b) holds G_k(:, i1, b)
      {
        data_t const* base = ck.data();
        for (std::size_t b = 0; b < rmid; b++)
          for (std::size_t i1 = 0; i1 < n_k; i1++)
            for (std::size_t a = 0; a < rka; a++)
              Gk(a, i1 + n_k * b) = base[a + rka * (i1 + n_k * b)];
      }
      Mat LG = L * Gk; // r_left x (n_k * r_mid)
      // B(:, i2, s2) = G_{k+1}(:, i2, :) R -> r_mid x (n_k1 * r_right)
      auto const& ck1 = cores[k + 1];
      const std::size_t rc = ck1.sizes(2);
      Mat Gk1(rmid * n_k1, rc); // row (b + r_mid * i2) holds G_{k+1}(b, i2, :)
      {
        data_t const* base = ck1.data();
        for (std::size_t c = 0; c < rc; c++)
          for (std::size_t i2 = 0; i2 < n_k1; i2++)
            for (std::size_t b = 0; b < rmid; b++)
              Gk1(b + rmid * i2, c) = base[b + rmid * (i2 + n_k1 * c)];
      }
      Mat GR = Gk1 * R; // (r_mid * n_k1) x r_right
      // T(s1, i1, i2, s2) = sum_b LG(s1, i1 + n_k b) GR(b + r_mid i2, s2)
      // Rearrange LG -> P (r_left * n_k) x r_mid and GR -> Q r_mid x (n_k1 * r_right)
      Mat P(r_left * n_k, rmid);
      for (std::size_t b = 0; b < rmid; b++)
        for (std::size_t i1 = 0; i1 < n_k; i1++)
          for (std::size_t s1 = 0; s1 < r_left; s1++)
            P(s1 + r_left * i1, b) = LG(s1, i1 + n_k * b);
      Mat Q(rmid, n_k1 * r_right);
      for (std::size_t s2 = 0; s2 < r_right; s2++)
        for (std::size_t i2 = 0; i2 < n_k1; i2++)
          for (std::size_t b = 0; b < rmid; b++)
            Q(b, i2 + n_k1 * s2) = GR(b + rmid * i2, s2);
      Mat T = P * Q; // (r_left * n_k) x (n_k1 * r_right): column-major == (s1, i1, i2, s2)
      std::copy(T.data(), T.data() + block_size, m_scratch.data() + t * block_size);
    }

    // one Python call for the whole block
    py::gil_scoped_acquire gil;
    try
    {
      py::tuple arguments(input_count);
      for (std::size_t t = 0; t < input_count; t++)
      {
        arguments[t] = py::array_t<data_t>(
          static_cast<py::ssize_t>(block_size), m_scratch.data() + t * block_size);
      }
      py::object result = m_function(*arguments);
      this->store_batch_result(result, block_size, values, "cross_function callback");
    }
    catch (py::error_already_set& error)
    {
      this->abort_with(error);
    }
    return true;
  }

private:
  py::object m_function;
  std::vector<RuntimeTensorTrain<data_t> const*> m_inputs;
  /// Reused between batches so a long sweep does not reallocate on every call.
  std::vector<data_t> m_scratch;
};

} // namespace boba_python
