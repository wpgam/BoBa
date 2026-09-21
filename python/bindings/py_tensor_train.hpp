// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "detail/runtime_tensor_train.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

/**
 * \file
 * \brief The Python-facing tensor train and the pieces shared across binding units.
 */

namespace boba_python
{

namespace py = pybind11;

/// Scalar types exposed in round one.
enum class ScalarKind
{
  Float32,
  Float64
};

/**
 * \brief Tensor train as seen from Python.
 *
 * Dtype is erased with a two-alternative variant. Dimension is deliberately *not* part
 * of that variant: each alternative already stores its cores in a `std::vector`, so the
 * tensor dimension is ordinary runtime state and there is no per-dimension
 * instantiation anywhere in this extension.
 *
 * Every accessor that hands data to Python returns a copy; see `detail/numpy_interop.hpp`.
 */
class PyTensorTrain
{
public:
  using storage_type = std::variant<RuntimeTensorTrain<float>, RuntimeTensorTrain<double>>;

  explicit PyTensorTrain(storage_type storage)
      : m_storage(std::move(storage))
  {
  }

  [[nodiscard]] ScalarKind scalar_kind() const noexcept
  {
    return (m_storage.index() == 0) ? ScalarKind::Float32 : ScalarKind::Float64;
  }

  [[nodiscard]] py::object dtype() const;
  [[nodiscard]] std::size_t ndim() const;
  [[nodiscard]] py::tuple shape() const;
  [[nodiscard]] py::tuple ranks() const;
  [[nodiscard]] py::list cores() const;
  [[nodiscard]] py::array to_numpy() const;
  [[nodiscard]] py::object getitem(py::object const& key) const;
  [[nodiscard]] std::string repr() const;

  [[nodiscard]] storage_type const& storage() const noexcept
  {
    return m_storage;
  }

private:
  storage_type m_storage;
};

/// Resolves a NumPy-compatible dtype specifier to a supported scalar kind.
[[nodiscard]] ScalarKind scalar_kind_from_dtype(py::handle dtype_like);

/// The NumPy dtype object matching \p kind.
[[nodiscard]] py::object dtype_object(ScalarKind kind);

/**
 * \brief Validates a user-supplied shape and returns it as extents.
 *
 * Rejects empty shapes, non-integers, negative values and zero extents, and rejects
 * shapes whose dense element count would overflow `std::size_t`.
 */
[[nodiscard]] std::vector<std::size_t> parse_shape(py::handle shape_like);

void register_tensor_train(py::module_& module);
void register_compress(py::module_& module);
void register_cross(py::module_& module);

} // namespace boba_python
