// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * \file
 * \brief Error types and overflow-checked size arithmetic shared by the Python extension.
 */

namespace boba_python
{

/**
 * \brief Signals that a Python evaluator callback failed.
 *
 * The evaluator restores the Python error indicator while it still holds the GIL and
 * then throws this marker, which carries no Python state. That lets the exception
 * unwind through GIL-released numerical code safely; the binding layer converts it
 * back into the original Python exception once the GIL is held again.
 */
struct PythonCallbackError
{
};

/// Maps to Python `MemoryError`.
struct MemoryErrorException : std::runtime_error
{
  using std::runtime_error::runtime_error;
};

/**
 * \brief Multiplies extents while rejecting `size_t` overflow.
 *
 * Dense tensor sizes grow exponentially with dimension, so every place that turns a
 * shape into an allocation goes through here and raises a Python-visible error rather
 * than wrapping around.
 */
[[nodiscard]] inline std::size_t checked_product(
  std::vector<std::size_t> const& extents,
  std::string const& what)
{
  std::size_t total = 1;
  for (std::size_t extent : extents)
  {
    if (extent != 0 && total > std::numeric_limits<std::size_t>::max() / extent)
    {
      throw MemoryErrorException(what + ": size overflows std::size_t");
    }
    total *= extent;
  }
  return total;
}

} // namespace boba_python
