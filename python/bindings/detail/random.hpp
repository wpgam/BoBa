// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/abstractions/random.hpp"

namespace boba_python
{

/**
 * \brief Per-thread generator behind every pyboba random fill.
 *
 * `boba::random::default_context()` is process-wide and not thread-safe, so concurrent
 * cross calls from a thread pool each draw from their own context instead. Seeded from
 * std::random_device on first use; reseeding affects only the calling thread.
 */
inline ::boba::random::RandomContext& thread_random_context()
{
  thread_local ::boba::random::RandomContext context;
  return context;
}

} // namespace boba_python
