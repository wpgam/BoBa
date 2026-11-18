# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Thread-level parallelism over independent pyboba calls.

pyboba releases the GIL inside cross sweeps, rounding and train arithmetic, so
independent calls (one per block, component or parameter value) overlap in threads.
Python callbacks still run under the GIL, so speedup is bounded by the share of time
spent in them.

Worker count, first match wins:

1. the ``threads`` argument;
2. ``PYBOBA_NUM_THREADS``;
3. ``SLURM_CPUS_PER_TASK`` (so ``srun -c N`` sets it per MPI rank);
4. the CPUs this process may run on.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_pool: Optional[ThreadPoolExecutor] = None
_pool_size = 0


def num_threads() -> int:
    """Default worker count for :func:`map` (see the module docstring)."""
    for var in ("PYBOBA_NUM_THREADS", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(var, "").strip()
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                raise ValueError(f"{var}={value!r} is not an integer") from None
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS
        return max(1, os.cpu_count() or 1)


def _get_pool(workers: int) -> ThreadPoolExecutor:
    global _pool, _pool_size
    if _pool is None or _pool_size != workers:
        if _pool is not None:
            _pool.shutdown(wait=True)
        _pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pyboba")
        _pool_size = workers
    return _pool


def map(function: Callable[[T], R], items: Iterable[T], threads: Optional[int] = None) -> List[R]:
    """``[function(x) for x in items]``, run concurrently in a persistent thread pool.

    Results keep the order of ``items``. The first exception raised by any call is
    re-raised here, after all submitted calls have finished. With one worker (or one
    item) everything runs serially in the calling thread.

    For reproducible results, give every call its own ``seed=`` (see :func:`pyboba.cross`):
    which thread runs which item is not deterministic.
    """
    items = list(items)
    workers = min(num_threads() if threads is None else max(1, int(threads)), len(items))
    if workers <= 1:
        return [function(x) for x in items]
    futures = [_get_pool(workers).submit(function, x) for x in items]
    errors = [f.exception() for f in futures]
    for error in errors:
        if error is not None:
            raise error
    return [f.result() for f in futures]
