# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Timing of cross and cross_function. Prints CSV: case,ndim,n,options,time_s,max_rank,rel_error.

Runs against whichever pyboba is importable, so the same script compares two builds:

    PYTHONPATH=/path/to/old/site python tests/python/benchmarks/bench_cross.py
    python tests/python/benchmarks/bench_cross.py
"""

import argparse
import time

import numpy as np

import pyboba

HAS_NEW_OPTIONS = hasattr(pyboba, "last_cross_info")


def inverse_sum(indices):
    return 1.0 / (1.0 + indices.sum(axis=1).astype(np.float64))


def dense_inverse_sum(shape):
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def best_time(fn, repeats):
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def rel_error(tt, dense):
    return float(np.linalg.norm(tt.to_numpy() - dense) / np.linalg.norm(dense))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndim", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    option_sets = {"default": {}}
    if HAS_NEW_OPTIONS:
        option_sets["svd_auto+val"] = {"svd_tolerance": "auto", "val_size": 1000}

    print("case,ndim,n,options,time_s,max_rank,rel_error")
    for ndim in args.ndim:
        shape = (args.n,) * ndim
        dense = dense_inverse_sum(shape)
        base = pyboba.compress(1.0 + dense) if ndim <= 6 else None
        for label, extra in option_sets.items():
            kwargs = dict(tolerance=args.tolerance, initial_rank=2, **extra)

            if hasattr(pyboba, "seed"):
                pyboba.seed(0)
            t, tt = best_time(lambda: pyboba.cross(shape, inverse_sum, vectorized=True, **kwargs),
                              args.repeats)
            print(f"cross,{ndim},{args.n},{label},{t:.4f},{max(tt.ranks)},{rel_error(tt, dense):.2e}")

            if base is not None:
                # 1 / (1 + dense) built from a train of 1 + dense: exercises the block path.
                t, tt = best_time(lambda: pyboba.cross_function(lambda a: 1.0 / a, [base], **kwargs),
                                  args.repeats)
                err = rel_error(tt, 1.0 / (1.0 + dense))
                print(f"cross_function,{ndim},{args.n},{label},{t:.4f},{max(tt.ranks)},{err:.2e}")


if __name__ == "__main__":
    main()
