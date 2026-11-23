# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Thread scaling of independent cross_function calls, shaped like a blockwise flux update.

Each task cross-approximates a Lax-Friedrichs-style flux of six input trains of shape
(cells, 2M, 2M), one task per block. Prints CSV: threads,tasks,time_s,speedup,efficiency
and checks every threaded result against the serial one.

    python tests/python/benchmarks/bench_threads.py --M 20 --tasks 24 --threads 1 2 4 8
"""

import argparse
import time

import numpy as np

import pyboba
from pyboba import parallel


def positive_train(shape, rank, rng):
    cores = []
    for d, n in enumerate(shape):
        left = 1 if d == 0 else rank
        right = 1 if d == len(shape) - 1 else rank
        cores.append(0.5 + rng.random((left, n, right)) / (left * right))
    return pyboba.from_cores(cores)


def flux(rho_l, mom_l, e_l, rho_r, mom_r, e_r):
    u_l, u_r = mom_l / rho_l, mom_r / rho_r
    p_l, p_r = 0.4 * (e_l - 0.5 * mom_l * u_l), 0.4 * (e_r - 0.5 * mom_r * u_r)
    speed = np.maximum(np.abs(u_l) + np.sqrt(np.abs(1.4 * p_l / rho_l)),
                       np.abs(u_r) + np.sqrt(np.abs(1.4 * p_r / rho_r)))
    return 0.5 * (mom_l * u_l + p_l + mom_r * u_r + p_r) - 0.5 * speed * (mom_r - mom_l)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=20, help="parameter cells per direction")
    parser.add_argument("--cells", type=int, default=12)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    shape = (args.cells, 2 * args.M, 2 * args.M)
    rng = np.random.default_rng(0)
    inputs = [[positive_train(shape, args.rank, rng) for _ in range(6)] for _ in range(args.tasks)]

    def task(i):
        return pyboba.cross_function(flux, inputs[i], tolerance=1e-3, initial_rank=2,
                                     n_sweeps=5, kick_rank=1, seed=i)

    reference = None
    base = None
    print("threads,tasks,time_s,speedup,efficiency")
    for threads in args.threads:
        best = float("inf")
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = parallel.map(task, range(args.tasks), threads=threads)
            best = min(best, time.perf_counter() - start)
        if reference is None:
            reference = [r.cores for r in result]
        for ref, r in zip(reference, result):
            assert all(np.array_equal(a, b) for a, b in zip(ref, r.cores)), "threaded != serial"
        base = base or best
        print(f"{threads},{args.tasks},{best:.4f},{base / best:.2f},{base / best / threads:.2f}")


if __name__ == "__main__":
    main()
