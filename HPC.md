# pyboba on HPC systems

How to build and run pyboba on a cluster, and how its parallelism is organized.
Examples use Slurm; the same structure applies to other schedulers.

## Parallel model

| Layer | Mechanism | Controlled by |
| --- | --- | --- |
| Inside one call | Serial C++ (ranks are small; threaded BLAS does not pay off) | — |
| Across independent calls in one process | Threads: `pyboba.parallel.map`. Native work releases the GIL | `threads=`, `PYBOBA_NUM_THREADS`, `SLURM_CPUS_PER_TASK` |
| Across independent runs | One process per scheduler task (`srun -n R -c C`) | job script |

There is no MPI layer. Independent runs (seeds, parameter sweeps) share no data, so
scheduler tasks distribute them without communication. Inside one run, the parallel units
of work (e.g. the blocks of one time-integration stage) typically number in the tens and
synchronize every stage, so one node's threads are enough.
GPU offload is not available from Python yet.

Expected speedup from threads is bounded by:

- the number of independent calls per synchronization point;
- their load balance (calls of very different cost);
- the share of time spent in Python callbacks, which hold the GIL. Check it with
  `cProfile`: `tottime` of your callbacks divided by the total.

## Install

Compute nodes on many systems have no internet, so build on a login node:

```bash
CMAKE_ARGS="-DEIGEN_DIR=$PWD/install/eigen_cpu" python -m pip install .
python -c "import pyboba, pyboba.parallel as p; print(p.num_threads())"
```

If the bundled Eigen fails to compile an AVX512-FP16 code path (seen with the Cray `CC`
wrapper on Sapphire Rapids), restrict the target ISA, e.g. add
`-DCMAKE_CXX_FLAGS=-march=x86-64-v3` (AVX2 + FMA) to `CMAKE_ARGS`.

## Configuration

| Variable | Effect |
| --- | --- |
| `PYBOBA_NUM_THREADS` | Default worker count of `pyboba.parallel.map` |
| `SLURM_CPUS_PER_TASK` | Used when `PYBOBA_NUM_THREADS` is unset |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` | Set to 1: NumPy's BLAS must not add threads on top |

On nodes with hyperthreading, `-c` counts hardware threads. To give a run `T`
physical cores: `srun -n1 -c $((2*T)) --cpu-bind=cores` with `PYBOBA_NUM_THREADS=T`.

## Reproducibility

- Each thread has its own random generator (cross starting guesses, rank enrichment).
- `pyboba.seed(v)` seeds the calling thread's generator.
- `cross(..., seed=s)` / `cross_function(..., seed=s)` make one call depend only on `s`.
- `pyboba.last_cross_info()` reports the most recent call on the calling thread.

Seed each unit of work from its identity, not from its position in a shared stream, and
results are bit-identical for every thread count:

```python
import pyboba
from pyboba import parallel

def block_task(i):
    pyboba.seed(hash((base_seed, step, stage, i)) & 0xFFFFFFFF)
    return compute_block(i)          # any sequence of pyboba calls

results = parallel.map(block_task, range(n_blocks))   # threads from the environment
```

## Job scripts

- `scripts/hpc/hpc_probe.sh`: environment report (scheduler, CPU/NUMA, modules, MPI,
  Python packages, filesystems). Read-only; works under Slurm, PBS and LSF.
- `tests/python/benchmarks/bench_threads.py`: thread scaling of independent
  `cross_function` calls, useful for choosing `PYBOBA_NUM_THREADS` on a new node type.

Template for a threaded production run:

```bash
#!/bin/bash
#SBATCH --account=<account> --partition=<partition> --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=32          # 16 physical cores with 2 hyperthreads
#SBATCH --time=12:00:00 --signal=TERM@900 --output=logs/%x_%j.out
source <environment with pyboba installed>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYBOBA_NUM_THREADS=16
srun --cpu-bind=cores python my_run.py
```

For many independent runs, split a node into `R` tasks of `T` threads
(`--ntasks-per-node=R --cpus-per-task=2T`, `R * T` = physical cores). Measure a few
`R x T` splits and pick the highest runs per hour, unless one run's latency matters more.

## I/O

Give every run its own output directory (checkpoints, snapshots, logs), so runs never
write the same file. Keep run directories on the scratch filesystem, not in `$HOME`.

## Limitations

- Thread speedup per run is bounded by the work items per synchronization point and by
  Python callbacks holding the GIL.
- Not checked with ThreadSanitizer: concurrency safety rests on the per-thread generator
  and diagnostics, read-only sharing of input trains, and bit-identity tests.
