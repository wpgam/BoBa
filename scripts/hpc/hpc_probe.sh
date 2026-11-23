#!/usr/bin/env bash
# Collects the cluster facts needed to tune pyboba for HPC. Read-only; safe on login nodes.
# Usage: bash hpc_probe.sh [python]   (writes hpc_probe_<host>.txt and prints it)

PY=${1:-python3}
OUT="hpc_probe_$(hostname -s)$( [ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}" ] && echo _job ).txt"
have() { command -v "$1" >/dev/null 2>&1; }
sec() { echo; echo "=== $* ==="; }

{
sec "host"
hostname; uname -sr; date -u +%FT%TZ
[ -f /etc/os-release ] && grep -E '^PRETTY_NAME' /etc/os-release

sec "scheduler"
if have sinfo; then
  sinfo --version
  sinfo -s 2>/dev/null | head -20
  echo "-- partitions (name, timelimit, nodes, cpus/node, mem/node MB, gres)"
  sinfo -h -o "%P %l %D %c %m %G" 2>/dev/null | sort -u | head -20
  scontrol show config 2>/dev/null | grep -iE 'SelectType|TaskPlugin|MpiDefault|ProctrackType'
elif have qstat && qstat --version >/dev/null 2>&1; then
  qstat --version 2>&1 | head -2; qstat -Q 2>/dev/null | head -20
elif have bqueues; then
  lsid 2>/dev/null | head -3; bqueues 2>/dev/null | head -20
else
  echo "no Slurm/PBS/LSF client found"
fi
env | grep -E '^(SLURM_(JOB_ID|CPUS_PER_TASK|NTASKS|NNODES|JOB_CPUS_PER_NODE|CPUS_ON_NODE|TASKS_PER_NODE)|PBS_(JOBID|NUM_PPN|NP)|LSB_(JOBID|DJOB_NUMPROC)|OMP_|MKL_NUM|OPENBLAS_NUM)=' | sort

sec "cpu"
lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|NUMA node|L3|Flags' | sed 's/\(Flags:\).*\(avx512f\).*/\1 ... \2 .../'
lscpu 2>/dev/null | grep -q avx512f || lscpu 2>/dev/null | grep -oE 'avx2' | head -1
echo "affinity of this shell: $(taskset -pc $$ 2>/dev/null | awk -F: '{print $2}')"
have nproc && echo "nproc: $(nproc)"

sec "numa"
have numactl && numactl -H 2>/dev/null | grep -E 'available|node [0-9]+ size|node distances' -A0 | head -12

sec "memory"
free -g 2>/dev/null | head -2

sec "gpu"
if have nvidia-smi; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -8
  nvidia-smi topo -m 2>/dev/null | head -10
elif have rocm-smi; then
  rocm-smi --showproductname 2>&1 | grep -E 'GPU|Card' | head -8
else
  echo "no nvidia-smi/rocm-smi"
fi

sec "modules"
if have module || type module >/dev/null 2>&1; then
  module -t avail 2>&1 | grep -iE 'python|mpi|gcc|intel|oneapi|cuda|rocm|blas|mkl|lapack|cmake|llvm|clang|eigen' | sort -u | head -60
  echo "-- loaded"; module -t list 2>&1 | head -20
else
  echo "no module command"
fi

sec "compilers / build tools"
for c in gcc g++ clang++ icpx mpicxx cmake; do have $c && echo "$c: $($c --version 2>&1 | head -1)"; done

sec "mpi"
for m in mpirun mpiexec srun; do have $m && echo "$m: $(command -v $m)"; done
have mpirun && mpirun --version 2>&1 | head -2
have ompi_info && ompi_info 2>/dev/null | grep -E 'MCA (btl|pml|mtl):' | awk '{print $2,$3}' | sort -u | tr '\n' ' '; echo
have fi_info && echo "libfabric providers: $(fi_info -l 2>/dev/null | grep -oE '^[a-z0-9_;]+' | sort -u | tr '\n' ' ')"
have ibv_devinfo && ibv_devinfo 2>/dev/null | grep -E 'hca_id|link_layer|active_mtu' | head -6
ls /sys/class/infiniband 2>/dev/null | head -4
ls -d /opt/cray 2>/dev/null && echo "Cray PE present"

sec "python ($PY)"
if have "$PY"; then
  "$PY" - <<'EOF'
import importlib, sys, platform
print("python", sys.version.split()[0], platform.machine(), sys.executable)
for m in ["numpy", "mpi4py", "numba", "dask", "joblib", "threadpoolctl", "pybind11", "scikit_build_core", "pytest", "cupy"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:18s} {getattr(mod, '__version__', 'present')}")
    except Exception as e:
        print(f"{m:18s} MISSING ({type(e).__name__})")
try:
    from mpi4py import MPI
    print("mpi4py MPI:", MPI.Get_library_version().splitlines()[0])
except Exception:
    pass
try:
    from threadpoolctl import threadpool_info
    for p in threadpool_info():
        print("blas:", p.get("internal_api"), p.get("version"), "threads", p.get("num_threads"))
except Exception:
    import io, contextlib, numpy as np
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    lines = [l for l in buf.getvalue().splitlines() if any(k in l.lower() for k in ("name", "openblas", "mkl", "blis", "lib"))]
    print("\n".join(lines[:12]))
EOF
else
  echo "$PY not found"
fi

sec "filesystems"
echo "HOME=$HOME"; for v in SCRATCH WORK PROJECT TMPDIR LOCAL_SCRATCH; do [ -n "${!v:-}" ] && echo "$v=${!v}"; done
df -hT "$HOME" ${SCRATCH:+"$SCRATCH"} ${WORK:+"$WORK"} /tmp 2>/dev/null | awk 'NR==1 || !seen[$0]++'
mount 2>/dev/null | grep -oE 'type (lustre|gpfs|nfs4?|beegfs|wekafs|xfs)' | sort | uniq -c
} 2>&1 | tee "$OUT"

echo; echo "Wrote $OUT"
