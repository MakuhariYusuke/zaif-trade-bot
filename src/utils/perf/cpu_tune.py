import os
import psutil
import logging
import torch

def apply_cpu_tuning():
    """
    CPU最適化設定を自動適用
    物理コア数・並列プロセス数に応じてtorch/OMP/MKL/OPENBLAS/NUMEXPRのスレッド数を設定
    """
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    procs = int(os.environ.get("PARALLEL_PROCESSES", "1"))
    per = max(1, physical // max(1, procs))

    os.environ.setdefault("OMP_NUM_THREADS", str(per))
    os.environ.setdefault("MKL_NUM_THREADS", str(per))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(per))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(per))
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")

    try:
        torch.set_num_threads(per)
    except Exception:
        pass

    logging.info(f"[CPU] physical={physical} logical={logical} procs={procs} threads_per_proc={per} "
                 f"torch={getattr(torch, 'get_num_threads', lambda: '?')()} "
                 f"OMP={os.environ['OMP_NUM_THREADS']} MKL={os.environ['MKL_NUM_THREADS']} "
                 f"OPENBLAS={os.environ['OPENBLAS_NUM_THREADS']}")