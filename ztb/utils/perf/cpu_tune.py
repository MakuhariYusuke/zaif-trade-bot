import os
import psutil
import logging
import torch
from typing import Dict, List, Optional, Any

def auto_config_threads(num_processes: int, pin_to_cores: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    CPU最適化設定を自動決定
    物理コア数・並列プロセス数・コア割当を考慮して各ライブラリのスレッド数を決定

    Args:
        num_processes: 並列プロセス数
        pin_to_cores: 割当コアリスト（指定なしなら自動）

    Returns:
        決定した設定の辞書
    """
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical

    # 環境変数優先
    default_procs = num_processes if num_processes is not None else 1
    pin_cores = pin_to_cores or []
    if not pin_cores:
        pin_cores = list(range(physical))  # デフォルト全物理コア
    else:
        # pin_cores の内容が物理コアに対応しているか確認し、範囲外を除外
        pin_cores = [core for core in pin_cores if 0 <= core < physical]
    assigned_cores = min(len(pin_cores), physical)
    # プロセス数が割当コア数より多い場合は1, それ以外は割当コア数/プロセス数
    if default_procs > assigned_cores:
        threads_per_proc = 1
    else:
        threads_per_proc = max(1, assigned_cores // default_procs)

    # 各ライブラリのスレッド数設定
    config = {
        'physical_cores': physical,
        'logical_cores': logical,
        'num_processes': default_procs,
        'assigned_cores': assigned_cores,
        'pin_to_cores': pin_cores,
        'threads_per_proc': threads_per_proc,
        'OMP_NUM_THREADS': threads_per_proc,
        'MKL_NUM_THREADS': threads_per_proc,
        'OPENBLAS_NUM_THREADS': threads_per_proc,
        'NUMEXPR_NUM_THREADS': threads_per_proc,
        'MKL_DYNAMIC': 'FALSE',
        'torch_threads': threads_per_proc
    }

    return config

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
    try:
        torch.set_num_threads(per)
    except Exception as e:
        logging.warning(f"Failed to set torch num threads: {e}")
    try:
        torch.set_num_threads(per)
    except Exception:
        pass

    logging.info(
        f"[CPU] physical={physical} logical={logical} procs={procs} threads_per_proc={per} "
        f"torch={getattr(torch, 'get_num_threads', lambda: '?')()} "
        f"OMP={os.environ['OMP_NUM_THREADS']} MKL={os.environ['MKL_NUM_THREADS']} "
        f"OPENBLAS={os.environ['OPENBLAS_NUM_THREADS']}"
    )