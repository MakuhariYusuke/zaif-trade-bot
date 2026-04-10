"""Compatibility shim for sidecar scheduler helpers.

実体は `ztb.training.sidecar.scheduler_common` へ移した。
v460 script 層の import 互換だけをここで維持する。
"""

from ztb.training.sidecar.scheduler_common import (
    BaseRetrainResult,
    DataFileRetrainTrigger,
    append_history_best_effort,
    append_history_jsonl,
    atomic_replace_with_tmp,
    best_effort_training_cleanup,
    install_shutdown_signal_handlers,
    push_neutral_signal_best_effort,
    record_trigger_result_best_effort,
    run_with_timeout,
)

__all__ = [
    "BaseRetrainResult",
    "DataFileRetrainTrigger",
    "append_history_best_effort",
    "append_history_jsonl",
    "atomic_replace_with_tmp",
    "best_effort_training_cleanup",
    "install_shutdown_signal_handlers",
    "push_neutral_signal_best_effort",
    "record_trigger_result_best_effort",
    "run_with_timeout",
]
