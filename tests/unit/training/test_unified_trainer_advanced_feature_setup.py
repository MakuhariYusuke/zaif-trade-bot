from __future__ import annotations

from types import SimpleNamespace

import torch

from ztb.training.unified_trainer.advanced_feature_setup import (
    build_continual_learning_config,
    collect_meta_learning_history,
    extract_algorithm_model,
    record_training_stat,
    resolve_federated_stats,
    resolve_model_input_dim,
    resolve_model_output_dim,
)


def test_extract_algorithm_model_returns_model_when_present() -> None:
    model = object()
    trainer = SimpleNamespace(model=model)
    assert extract_algorithm_model(trainer) is model


def test_extract_algorithm_model_returns_none_when_missing() -> None:
    assert extract_algorithm_model(None) is None
    assert extract_algorithm_model(SimpleNamespace()) is None
    assert extract_algorithm_model(SimpleNamespace(model=None)) is None


def test_build_continual_learning_config_uses_defaults_and_overrides() -> None:
    config = {
        "continual_method": "mas",
        "continual_ewc_lambda": 0.25,
        "continual_buffer_size": 2048,
        "continual_max_tasks": 7,
    }

    continual = build_continual_learning_config(config)

    assert continual.method == "mas"
    assert continual.ewc_lambda == 0.25
    assert continual.rehearsal_buffer_size == 2048
    assert continual.max_tasks_in_memory == 7
    assert continual.enable_memory_tracking is True


def test_resolve_model_dims_use_model_parameters_and_defaults() -> None:
    trainer = SimpleNamespace(model=torch.nn.Linear(6, 3))

    assert resolve_model_input_dim(trainer) == 6
    assert resolve_model_output_dim(trainer) == 3
    assert resolve_model_input_dim(SimpleNamespace(model=None), default=11) == 11
    assert resolve_model_output_dim(SimpleNamespace(), default=7) == 7


def test_collect_meta_learning_history_prefers_nested_buffer_and_falls_back() -> None:
    nested_meta = SimpleNamespace(task_buffer=[1])
    meta = SimpleNamespace(
        meta_learner=nested_meta,
        train_on_markets=lambda num_epochs=50: {"epochs": num_epochs},
    )
    assert collect_meta_learning_history(meta, num_epochs=3) == {"epochs": 3}

    direct_meta = SimpleNamespace(
        task_buffer=[1],
        train_on_markets=lambda num_epochs=50: {"epochs": num_epochs},
    )
    assert collect_meta_learning_history(direct_meta, num_epochs=2) == {"epochs": 2}
    assert collect_meta_learning_history(SimpleNamespace(task_buffer=[])) is None


def test_resolve_federated_stats_and_record_training_stat() -> None:
    training_stats: dict[str, object] = {}
    learner = SimpleNamespace(get_federated_stats=lambda: {"rounds": 2})

    assert resolve_federated_stats(learner) == {"rounds": 2}
    assert resolve_federated_stats(SimpleNamespace()) == {}

    record_training_stat(training_stats, "federated_learning", {"rounds": 2})

    assert training_stats == {"federated_learning": {"rounds": 2}}
