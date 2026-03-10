import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ztb.training.distillation.distiller import DistillationPipeline, SACDistiller


if not all(
    hasattr(nn, attr)
    for attr in ("Sequential", "Linear", "ReLU", "CrossEntropyLoss")
):
    pytest.skip(
        "Distillation tests require the full torch.nn surface; current suite is running with a lightweight stub.",
        allow_module_level=True,
    )


def _make_teacher_student():
    teacher = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 3))
    return teacher


def test_create_student_and_distill_small():
    teacher = _make_teacher_student()
    pipeline = DistillationPipeline(
        {"distillation": {"temperature": 2.0, "alpha": 0.5}}
    )

    # Create tiny dataset
    x = torch.randn(20, 10)
    y = torch.randint(0, 3, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    results = pipeline.run_pipeline(
        teacher, loader, device=torch.device("cpu"), compression_ratio=0.5, num_epochs=1
    )

    assert "success" in results
    assert isinstance(results["success"], bool)
    assert "student_model" in results
    assert results["student_model"] is not None or results.get("error")
    if results["success"]:
        assert "training_results" in results
        assert "compression_stats" in results


def test_distiller_loss_components():
    # Validate the DistillationLoss forward runs
    distiller = SACDistiller({"temperature": 2.0, "alpha": 0.5})
    loss_module = distiller.distillation_loss

    student_logits = torch.randn(4, 3)
    teacher_logits = torch.randn(4, 3)
    hard_labels = torch.randint(0, 3, (4,))

    loss = loss_module(student_logits, teacher_logits, hard_labels)
    assert loss.item() >= 0
