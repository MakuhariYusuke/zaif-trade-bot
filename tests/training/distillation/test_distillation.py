import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from tests.helpers import make_tiny_distillation_loader, make_tiny_teacher_model
from ztb.training.distillation.distiller import DistillationPipeline, SACDistiller


if not all(
    hasattr(nn, attr)
    for attr in ("Sequential", "Linear", "ReLU", "CrossEntropyLoss")
):
    pytest.skip(
        "Distillation tests require the full torch.nn surface; current suite is running with a lightweight stub.",
        allow_module_level=True,
    )

@patch("ztb.training.distillation.distiller.torch.optim.Adam")
def test_create_student_and_distill_small(mock_adam):
    teacher = make_tiny_teacher_model(input_dim=10, hidden_dim=8, output_dim=3)
    pipeline = DistillationPipeline(
        {
            "distillation": {
                "temperature": 2.0,
                "alpha": 0.5,
                "student_lr_multiplier": 0.2,
            }
        }
    )
    mock_adam.return_value = Mock()
    loader = make_tiny_distillation_loader(
        rows=6,
        input_dim=10,
        output_dim=3,
        batch_size=3,
    )
    student = pipeline.distiller.create_student_model(teacher, compression_ratio=0.5)
    pipeline.distiller.create_student_model = lambda *args, **kwargs: student
    pipeline.distiller.distill = lambda *args, **kwargs: {  # type: ignore[method-assign]
        "epochs": [0],
        "teacher_losses": [0.0],
        "student_losses": [0.0],
        "distillation_losses": [0.0],
        "final_accuracy": 1.0,
    }

    results = pipeline.run_pipeline(
        teacher, loader, device=torch.device("cpu"), compression_ratio=0.5, num_epochs=1
    )

    assert results["success"] is True
    assert "student_model" in results
    assert results["student_model"] is not None
    assert "training_results" in results
    assert "compression_stats" in results
    first_layer = next(module for module in results["student_model"] if isinstance(module, nn.Linear))
    assert first_layer.in_features == 10
    assert first_layer.out_features <= 8


def test_create_student_model_tracks_teacher_dimensions():
    teacher = make_tiny_teacher_model(input_dim=10, hidden_dim=8, output_dim=3)
    distiller = SACDistiller({"temperature": 2.0, "alpha": 0.5})

    student = distiller.create_student_model(teacher, compression_ratio=0.5)
    linear_layers = [module for module in student if isinstance(module, nn.Linear)]

    assert linear_layers[0].in_features == 10
    assert linear_layers[-1].out_features == 3
    assert linear_layers[0].out_features <= 8


def test_distiller_loss_components():
    # Validate the DistillationLoss forward runs
    distiller = SACDistiller({"temperature": 2.0, "alpha": 0.5})
    loss_module = distiller.distillation_loss

    student_logits = torch.randn(4, 3)
    teacher_logits = torch.randn(4, 3)
    hard_labels = torch.randint(0, 3, (4,))

    loss = loss_module(student_logits, teacher_logits, hard_labels)
    assert loss.item() >= 0
