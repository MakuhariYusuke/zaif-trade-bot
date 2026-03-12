"""
Tests for Gradient Accumulation Utilities

Tests gradient accumulation functionality including:
- Basic accumulation
- Gradient clipping
- Mixed precision support
- Trainer integration
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from ztb.training.gradient_accumulation import (
    GradientAccumulationTrainer,
    GradientAccumulator,
    create_gradient_accumulator,
    effective_batch_size_info,
)


if not (
    hasattr(nn, "Linear")
    and hasattr(nn, "MSELoss")
    and hasattr(torch, "relu")
    and hasattr(torch, "optim")
    and hasattr(torch.optim, "SGD")
):
    pytest.skip(
        "Gradient accumulation tests require full torch autograd/optim support; current suite is running with a lightweight stub.",
        allow_module_level=True,
    )


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.output(torch.relu(self.linear(x)))


class TestGradientAccumulator:
    """Test GradientAccumulator class."""

    def test_basic_accumulation(self):
        """Test basic gradient accumulation."""
        accumulator = GradientAccumulator(accumulation_steps=3)

        # Mock optimizer
        optimizer = Mock()
        optimizer.param_groups = [{"params": [Mock()]}]
        optimizer.step = Mock()
        optimizer.zero_grad = Mock()

        # Test accumulation steps
        loss1 = torch.tensor(2.0, requires_grad=True)
        loss2 = torch.tensor(4.0, requires_grad=True)
        loss3 = torch.tensor(6.0, requires_grad=True)

        # Step 1 - accumulate
        step_info1 = accumulator.accumulate_step(loss1, optimizer)
        assert not step_info1["should_update"]
        assert step_info1["step_loss"] == 2.0 / 3
        assert step_info1["accumulated_loss"] == 2.0 / 3
        assert not optimizer.step.called

        # Step 2 - accumulate
        step_info2 = accumulator.accumulate_step(loss2, optimizer)
        assert not step_info2["should_update"]
        assert step_info2["step_loss"] == 4.0 / 3
        assert step_info2["accumulated_loss"] == (2.0 + 4.0) / 3
        assert not optimizer.step.called

        # Step 3 - update
        step_info3 = accumulator.accumulate_step(loss3, optimizer)
        assert step_info3["should_update"]
        assert step_info3["step_loss"] == 6.0 / 3
        assert step_info3["accumulated_loss"] == (2.0 + 4.0 + 6.0) / 3
        assert optimizer.step.called
        assert optimizer.zero_grad.called

    def test_gradient_clipping_norm(self):
        """Test gradient norm clipping."""
        accumulator = GradientAccumulator(
            accumulation_steps=1, clip_grad_norm=1.0
        )  # Update after each step

        parameters = [
            nn.Parameter(torch.zeros(2, 2)),
            nn.Parameter(torch.zeros(2)),
        ]
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, 10.0)

        optimizer = Mock()
        optimizer.param_groups = [{"params": parameters}]
        optimizer.step = Mock()
        optimizer.zero_grad = Mock()
        accumulator._update_parameters(optimizer)

        # Check that gradients were clipped
        grad_norms = [torch.norm(parameter.grad.detach()) for parameter in parameters]
        if grad_norms:
            total_norm = torch.norm(torch.stack(grad_norms))
            assert total_norm <= 1.0

    def test_gradient_clipping_value(self):
        """Test gradient value clipping."""
        accumulator = GradientAccumulator(
            accumulation_steps=1, clip_grad_value=0.1
        )  # Update after each step

        parameters = [
            nn.Parameter(torch.zeros(2, 2)),
            nn.Parameter(torch.zeros(2)),
        ]
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, 5.0)

        optimizer = Mock()
        optimizer.param_groups = [{"params": parameters}]
        optimizer.step = Mock()
        optimizer.zero_grad = Mock()
        accumulator._update_parameters(optimizer)

        # Check that gradients are clipped by value
        for parameter in parameters:
            assert torch.all(parameter.grad <= 0.1)
            assert torch.all(parameter.grad >= -0.1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision(self):
        """Test mixed precision support."""
        accumulator = GradientAccumulator(
            accumulation_steps=2,
            mixed_precision=True,
            scaler=torch.cuda.amp.GradScaler(),
        )

        model = SimpleModel().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(5, 10).cuda()
        y = torch.randn(5, 1).cuda()

        with torch.cuda.amp.autocast():
            output = model(x)
            loss = nn.MSELoss()(output, y)

        step_info = accumulator.accumulate_step(loss, optimizer)
        assert not step_info["should_update"]

        # Second step should trigger update
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = nn.MSELoss()(output, y)

        step_info = accumulator.accumulate_step(loss, optimizer)
        assert step_info["should_update"]

    def test_reset(self):
        """Test accumulator reset."""
        accumulator = GradientAccumulator(accumulation_steps=2)

        # Mock optimizer
        optimizer = Mock()
        optimizer.param_groups = [{"params": [Mock()]}]

        # Accumulate one step
        loss = torch.tensor(1.0, requires_grad=True)
        accumulator.accumulate_step(loss, optimizer)
        assert accumulator.step_count == 1
        assert accumulator.accumulated_loss == 1.0 / 2

        # Reset
        accumulator.reset()
        assert accumulator.step_count == 0
        assert accumulator.accumulated_loss == 0.0

    def test_accumulation_context(self):
        """Test accumulation context manager."""
        accumulator = GradientAccumulator(accumulation_steps=3)

        optimizer = Mock()
        optimizer.param_groups = [{"params": [Mock()]}]
        optimizer.step = Mock()
        optimizer.zero_grad = Mock()

        losses = [torch.tensor(float(i + 1), requires_grad=True) for i in range(5)]

        with accumulator.accumulation_context(optimizer) as accumulate:
            for i, loss in enumerate(losses):
                step_info = accumulate(loss)
                if i < 2:  # First two steps
                    assert not step_info["should_update"]
                elif i == 2:  # Third step - update
                    assert step_info["should_update"]
                elif i == 3:  # Fourth step - accumulate again
                    assert not step_info["should_update"]
                # i == 4 will be handled by context manager finalizer


class TestGradientAccumulationTrainer:
    """Test GradientAccumulationTrainer class."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = Mock()
        optimizer = Mock()

        trainer = GradientAccumulationTrainer(
            model=model,
            optimizer=optimizer,
            accumulation_steps=4,
            clip_grad_norm=1.0,
            mixed_precision=False,
        )

        assert trainer.accumulation_steps == 4
        assert trainer.accumulator.accumulation_steps == 4
        assert trainer.scaler is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trainer_mixed_precision(self):
        """Test trainer with mixed precision."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = GradientAccumulationTrainer(
            model=model,
            optimizer=optimizer,
            accumulation_steps=2,
            mixed_precision=True,
            device="cuda",
        )

        assert trainer.mixed_precision
        assert trainer.scaler is not None

    def test_training_step(self):
        """Test training step."""
        model = Mock(return_value=torch.ones(3, 1))
        optimizer = Mock()

        trainer = GradientAccumulationTrainer(
            model=model, optimizer=optimizer, accumulation_steps=2
        )
        trainer.accumulator.accumulate_step = Mock(
            side_effect=[
                {
                    "step_loss": 0.5,
                    "accumulated_loss": 0.5,
                    "step_count": 1,
                    "parameters_updated": False,
                },
                {
                    "step_loss": 0.5,
                    "accumulated_loss": 1.0,
                    "step_count": 2,
                    "parameters_updated": True,
                },
            ]
        )

        # Create batch
        batch = torch.randn(3, 10)

        def loss_fn(output, target):
            del output, target
            return torch.tensor(1.0, requires_grad=True)

        # First step
        step_info = trainer.training_step(batch, loss_fn)
        assert not step_info["parameters_updated"]
        assert step_info["effective_batch_size"] == 6  # 3 * 2

        # Second step - should update
        step_info = trainer.training_step(batch, loss_fn)
        assert step_info["parameters_updated"]
        assert step_info["effective_batch_size"] == 6

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        trainer = GradientAccumulationTrainer(
            Mock(),
            Mock(),
            accumulation_steps=4,
        )

        assert trainer.get_effective_batch_size(8) == 32
        assert trainer.get_effective_batch_size(16) == 64


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_gradient_accumulator(self):
        """Test gradient accumulator creation."""
        accumulator = create_gradient_accumulator(
            accumulation_steps=4, clip_grad_norm=2.0, mixed_precision=False
        )

        assert accumulator.accumulation_steps == 4
        assert accumulator.clip_grad_norm == 2.0
        assert not accumulator.mixed_precision

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_gradient_accumulator_mixed_precision(self):
        """Test gradient accumulator creation with mixed precision."""
        accumulator = create_gradient_accumulator(
            accumulation_steps=2, mixed_precision=True, device="cuda"
        )

        assert accumulator.mixed_precision
        assert accumulator.scaler is not None

    def test_effective_batch_size_info(self):
        """Test effective batch size information."""
        info = effective_batch_size_info(
            actual_batch_size=8, accumulation_steps=4, num_epochs=10
        )

        assert info["actual_batch_size"] == 8
        assert info["accumulation_steps"] == 4
        assert info["effective_batch_size"] == 32
        assert info["gradient_updates_per_epoch"] == 8
        assert "training_notes" in info
        assert len(info["training_notes"]) == 4
