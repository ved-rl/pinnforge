import pytest
import torch
import numpy as np
from pinnforge.core.pinn import PINN, PINNTrainer
from pinnforge.physics.pdes import BurgersEquation, HeatEquation


def test_pinn_initialization():
    """Test PINN model initialization."""
    model = PINN(layers=[2, 10, 10, 1], activation="tanh")
    assert len(model.network) == 3
    assert isinstance(model.network[0], torch.nn.Linear)
    assert model.device == "cpu"


def test_pinn_forward():
    """Test forward pass."""
    model = PINN(layers=[2, 10, 10, 1])
    x = torch.randn(5, 2)
    y = model(x)
    assert y.shape == (5, 1)
    assert not torch.isnan(y).any()


def test_pde_residual_burgers():
    """Test Burgers equation residual computation."""
    pde = BurgersEquation(nu=0.01)
    model = PINN(layers=[2, 10, 10, 1])
    
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    
    # Test loss_pde method
    loss = model.loss_pde(x, t, pde.residual)
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_pinn_trainer():
    """Test PINN trainer basic functionality."""
    pde = HeatEquation(alpha=1.0)
    model = PINN(layers=[2, 10, 10, 1])
    trainer = PINNTrainer(model, pde.residual)
    
    # Create dummy data
    x = torch.rand(100, 1)
    t = torch.rand(100, 1)
    x_bc = torch.rand(20, 1)
    t_bc = torch.rand(20, 1)
    bc_values = torch.zeros(20, 1)
    x_ic = torch.rand(20, 1)
    t_ic = torch.zeros(20, 1)
    ic_values = torch.sin(torch.pi * x_ic)
    
def test_heat_quick_run():
    from pinnforge import solve_pde
    r = solve_pde('heat', epochs=300, verbose=False)
    assert r['metrics']['relative_l2_error'] < 0.5


def test_forward_shape():
    import torch
    from pinnforge import PINN
    model = PINN(layers=[2, 16, 16, 1], use_fourier_features=False)
    out = model(torch.randn(5, 2))
    assert out.shape == (5, 1)