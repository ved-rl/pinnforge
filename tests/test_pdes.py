import pytest
import torch
from pinnforge.physics.pdes import (
    BurgersEquation,
    HeatEquation,
    WaveEquation,
    create_pde,
)


def test_burgers_equation():
    """Test Burgers equation."""
    pde = BurgersEquation(nu=0.01)
    assert pde.nu == 0.01
    assert pde.name == "Burgers"
    
    # Test residual computation
    u = torch.rand(10, 1)
    u_t = torch.rand(10, 1)
    u_x = torch.rand(10, 1)
    u_xx = torch.rand(10, 1)
    
    residual = pde.residual(u, u_t, u_x, u_xx)
    assert residual.shape == (10, 1)
    
    # Test initial condition
    x = torch.linspace(0, 1, 10).reshape(-1, 1)
    ic = pde.initial_condition(x)
    assert ic.shape == (10, 1)
    assert torch.allclose(ic, -torch.sin(torch.pi * x))


def test_heat_equation():
    """Test Heat equation."""
    pde = HeatEquation(alpha=1.0)
    assert pde.alpha == 1.0
    assert pde.name == "Heat"
    
    # Test residual
    u = torch.rand(10, 1)
    u_t = torch.rand(10, 1)
    u_x = torch.rand(10, 1)
    u_xx = torch.rand(10, 1)
    
    residual = pde.residual(u, u_t, u_x, u_xx)
    assert residual.shape == (10, 1)
    
    # Test analytical solution
    x = torch.rand(10, 1)
    t = torch.rand(10, 1)
    u_analytical = pde.analytical_solution(x, t)
    assert u_analytical.shape == (10, 1)
    assert not torch.isnan(u_analytical).any()


def test_wave_equation():
    """Test Wave equation."""
    pde = WaveEquation(c=1.0)
    assert pde.c == 1.0
    assert pde.name == "Wave"


def test_pde_factory():
    """Test PDE factory function."""
    pde1 = create_pde("burgers", nu=0.02)
    assert isinstance(pde1, BurgersEquation)
    assert pde1.nu == 0.02
    
    pde2 = create_pde("heat", alpha=2.0)
    assert isinstance(pde2, HeatEquation)
    assert pde2.alpha == 2.0
    
    with pytest.raises(ValueError):
        create_pde("invalid_pde")