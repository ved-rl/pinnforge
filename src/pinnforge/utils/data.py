import numpy as np
import torch
from typing import Tuple, Optional


def generate_simulation_data(
    pde_fn: callable,
    x_range: Tuple[float, float] = (0, 1),
    t_range: Tuple[float, float] = (0, 1),
    n_points: int = 1000,
    noise_level: float = 0.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for PINN training.
    
    Args:
        pde_fn: PDE function (for generating ground truth)
        x_range: Spatial domain
        t_range: Time domain
        n_points: Number of points
        noise_level: Gaussian noise standard deviation
    
    Returns:
        x, t, u (ground truth)
    """
    # Generate random points in domain
    x = torch.rand(n_points, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    t = torch.rand(n_points, 1, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    
    # Compute ground truth
    u = torch.sin(torch.pi * x) * torch.exp(-t)  # Placeholder
    
    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(u) * noise_level
        u = u + noise
    
    return x, t, u


def add_noise(
    data: torch.Tensor,
    noise_level: float = 0.01,
    noise_type: str = "gaussian",
) -> torch.Tensor:
    """
    Add noise to data for robustness testing.
    
    Args:
        data: Input data tensor
        noise_level: Standard deviation of noise
        noise_type: 'gaussian' or 'uniform'
    """
    if noise_type == "gaussian":
        noise = torch.randn_like(data) * noise_level
    elif noise_type == "uniform":
        noise = (torch.rand_like(data) - 0.5) * 2 * noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return data + noise


def create_collocation_points(
    x_range: Tuple[float, float] = (0, 1),
    t_range: Tuple[float, float] = (0, 1),
    n_points: int = 10000,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate collocation points for PDE residual evaluation."""
    x = torch.rand(n_points, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    t = torch.rand(n_points, 1, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    return x, t


def create_boundary_points(
    x_range: Tuple[float, float] = (0, 1),
    t_range: Tuple[float, float] = (0, 1),
    n_points: int = 100,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Left and right boundaries
    n_left = n_right = n_points // 2
    
    x_left = torch.zeros(n_left, 1, device=device)
    t_left = torch.rand(n_left, 1, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    
    x_right = torch.ones(n_right, 1, device=device)
    t_right = torch.rand(n_right, 1, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    
    x = torch.cat([x_left, x_right])
    t = torch.cat([t_left, t_right])
    
    return x, t


def create_initial_points(
    x_range: Tuple[float, float] = (0, 1),
    n_points: int = 100,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate initial condition points."""
    x = torch.rand(n_points, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    t = torch.zeros(n_points, 1, device=device)
    return x, t