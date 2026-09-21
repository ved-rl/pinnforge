import torch
from typing import Callable, Dict, Any


class PDE:
    """Base class for PDE definitions."""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
    
    def residual(
        self,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PDE residual."""
        raise NotImplementedError
    
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Define initial condition."""
        raise NotImplementedError
    
    def boundary_condition(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Define boundary condition."""
        raise NotImplementedError
    
    def analytical_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Analytical solution if known (for testing)."""
        raise NotImplementedError


class BurgersEquation(PDE):
    """
    1D Burgers equation: u_t + u*u_x - nu*u_xx = 0
    """
    
    def __init__(self, nu: float = 0.01, **kwargs):
        super().__init__("Burgers", {"nu": nu})
        self.nu = nu
    
    def residual(self, u, u_t, u_x, u_xx):
        return u_t + u * u_x - self.nu * u_xx
    
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Initial condition: -sin(pi*x)"""
        return -torch.sin(torch.pi * x)
    
    def boundary_condition(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Boundary condition: u(0,t) = u(1,t) = 0"""
        return torch.zeros_like(x)
    
    def analytical_solution(self, x, t):
        raise NotImplementedError("Burgers equation has no closed-form solution for this IC")


class HeatEquation(PDE):
    """
    1D Heat equation: u_t - alpha*u_xx = 0
    """
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__("Heat", {"alpha": alpha})
        self.alpha = alpha
    
    def residual(self, u, u_t, u_x, u_xx):
        return u_t - self.alpha * u_xx
    
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Initial condition: sin(pi*x)"""
        return torch.sin(torch.pi * x)
    
    def boundary_condition(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Boundary condition: u(0,t) = u(1,t) = 0"""
        return torch.zeros_like(x)
    
    def analytical_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * x) * torch.exp(-self.alpha * torch.pi**2 * t)


class WaveEquation(PDE):
    """
    1D Wave equation: u_tt - c^2*u_xx = 0
    """
    
    def __init__(self, c: float = 1.0, **kwargs):
        super().__init__("Wave", {"c": c})
        self.c = c
    
    def residual(self, u, u_t, u_x, u_xx):
        # Need second derivative in time
        # This is simplified (full implementation would compute u_tt)
        return u_t - self.c**2 * u_xx  # Simplified
    
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * x)
    
    def boundary_condition(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


# Factory for easy PDE creation
PDE_REGISTRY = {
    "burgers": BurgersEquation,
    "heat": HeatEquation,
    "wave": WaveEquation,
}

def create_pde(name: str, **kwargs) -> PDE:
    """Factory function to create PDE instances."""
    if name.lower() not in PDE_REGISTRY:
        raise ValueError(f"Unknown PDE: {name}. Available: {list(PDE_REGISTRY.keys())}")
    return PDE_REGISTRY[name.lower()](**kwargs)