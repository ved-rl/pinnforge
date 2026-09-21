"""
Numerical solvers for comparison and validation.
"""
import torch
import numpy as np
from typing import Tuple, Callable, Optional
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class PDESolver:
    """Base class for PDE solvers."""
    
    def __init__(self, pde_fn: Callable):
        self.pde_fn = pde_fn
    
    def solve(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Solve the PDE numerically."""
        raise NotImplementedError


class NumericalSolver(PDESolver):
    """
    Finite difference solver for 1D PDEs.
    """
    
    def __init__(
        self,
        pde_fn: Callable,
        nx: int = 100,
        nt: int = 100,
        x_range: Tuple[float, float] = (0, 1),
        t_range: Tuple[float, float] = (0, 1),
    ):
        super().__init__(pde_fn)
        self.nx = nx
        self.nt = nt
        self.x_range = x_range
        self.t_range = t_range
        
        self.dx = (x_range[1] - x_range[0]) / nx
        self.dt = (t_range[1] - t_range[0]) / nt
        self.x = np.linspace(x_range[0], x_range[1], nx + 1)
        self.t = np.linspace(t_range[0], t_range[1], nt + 1)
    
    def solve_heat(
        self,
        initial_condition: Callable,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """
        Solve heat equation using implicit finite differences.
        u_t = alpha * u_xx
        """
        # Create tridiagonal matrix
        A = diags(
            [
                -alpha * self.dt / (2 * self.dx**2),
                1 + alpha * self.dt / self.dx**2,
                -alpha * self.dt / (2 * self.dx**2),
            ],
            [-1, 0, 1],
            shape=(self.nx + 1, self.nx + 1),
        ).tocsr()
        
        # Apply boundary conditions
        A = A.tolil()
        A[0, :] = 0
        A[0, 0] = 1
        A[self.nx, :] = 0
        A[self.nx, self.nx] = 1
        A = A.tocsr()
        
        # Initial condition
        u = initial_condition(self.x)
        u_history = [u.copy()]
        
        # Time stepping
        for _ in range(self.nt):
            u = spsolve(A, u)
            u_history.append(u.copy())
        
        return np.array(u_history)
    
    def solve_burgers(
        self,
        initial_condition: Callable,
        nu: float = 0.01,
    ) -> np.ndarray:
        """
        Solve Burgers equation using finite differences.
        u_t + u*u_x = nu*u_xx
        """
        u = initial_condition(self.x)
        u_history = [u.copy()]
        
        for _ in range(self.nt):
            # Explicit scheme
            u_new = u.copy()
            for i in range(1, self.nx):
                # Advection term
                adv = u[i] * (u[i+1] - u[i-1]) / (2 * self.dx)
                # Diffusion term
                diff = nu * (u[i+1] - 2*u[i] + u[i-1]) / self.dx**2
                u_new[i] = u[i] - self.dt * adv + self.dt * diff
            
            # Boundary conditions (Dirichlet)
            u_new[0] = 0
            u_new[self.nx] = 0
            
            u = u_new
            u_history.append(u.copy())
        
        return np.array(u_history)
    
    def to_torch(self, u: np.ndarray) -> torch.Tensor:
        """Convert numpy solution to torch tensor."""
        return torch.tensor(u, dtype=torch.float32)