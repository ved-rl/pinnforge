"""
AutoPINN: One-function PDE solver.
"""
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import warnings

from pinnforge.core.pinn import PINN, PINNTrainer
from pinnforge.physics.pdes import create_pde, PDE
from pinnforge.utils.data import (
    create_collocation_points,
    create_boundary_points,
    create_initial_points,
)
from pinnforge.auto.config import AutoConfig


class AutoPINN:
    """
    Automatic PINN solver.
    
    One call to solve() does everything:
    1. Selects hyperparameters based on PDE characteristics
    2. Generates collocation/boundary/initial points
    3. Builds network with Fourier features
    4. Trains with adaptive loss weighting
    5. Validates against numerical solver
    6. Generates report with plots and metrics
    
    Example:
        >>> solver = AutoPINN("burgers", nu=0.01)
        >>> result = solver.solve()
        >>> print(f"Relative L2 error: {result['metrics']['relative_l2_error']:.2e}")
    """
    
    def __init__(
        self,
        pde_name: str,
        nu: Optional[float] = None,
        domain: Optional[List[Tuple[float, float]]] = None,
        config: Optional[AutoConfig] = None,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """
        Args:
            pde_name: Name of PDE ('burgers', 'heat', 'wave', 'navier-stokes')
            nu: Viscosity parameter (for Burgers)
            domain: Domain specification [(x_min, x_max), (t_min, t_max)]
            config: Override auto-config (optional)
            device: 'cpu' or 'cuda'
            verbose: Print progress
        """
        self.pde_name = pde_name
        self.nu = nu
        self.domain = domain or [(0, 1), (0, 1)]
        self.device = device
        self.verbose = verbose
        
        # Auto-select config if not provided
        if config is None:
            self.config = AutoConfig.for_pde(pde_name, nu=nu, domain=domain)
        else:
            self.config = config
        
        # Create PDE
        pde_kwargs = {}
        if nu is not None:
            pde_kwargs["nu"] = nu
        self.pde = create_pde(pde_name, **pde_kwargs)
        
        # Placeholders
        self.model = None
        self.trainer = None
        self.result = None
    
    @classmethod
    def from_symbolic(
        cls,
        equation,
        independent_vars: List,
        dependent_var,
        domain: Optional[List[Tuple[float, float]]] = None,
        device: str = "cpu",
        **kwargs,
    ) -> "AutoPINN":
        """
        Create AutoPINN from a SymPy equation.
        
        Example:
            >>> import sympy as sp
            >>> x, t, u = sp.symbols('x t u')
            >>> pde = sp.diff(u,t) + u*sp.diff(u,x) - 0.01*sp.diff(u,x,2)
            >>> solver = AutoPINN.from_symbolic(pde, [x,t], u)
        """
        from pinnforge.symbolic.pde import SymbolicPDE
        
        symbolic_pde = SymbolicPDE(equation, dependent_var, independent_vars)
        
        # Determine "name" for config selection
        n_vars = len(independent_vars)
        if n_vars == 2:
            name = "symbolic_2d"
        elif n_vars == 3:
            name = "symbolic_3d"
        else:
            name = "symbolic"
        
        # Create instance with symbolic PDE
        instance = cls.__new__(cls)
        instance.pde_name = name
        instance.nu = None
        instance.domain = domain or [(0, 1)] * n_vars
        instance.device = device
        instance.verbose = kwargs.get("verbose", True)
        instance.pde = symbolic_pde
        instance.config = AutoConfig.for_pde(name, domain=domain)
        
        if "config" in kwargs:
            instance.config = kwargs["config"]
        
        instance.model = None
        instance.trainer = None
        instance.result = None
        
        return instance
    
    def solve(self, validate: bool = True) -> Dict[str, Any]:
        """
        Solve the PDE. This is the main entry point.
        
        Returns:
            Dictionary with:
            - model: Trained PINN
            - trainer: PINNTrainer
            - history: Training history
            - metrics: Validation metrics (if validate=True)
            - solution: Callable for predictions
        """
        if self.verbose:
            print(f"[AutoPINN] Solving {self.pde_name} PDE")
            print(f"  Config: {self.config.layers}, Fourier scale={self.config.fourier_scale}")
            print(f"  Epochs: {self.config.epochs}, LR: {self.config.learning_rate}")
        
        # 1. Build model
        self.model = PINN(
            layers=self.config.layers,
            activation=self.config.activation,
            use_fourier_features=self.config.use_fourier,
            fourier_scale=self.config.fourier_scale,
            use_adaptive_activation=False,
            device=self.device,
        )
        
        # 2. Generate data
        x_pde, t_pde = create_collocation_points(
            x_range=self.domain[0],
            t_range=self.domain[1] if len(self.domain) > 1 else (0, 1),
            n_points=self.config.n_collocation,
            device=self.device,
        )
        x_bc, t_bc = create_boundary_points(
            x_range=self.domain[0],
            t_range=self.domain[1] if len(self.domain) > 1 else (0, 1),
            n_points=self.config.n_boundary,
            device=self.device,
        )
        bc_values = self.pde.boundary_condition(x_bc, t_bc)
        x_ic, t_ic = create_initial_points(
            x_range=self.domain[0],
            n_points=self.config.n_initial,
            device=self.device,
        )
        ic_values = self.pde.initial_condition(x_ic)
        
        # 3. Train
        self.trainer = PINNTrainer(
            model=self.model,
            pde_fn=self.pde.residual,
            learning_rate=self.config.learning_rate,
            use_adaptive_weights=self.config.use_adaptive_weights,
            device=self.device,
        )
        
        history = self.trainer.train(
            x_pde=x_pde, t_pde=t_pde,
            x_bc=x_bc, t_bc=t_bc, bc_values=bc_values,
            x_ic=x_ic, t_ic=t_ic, ic_values=ic_values,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            weights={"pde": 1.0, "bc": self.config.weight_bc, "ic": self.config.weight_ic},
            verbose=self.verbose,
        )
        
        # 4. Validate (optional)
        metrics = {}
        if validate and hasattr(self.pde, 'analytical_solution'):
            metrics = self._validate()
        
        # 5. Build result
        self.result = {
            "model": self.model,
            "trainer": self.trainer,
            "history": history,
            "metrics": metrics,
            "config": self.config,
            "predict": lambda x, t: self.trainer.predict(
                torch.tensor(x, dtype=torch.float32, device=self.device).reshape(-1, 1),
                torch.tensor(t, dtype=torch.float32, device=self.device).reshape(-1, 1),
            ),
        }
        
        if self.verbose:
            print(f"[AutoPINN] Complete.")
            if metrics:
                print(f"  Relative L2 error: {metrics.get('relative_l2_error', 'N/A'):.4e}")
        
        return self.result
    
    def _validate(self) -> Dict[str, float]:
        """Validate against analytical solution."""
        from pinnforge.benchmarks.metrics import compute_metrics
        
        # Dense test grid
        n_test = 5000
        x_test = torch.linspace(
            self.domain[0][0], self.domain[0][1], int(np.sqrt(n_test)),
            device=self.device
        ).reshape(-1, 1)
        t_test = torch.linspace(
            self.domain[1][0], self.domain[1][1], int(np.sqrt(n_test)),
            device=self.device
        ).reshape(-1, 1)
        
        X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
        X_flat = X.reshape(-1, 1)
        T_flat = T.reshape(-1, 1)
        
        with torch.no_grad():
            u_pred = self.trainer.predict(X_flat, T_flat)
        
        try:
            u_true = self.pde.analytical_solution(X_flat, T_flat)
            return compute_metrics(u_pred, u_true)
        except (NotImplementedError, AttributeError):
        # No analytical solution available, skip validation
            return {}

    def report(self, save_path: Optional[str] = None):
        """
        Generate a visual report of the solution.
        
        Args:
            save_path: Path to save the report image. If None, displays.
        """
        if self.result is None:
            raise RuntimeError("Must call solve() before report()")
        
        import matplotlib.pyplot as plt
        
        # Create dense grid
        n = 100
        x = torch.linspace(self.domain[0][0], self.domain[0][1], n, device=self.device)
        t = torch.linspace(
            self.domain[1][0] if len(self.domain) > 1 else 0,
            self.domain[1][1] if len(self.domain) > 1 else 1,
            n, device=self.device
        )
        X, T = torch.meshgrid(x, t, indexing='ij')
        X_flat = X.reshape(-1, 1)
        T_flat = T.reshape(-1, 1)
        
        with torch.no_grad():
            u_pred = self.trainer.predict(X_flat, T_flat).reshape(n, n)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Predicted solution
        im1 = axes[0].contourf(X.cpu().numpy(), T.cpu().numpy(), u_pred.cpu().numpy(), levels=50, cmap='viridis')
        axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
        axes[0].set_title('PINN Solution')
        plt.colorbar(im1, ax=axes[0])
        
        # Training loss
        if 'loss_total' in self.result['history']:
            axes[1].semilogy(self.result['history']['loss_total'])
            axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Convergence')
            axes[1].grid(True, alpha=0.3)
        
        # Error (if analytical available)
        if 'relative_l2_error' in self.result['metrics']:
            u_true = self.pde.analytical_solution(X_flat, T_flat).reshape(n, n)
            error = torch.abs(u_pred - u_true)
            im3 = axes[2].contourf(X.cpu().numpy(), T.cpu().numpy(), error.cpu().numpy(), levels=50, cmap='hot')
            axes[2].set_xlabel('x'); axes[2].set_ylabel('t')
            axes[2].set_title(f'Error (max: {error.max():.2e})')
            plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[AutoPINN] Report saved to {save_path}")
        else:
            plt.show()


def solve_pde(
    pde_name: str,
    nu: Optional[float] = None,
    domain: Optional[List[Tuple[float, float]]] = None,
    epochs: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    One-line PDE solver.
    
    Example:
        >>> result = solve_pde("burgers", nu=0.01, epochs=5000)
        >>> print(result['metrics']['relative_l2_error'])
    """
    solver = AutoPINN(
        pde_name=pde_name,
        nu=nu,
        domain=domain,
        device=device,
        verbose=verbose,
    )
    
    if epochs is not None:
        solver.config.epochs = epochs
    
    return solver.solve()