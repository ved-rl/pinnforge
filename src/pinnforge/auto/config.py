"""
Auto-configuration heuristics for PINNs.
Based on PDE characteristics, auto-selects hyperparameters.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class AutoConfig:
    """
    Auto-selected configuration for a PINN solve.
    """
    # Network
    layers: List[int] = field(default_factory=lambda: [2, 64, 64, 64, 1])
    activation: str = "tanh"
    
    # Fourier features
    use_fourier: bool = False
    fourier_scale: float = 8.0
    
    # Training
    n_collocation: int = 10000
    n_boundary: int = 200
    n_initial: int = 200
    epochs: int = 5000
    learning_rate: float = 1e-3
    batch_size: int = 1000
    
    # Loss
    use_adaptive_weights: bool = False
    weight_bc: float = 1.0
    weight_ic: float = 1.0
    
    # Precision
    precision: str = "float32"
    
    @classmethod
    def for_pde(cls, pde_name: str, nu: Optional[float] = None, 
                domain: Optional[List[Tuple]] = None) -> "AutoConfig":
        """
        Auto-select config based on PDE type and parameters.
        
        Heuristics based on PINNacle findings and common failure modes:
        - Stiff problems (low nu) need more collocation + larger Fourier scale
        - High-frequency problems need wider networks
        - Long time domains need more collocation points
        """
        config = cls()
        
        if pde_name.lower() == "burgers":
            if nu is not None and nu < 0.02:
                # Stiff Burgers: needs more points, larger Fourier scale
                config.layers = [2, 128, 128, 128, 1]
                config.fourier_scale = 15.0
                config.n_collocation = 20000
                config.epochs = 10000
                config.learning_rate = 5e-4
            else:
                config.layers = [2, 64, 64, 64, 1]
                config.fourier_scale = 10.0
                
        elif pde_name.lower() == "heat":
            config.layers = [2, 64, 64, 64, 1]
            config.fourier_scale = 8.0
            config.epochs = 3000
            
        elif pde_name.lower() == "wave":
            # Wave equations are high-frequency sensitive
            config.layers = [2, 96, 96, 96, 1]
            config.fourier_scale = 20.0
            config.epochs = 8000
            config.precision = "float64"  # Waves need precision
            
        elif pde_name.lower() == "navier-stokes":
            # NS is the hardest: needs big network + precision
            config.layers = [3, 128, 128, 128, 128, 2]  # (x,y,t) -> (u,v)
            config.fourier_scale = 12.0
            config.n_collocation = 50000
            config.epochs = 20000
            config.learning_rate = 1e-4
            config.precision = "float64"
            
        # Domain-based adjustments
        if domain:
            # Longer time domains need more collocation
            t_range = domain[1] if len(domain) > 1 else (0, 1)
            t_span = t_range[1] - t_range[0]
            if t_span > 2:
                config.n_collocation = int(config.n_collocation * t_span / 1.0)
                
        return config