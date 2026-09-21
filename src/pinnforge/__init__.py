"""
PINNForge - Physics-Informed Neural Networks made easy
"""

from pinnforge.core.pinn import PINN, PINNTrainer, FourierFeatureMapping
from pinnforge.physics.pdes import BurgersEquation, HeatEquation, WaveEquation, PDE, create_pde
from pinnforge.utils.data import generate_simulation_data, add_noise
from pinnforge.utils.logging import ExperimentLogger

# Auto-solver
from pinnforge.auto import AutoPINN, solve_pde

# Symbolic
try:
    from pinnforge.symbolic import SymbolicPDE, SymbolicPINNTrainer
except ImportError:
    SymbolicPDE = None

# Benchmarks
try:
    from pinnforge.benchmarks import compute_relative_l2_error, compute_metrics, BenchmarkRunner
except ImportError:
    compute_relative_l2_error = None

__version__ = "0.1.0"

__all__ = [
    # One-line API
    "solve_pde",
    "AutoPINN",
    
    # Core
    "PINN",
    "PINNTrainer",
    "FourierFeatureMapping",
    
    # Physics
    "BurgersEquation",
    "HeatEquation",
    "WaveEquation",
    "PDE",
    "create_pde",
    
    # Utils
    "generate_simulation_data",
    "add_noise",
    "ExperimentLogger",
    
    # Symbolic
    "SymbolicPDE",
    "SymbolicPINNTrainer",
    
    # Benchmarks
    "compute_relative_l2_error",
    "compute_metrics",
    "BenchmarkRunner",
]