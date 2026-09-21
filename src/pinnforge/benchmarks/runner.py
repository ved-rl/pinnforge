"""
Automated benchmarking runner.
"""
import torch
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate


class BenchmarkRunner:
    """
    Run automated benchmarks for PINN models.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
    
    def run_benchmark(
        self,
        model,
        trainer,
        test_data: Dict[str, torch.Tensor],
        true_solution: Optional[torch.Tensor] = None,
        name: str = "benchmark",
    ) -> Dict[str, Any]:
        """
        Run a single benchmark.
        
        Args:
            model: Trained PINN model
            trainer: PINN trainer
            test_data: Dictionary with 'x', 't' tensors
            true_solution: Ground truth solution (optional)
            name: Benchmark name
            
        Returns:
            Benchmark results dictionary
        """
        from .metrics import compute_metrics
        
        result = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "model_params": sum(p.numel() for p in model.parameters()),
        }
        
        # Timing inference
        x = test_data['x']
        t = test_data['t']
        
        start_time = time.time()
        with torch.no_grad():
            if hasattr(trainer, 'predict'):
                u_pred = trainer.predict(x, t)
            else:
                u_pred = model(torch.cat([x, t], dim=1))
        inference_time = time.time() - start_time
        
        result["inference_time"] = inference_time
        result["num_points"] = len(x)
        
        # Compute metrics if true solution is provided
        if true_solution is not None:
            metrics = compute_metrics(u_pred, true_solution)
            result.update(metrics)
        
        # Store training history summary
        if hasattr(trainer, 'history'):
            history = trainer.history
            if 'loss_total' in history:
                final_loss = history['loss_total'][-1] if history['loss_total'] else None
                min_loss = min(history['loss_total']) if history['loss_total'] else None
                result['final_loss'] = final_loss
                result['min_loss'] = min_loss
                result['training_epochs'] = len(history['loss_total'])
        
        self.results.append(result)
        return result
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save all benchmark results to JSON."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_summary(self):
        """Print a summary table of all benchmarks."""
        if not self.results:
            print("No results to display.")
            return
        
        # Prepare table
        table_data = []
        headers = ["Name", "Params", "Points", "Inf. Time (s)", "Rel. L2 Error", "MSE", "R²"]
        
        for r in self.results:
            row = [
                r.get('name', ''),
                r.get('model_params', 0),
                r.get('num_points', 0),
                f"{r.get('inference_time', 0):.4f}",
                f"{r.get('relative_l2_error', 0):.4e}" if 'relative_l2_error' in r else 'N/A',
                f"{r.get('mse', 0):.4e}" if 'mse' in r else 'N/A',
                f"{r.get('r2_score', 0):.4f}" if 'r2_score' in r else 'N/A',
            ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def plot_comparison(
        self,
        u_pred: torch.Tensor,
        u_true: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        title: str = "PINN vs True Solution",
    ):
        """
        Plot comparison between PINN and true solution.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Reshape for plotting
        # This assumes 1D spatial domain
        n_x = int(np.sqrt(len(x)))
        n_t = len(x) // n_x
        
        u_pred_2d = u_pred.reshape(n_x, n_t).detach().cpu().numpy()
        u_true_2d = u_true.reshape(n_x, n_t).detach().cpu().numpy()
        error = np.abs(u_pred_2d - u_true_2d)
        
        x_2d = x.reshape(n_x, n_t).detach().cpu().numpy()
        t_2d = t.reshape(n_x, n_t).detach().cpu().numpy()
        
        # Plot PINN
        im1 = axes[0].contourf(t_2d, x_2d, u_pred_2d, levels=50, cmap='viridis')
        axes[0].set_title('PINN Prediction')
        axes[0].set_xlabel('t')
        axes[0].set_ylabel('x')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot true
        im2 = axes[1].contourf(t_2d, x_2d, u_true_2d, levels=50, cmap='viridis')
        axes[1].set_title('True Solution')
        axes[1].set_xlabel('t')
        axes[1].set_ylabel('x')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot error
        im3 = axes[2].contourf(t_2d, x_2d, error, levels=50, cmap='hot')
        axes[2].set_title(f'Error (max: {error.max():.2e})')
        axes[2].set_xlabel('t')
        axes[2].set_ylabel('x')
        plt.colorbar(im3, ax=axes[2])
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{title.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=150)
        plt.show()


def run_benchmark(pde_name: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run a benchmark for a predefined PDE.
    
    Args:
        pde_name: Name of PDE ('burgers', 'heat', 'wave')
        **kwargs: Additional arguments
        
    Returns:
        Benchmark results
    """
    from pinnforge.physics.pdes import create_pde
    from pinnforge.core.pinn import PINN, PINNTrainer
    from pinnforge.utils.data import (
        create_collocation_points,
        create_boundary_points,
        create_initial_points,
    )
    
    # Create PDE
    pde = create_pde(pde_name, **kwargs)
    
    # Create model
    model = PINN(
        layers=kwargs.get('layers', [2, 50, 50, 50, 1]),
        use_fourier_features=kwargs.get('use_fourier_features', True),
        device=kwargs.get('device', 'cpu'),
    )
    
    # Generate data
    device = kwargs.get('device', 'cpu')
    x_pde, t_pde = create_collocation_points(
        n_points=kwargs.get('n_collocation', 10000),
        device=device,
    )
    x_bc, t_bc = create_boundary_points(
        n_points=kwargs.get('n_boundary', 200),
        device=device,
    )
    bc_values = pde.boundary_condition(x_bc, t_bc)
    x_ic, t_ic = create_initial_points(
        n_points=kwargs.get('n_initial', 200),
        device=device,
    )
    ic_values = pde.initial_condition(x_ic)
    
    # Create trainer
    trainer = PINNTrainer(
        model=model,
        pde_fn=pde.residual,
        device=device,
    )
    
    # Train
    history = trainer.train(
        x_pde=x_pde, t_pde=t_pde,
        x_bc=x_bc, t_bc=t_bc, bc_values=bc_values,
        x_ic=x_ic, t_ic=t_ic, ic_values=ic_values,
        epochs=kwargs.get('epochs', 1000),
        verbose=kwargs.get('verbose', False),
    )
    
    return {
        'model': model,
        'trainer': trainer,
        'history': history,
    }