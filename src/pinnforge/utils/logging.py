import json
import yaml
import os
import datetime
import torch
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np


class ExperimentLogger:
    """
    Automatic experiment logging and tracking for PINN training.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        save_model: bool = True,
    ):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_model = save_model
        
        # Create log directory
        self.exp_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Initialize log files
        self.metrics_file = os.path.join(self.exp_dir, "metrics.json")
        self.config_file = os.path.join(self.exp_dir, "config.yaml")
        self.model_dir = os.path.join(self.exp_dir, "checkpoints")
        if save_model:
            os.makedirs(self.model_dir, exist_ok=True)
        
        # Track metrics
        self.metrics = []
        self.best_model_state = None
        self.best_loss = float("inf")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        with open(self.config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics."""
        entry = {"step": step, **metrics}
        self.metrics.append(entry)
        
        # Save to JSON file
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        # Check if this is the best model
        if "loss_total" in metrics and metrics["loss_total"] < self.best_loss:
            self.best_loss = metrics["loss_total"]
            self.best_model_state = None  # Would store model state here
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
        """Save model checkpoint."""
        if not self.save_model:
            return
        
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": self.best_loss,
        }
        
        checkpoint_path = os.path.join(self.model_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Also save best model separately
        if self.best_model_state is not None:
            best_path = os.path.join(self.model_dir, "best_model.pt")
            torch.save({"model_state": self.best_model_state}, best_path)
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_loss = checkpoint["best_loss"]
        return checkpoint["epoch"]
    
    def plot_metrics(self, metrics_to_plot: List[str] = None):
        """Generate plots of training metrics."""
        if not self.metrics:
            print("No metrics to plot")
            return
        
        if metrics_to_plot is None:
            metrics_to_plot = ["loss_total", "loss_pde", "loss_bc", "loss_ic"]
        
        steps = [m["step"] for m in self.metrics]
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 3*len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, metrics_to_plot):
            if metric_name in self.metrics[0]:
                values = [m[metric_name] for m in self.metrics]
                ax.semilogy(steps, values)
                ax.set_xlabel("Step")
                ax.set_ylabel(metric_name)
                ax.set_title(f"{metric_name} over training")
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.exp_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.show()
    
    def log_system_info(self):
        """Log system and environment information."""
        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(self.exp_dir, "system_info.json"), "w") as f:
            json.dump(info, f, indent=2)