import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import time
from collections import defaultdict
import math


class FourierFeatureMapping(nn.Module):
    """
    Fourier feature mapping to mitigate spectral bias.
    Maps input coordinates to high-frequency features.
    """
    
    def __init__(self, input_dim: int, output_dim: int, scale: float = 10.0, trainable: bool = False):
        """
        Args:
            input_dim: Number of input dimensions
            output_dim: Number of Fourier features (recommended: 2x input_dim)
            scale: Standard deviation for random sampling
            trainable: Whether to learn the frequencies
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Sample frequencies from Gaussian distribution
        B = torch.randn(output_dim // 2, input_dim) * scale
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature mapping: [cos(2πBx), sin(2πBx)]
        """
        x_proj = 2 * torch.pi * x @ self.B.T
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network with Fourier features and adaptive activation.
    """
    
    def __init__(
        self,
        layers: List[int],
        activation: str = "tanh",
        use_fourier_features: bool = True,
        fourier_scale: float = 10.0,
        use_adaptive_activation: bool = False,
        device: str = "cpu",
    ):
        """
        Args:
            layers: List of neurons per layer [input_dim, hidden1, ..., output_dim]
            activation: Activation function ('tanh', 'relu', 'sigmoid', 'sin')
            use_fourier_features: Enable Fourier feature mapping (recommended)
            fourier_scale: Scale for Fourier feature frequencies
            use_adaptive_activation: Enable adaptive activation (SA-PINN)
            device: Computing device
        """
        super().__init__()
        self.layers_list = layers
        self.device = device
        self.use_fourier = use_fourier_features
        self.use_adaptive = use_adaptive_activation
        self.input_dim = layers[0]
        
        # Fourier feature mapping
        if use_fourier_features:
            fourier_dim = layers[0] * 2
            self.fourier = FourierFeatureMapping(
                input_dim=layers[0],
                output_dim=fourier_dim,
                scale=fourier_scale,
                trainable=False
            )
            # Adjust first layer input dimension
            actual_input_dim = fourier_dim
        else:
            self.fourier = None
            actual_input_dim = layers[0]
        
        # Build network with adjusted input dimension
        modified_layers = [actual_input_dim] + layers[1:]
        self.activation = self._get_activation(activation)
        self.network = self._build_network(modified_layers)
        
        # Adaptive activation parameter
        if use_adaptive_activation:
            self.scale_param = nn.Parameter(torch.tensor(1.0, device=device))
        else:
            self.scale_param = torch.tensor(1.0, device=device)
        
        self._initialize_weights()
    
    def _build_network(self, layers: List[int]) -> nn.ModuleList:
        """Build the neural network layers."""
        modules = nn.ModuleList()
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
        return modules
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            "tanh": torch.tanh,
            "relu": nn.ReLU(),
            "sigmoid": torch.sigmoid,
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "sin": torch.sin,
            "swish": nn.SiLU(),
        }
        return activations.get(name, torch.tanh)
    
    def _initialize_weights(self):
        """Xavier/Glorot initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional Fourier features."""
        # Apply Fourier features if enabled
        if self.use_fourier and self.fourier is not None:
            x = self.fourier(x)
        
        # Pass through network
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i < len(self.network) - 1:  # No activation after last layer
                x = self.activation(self.scale_param * x)
        return x
    
    def loss_pde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        pde_fn: Callable,
        u: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute PDE residual loss using automatic differentiation."""
        # Ensure gradients
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)
        
        # Predict solution
        u_pred = self.forward(torch.cat([x, t], dim=1))
        
        # First derivatives
        u_t = torch.autograd.grad(
            u_pred, t, grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            u_pred, x, grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]
        
        # Second derivative (for diffusion terms)
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        
        # Compute PDE residual
        residual = pde_fn(u_pred, u_t, u_x, u_xx)
        
        return torch.mean(residual ** 2)
    
    def loss_boundary(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        boundary_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary condition loss."""
        u_pred = self.forward(torch.cat([x, t], dim=1))
        return torch.mean((u_pred - boundary_values) ** 2)
    
    def loss_initial(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        initial_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute initial condition loss."""
        u_pred = self.forward(torch.cat([x, t], dim=1))
        return torch.mean((u_pred - initial_values) ** 2)


class AdaptiveLossWeights(nn.Module):
    """
    Adaptive loss weighting using gradient statistics.
    Implements Learning Rate Annealing from the original PINN paper.
    """

    def __init__(self, num_losses: int = 3, alpha: float = 0.1):
        """
        Args:
            num_losses: Number of loss components (pde, bc, ic)
            alpha: Learning rate for weight updates
        """
        super().__init__()
        self.num_losses = num_losses
        self.alpha = alpha
        self.log_weights = nn.Parameter(torch.zeros(num_losses))

    def get_weights(self) -> torch.Tensor:
        """Get current loss weights (normalized)."""
        return torch.softmax(self.log_weights, dim=0)

    def update(self, gradients: List[torch.Tensor], losses: List[torch.Tensor]):
        """
        Update weights based on gradient statistics.

        Args:
            gradients: List of gradient vectors for each loss component
            losses: List of loss values
        """
        # Compute gradient norms as a stacked tensor (not a list)
        grad_norms = torch.stack([torch.norm(g) for g in gradients])

        mean_norm = grad_norms.mean()

        # Update weights (in log space) based on relative gradient magnitudes
        with torch.no_grad():
            for i in range(self.num_losses):
                if grad_norms[i] > 0:
                    # Increase weight if this loss's gradient is large relative to the mean
                    update = self.alpha * (grad_norms[i] / (mean_norm + 1e-8))
                    self.log_weights[i] += update


class PINNTrainer:
    """
    Advanced PINN trainer with adaptive loss weighting and Fourier features.
    """
    
    def __init__(
        self,
        model: PINN,
        pde_fn: Callable,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        use_adaptive_weights: bool = True,
        scheduler_step: int = 1000,
        scheduler_gamma: float = 0.9,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.pde_fn = pde_fn
        self.device = device
        self.history = defaultdict(list)
        self.best_loss = float("inf")
        self.use_adaptive_weights = use_adaptive_weights
        
        # Adaptive loss weights
        if use_adaptive_weights:
            self.loss_weights = AdaptiveLossWeights(num_losses=3, alpha=0.1)
        else:
            self.loss_weights = None
        
        # Optimizer
        params = list(model.parameters())
        if self.loss_weights is not None:
            params += list(self.loss_weights.parameters())
        
        self.optimizer = optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_step,
            gamma=scheduler_gamma,
        )
    
    def train_step(
        self,
        x_pde: torch.Tensor,
        t_pde: torch.Tensor,
        x_bc: torch.Tensor,
        t_bc: torch.Tensor,
        bc_values: torch.Tensor,
        x_ic: torch.Tensor,
        t_ic: torch.Tensor,
        ic_values: torch.Tensor,
        weights: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """
        Single training step with adaptive loss weighting.
        """
        if weights is None:
            weights = {"pde": 1.0, "bc": 1.0, "ic": 1.0}

        self.optimizer.zero_grad()

        # Compute individual losses
        loss_pde = self.model.loss_pde(x_pde, t_pde, self.pde_fn)
        loss_bc = self.model.loss_boundary(x_bc, t_bc, bc_values)
        loss_ic = self.model.loss_initial(x_ic, t_ic, ic_values)

        # Get adaptive weights if enabled
        if self.use_adaptive_weights and self.loss_weights is not None:
            # Update weights based on gradient statistics BEFORE the main backward.
            # We compute per-loss gradients here using retain_graph=True,
            # then do the weighted total backward.
            grads = []
            for loss in [loss_pde, loss_bc, loss_ic]:
                grad = torch.autograd.grad(
                    loss,
                    self.model.parameters(),
                    retain_graph=True,
                    allow_unused=True,
                )
                # Filter out None grads and flatten
                grad_flat = torch.cat([g.flatten() for g in grad if g is not None])
                grads.append(grad_flat)

            self.loss_weights.update(grads, [loss_pde, loss_bc, loss_ic])

            w = self.loss_weights.get_weights()
            w_pde, w_bc, w_ic = w[0], w[1], w[2]
        else:
            w_pde = torch.tensor(weights["pde"], device=self.device)
            w_bc = torch.tensor(weights["bc"], device=self.device)
            w_ic = torch.tensor(weights["ic"], device=self.device)

        # Weighted total loss
        loss_total = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic

        # Backward pass
        loss_total.backward()

        self.optimizer.step()
        self.scheduler.step()

        # Track losses
        losses = {
            "loss_total": loss_total.item(),
            "loss_pde": loss_pde.item(),
            "loss_bc": loss_bc.item(),
            "loss_ic": loss_ic.item(),
        }

        if self.use_adaptive_weights and self.loss_weights is not None:
            losses["w_pde"] = w_pde.item()
            losses["w_bc"] = w_bc.item()
            losses["w_ic"] = w_ic.item()

        for key, value in losses.items():
            self.history[key].append(value)

        if loss_total.item() < self.best_loss:
            self.best_loss = loss_total.item()

        return losses
        
    def train(
        self,
        x_pde: torch.Tensor,
        t_pde: torch.Tensor,
        x_bc: torch.Tensor,
        t_bc: torch.Tensor,
        bc_values: torch.Tensor,
        x_ic: torch.Tensor,
        t_ic: torch.Tensor,
        ic_values: torch.Tensor,
        epochs: int = 10000,
        batch_size: Optional[int] = None,
        weights: Dict[str, float] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Full training loop with progress tracking."""
        n_samples = len(x_pde)
        if batch_size is None or batch_size >= n_samples:
            batch_size = n_samples
        
        if verbose:
            pbar = tqdm(range(epochs), desc="Training PINN")
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            # Mini-batch training
            indices = torch.randperm(n_samples)[:batch_size]
            
            losses = self.train_step(
                x_pde[indices], t_pde[indices],
                x_bc, t_bc, bc_values,
                x_ic, t_ic, ic_values,
                weights,
            )
            
            # Update progress bar
            if verbose and isinstance(pbar, tqdm):
                postfix = {
                    "total": f"{losses['loss_total']:.4e}",
                    "pde": f"{losses['loss_pde']:.4e}",
                }
                if self.use_adaptive_weights:
                    postfix["w"] = f"{losses.get('w_pde', 0):.2f}"
                pbar.set_postfix(postfix)
        
        return dict(self.history)
    
    def predict(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Make predictions on new points."""
        self.model.eval()
        with torch.no_grad():
            x = torch.cat([x, t], dim=1)
            return self.model(x)