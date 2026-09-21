"""
Example: Solving the Heat equation with PINNForge.
"""
import torch
import matplotlib.pyplot as plt
from pinnforge import PINN, PINNTrainer, HeatEquation
from pinnforge.utils.data import (
    create_collocation_points,
    create_boundary_points,
    create_initial_points,
)
from pinnforge.utils.logging import ExperimentLogger


def run_heat_example():
    """Run Heat equation example."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define PDE
    pde = HeatEquation(alpha=1.0)
    
    # Create PINN model
    model = PINN(
        layers=[2, 50, 50, 50, 1],
        activation="tanh",
        use_adaptive_activation=True,
        device=device,
    )
    
    # Generate collocation points
    x_pde, t_pde = create_collocation_points(
        x_range=(0, 1),
        t_range=(0, 1),
        n_points=10000,
        device=device,
    )
    
    # Generate boundary points
    x_bc, t_bc = create_boundary_points(
        x_range=(0, 1),
        t_range=(0, 1),
        n_points=200,
        device=device,
    )
    bc_values = pde.boundary_condition(x_bc, t_bc)
    
    # Generate initial points
    x_ic, t_ic = create_initial_points(
        x_range=(0, 1),
        n_points=200,
        device=device,
    )
    ic_values = pde.initial_condition(x_ic)
    
    # Create trainer
    trainer = PINNTrainer(
        model=model,
        pde_fn=pde.residual,
        learning_rate=1e-3,
        weight_decay=1e-6,
        device=device,
    )
    
    # Setup logging
    logger = ExperimentLogger(
        log_dir="logs",
        experiment_name="heat_pinn",
        save_model=True,
    )
    logger.log_config({
        "pde": "Heat",
        "alpha": pde.alpha,
        "layers": [2, 50, 50, 50, 1],
        "activation": "tanh",
        "learning_rate": 1e-3,
        "epochs": 3000,
    })
    
    # Train
    print("Training PINN...")
    history = trainer.train(
        x_pde=x_pde,
        t_pde=t_pde,
        x_bc=x_bc,
        t_bc=t_bc,
        bc_values=bc_values,
        x_ic=x_ic,
        t_ic=t_ic,
        ic_values=ic_values,
        epochs=3000,
        batch_size=1000,
        weights={"pde": 1.0, "bc": 10.0, "ic": 10.0},
        verbose=True,
    )
    
    # Generate predictions
    x_test = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
    t_test = torch.linspace(0, 1, 50, device=device).reshape(-1, 1)
    
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze())
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)
    
    with torch.no_grad():
        u_pred = trainer.predict(X_flat, T_flat)
        u_pred = u_pred.reshape(X.shape)
    
    # Analytical solution for comparison
    u_analytical = pde.analytical_solution(X_flat, T_flat).reshape(X.shape)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = axes[0].contourf(
        X.cpu().numpy(),
        T.cpu().numpy(),
        u_pred.cpu().numpy(),
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title("PINN Solution")
    
    im2 = axes[1].contourf(
        X.cpu().numpy(),
        T.cpu().numpy(),
        u_analytical.cpu().numpy(),
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("Analytical Solution")
    
    plt.tight_layout()
    plt.savefig("heat_solution.png", dpi=150)
    plt.show()
    
    # Compute error
    error = torch.mean((u_pred - u_analytical) ** 2)
    print(f"Mean Squared Error: {error.item():.6e}")
    
    print("Heat example completed!")
    return model, trainer, history


if __name__ == "__main__":
    run_heat_example()