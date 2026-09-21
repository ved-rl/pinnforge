"""
Example: Solving Burgers' equation with PINNForge.
"""
import torch
import matplotlib.pyplot as plt
from pinnforge import PINN, PINNTrainer, BurgersEquation
from pinnforge.utils.data import (
    create_collocation_points,
    create_boundary_points,
    create_initial_points,
)
from pinnforge.utils.logging import ExperimentLogger


def run_burgers_example():
    """Run Burgers' equation example."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define PDE
    pde = BurgersEquation(nu=0.01)
    
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
        experiment_name="burgers_pinn",
        save_model=True,
    )
    logger.log_config({
        "pde": "Burgers",
        "nu": pde.nu,
        "layers": [2, 50, 50, 50, 1],
        "activation": "tanh",
        "learning_rate": 1e-3,
        "epochs": 5000,
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
        epochs=5000,
        batch_size=1000,
        weights={"pde": 1.0, "bc": 10.0, "ic": 10.0},
        verbose=True,
    )
    
    # Log metrics
    for i, loss_total in enumerate(history["loss_total"]):
        logger.log_metrics({
            "loss_total": loss_total,
            "loss_pde": history["loss_pde"][i],
            "loss_bc": history["loss_bc"][i],
            "loss_ic": history["loss_ic"][i],
        }, step=i)
    
    # Save checkpoint
    logger.save_checkpoint(model, trainer.optimizer, 5000)
    
    # Generate predictions
    x_test = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
    t_test = torch.linspace(0, 1, 50, device=device).reshape(-1, 1)
    
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze())
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)
    
    with torch.no_grad():
        u_pred = trainer.predict(X_flat, T_flat)
        u_pred = u_pred.reshape(X.shape)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.contourf(
        X.cpu().numpy(),
        T.cpu().numpy(),
        u_pred.cpu().numpy(),
        levels=50,
        cmap="viridis",
    )
    plt.colorbar(im)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("PINN Solution of Burgers' Equation")
    plt.savefig("burgers_solution.png", dpi=150)
    plt.show()
    
    print("Burgers example completed!")
    return model, trainer, history


if __name__ == "__main__":
    run_burgers_example()