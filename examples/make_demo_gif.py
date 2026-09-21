"""Generate a demo GIF showing PINN solving the heat equation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
import os

from pinnforge import AutoPINN

# Make sure the output directory exists
os.makedirs('docs', exist_ok=True)

print("Training PINN on heat equation...")
solver = AutoPINN("heat", verbose=True)
solver.config.epochs = 5000
result = solver.solve(validate=False)

print("Generating animation frames...")
n_frames = 40
x = torch.linspace(0, 1, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

def make_frame(idx):
    t_val = idx / (n_frames - 1) * 1.0
    t = torch.full_like(x, t_val)
    u_pred = result['predict'](x.numpy(), t.numpy()).squeeze().detach().cpu().numpy()
    u_true = (torch.sin(torch.pi * x) * torch.exp(torch.tensor(-torch.pi**2 * t_val))).numpy()
    return u_pred, u_true, t_val

def update(idx):
    u_pred, u_true, t_val = make_frame(idx)
    axes[0].clear()
    axes[1].clear()
    axes[0].plot(x.numpy(), u_pred, 'b-', label='PINN', lw=2)
    axes[0].plot(x.numpy(), u_true, 'r--', label='Exact', lw=2)
    axes[0].set_title(f'Heat Equation  (t = {t_val:.2f})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x,t)')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    error = np.abs(u_pred - u_true)
    axes[1].plot(x.numpy(), error, 'k-', lw=2)
    axes[1].set_title(f'Absolute Error  (max = {error.max():.2e})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('|PINN - exact|')
    axes[1].set_ylim(0, max(0.05, error.max() * 1.1))
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

print("Rendering...")
anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=150)
anim.save('docs/demo.gif', writer='pillow', fps=8)
print("Saved to docs/demo.gif")