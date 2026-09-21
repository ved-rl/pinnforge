"""
Benchmarking metrics for PINN evaluation.
Pure-PyTorch implementation, no external ML dependencies.
"""
import torch
from typing import Dict, List


def compute_relative_l2_error(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
) -> float:
    """
    Compute relative L2 error.

    Args:
        u_pred: Predicted solution
        u_true: Ground truth solution

    Returns:
        Relative L2 error
    """
    u_pred = u_pred.flatten()
    u_true = u_true.flatten()

    numerator = torch.norm(u_pred - u_true, p=2)
    denominator = torch.norm(u_true, p=2)

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


def compute_metrics(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute comprehensive set of metrics.

    Returns:
        Dictionary with: relative_l2_error, mse, rmse, r2_score, max_error, mean_error
    """
    u_pred = u_pred.flatten().detach()
    u_true = u_true.flatten().detach()

    # Mean squared error
    mse = torch.mean((u_pred - u_true) ** 2).item()

    # Root mean squared error
    rmse = mse ** 0.5

    # R^2 score: 1 - SS_res / SS_tot
    ss_res = torch.sum((u_true - u_pred) ** 2)
    ss_tot = torch.sum((u_true - u_true.mean()) ** 2)
    if ss_tot.item() == 0:
        r2 = 0.0
    else:
        r2 = (1.0 - ss_res / ss_tot).item()

    return {
        "relative_l2_error": compute_relative_l2_error(u_pred, u_true),
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2,
        "max_error": torch.max(torch.abs(u_pred - u_true)).item(),
        "mean_error": torch.mean(torch.abs(u_pred - u_true)).item(),
    }


def compute_convergence_rate(
    errors: List[float],
    iterations: List[int],
) -> float:
    """
    Estimate convergence rate from error history.

    Args:
        errors: List of errors over iterations
        iterations: Corresponding iteration numbers

    Returns:
        Convergence rate (higher is better)
    """
    if len(errors) < 2:
        return 0.0

    import math

    # Fit power law: error = C * iter^(-rate)
    log_errors = [math.log(e) for e in errors]
    log_iters = [math.log(i) for i in iterations]

    n = len(log_errors)
    mean_x = sum(log_iters) / n
    mean_y = sum(log_errors) / n

    num = sum((log_iters[i] - mean_x) * (log_errors[i] - mean_y) for i in range(n))
    den = sum((log_iters[i] - mean_x) ** 2 for i in range(n))

    if den == 0:
        return 0.0

    slope = num / den
    return -slope