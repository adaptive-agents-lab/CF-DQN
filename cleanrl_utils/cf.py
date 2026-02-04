# cleanrl_utils/cf.py
# Characteristic Function utilities for CF-DQN
# Ported from cvi_rl/cf/ with PyTorch tensor support and batched operations

from __future__ import annotations

from typing import List

import torch
import numpy as np


# -----------------------------------------------------------------------------
# Frequency Grid Construction
# -----------------------------------------------------------------------------

def make_omega_grid(
    W: float,
    K: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a frequency grid using the three_density_regions strategy.
    
    Allocates points with varying density:
    - Center (50% of points in inner 10% of range): dense core for accurate mean extraction
    - Middle (30% of points in next 30% of range): medium density transition
    - Tails (20% of points in outer 60% of range): sparse coverage of high frequencies
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency (grid spans [-W, W]).
    K : int
        Total number of grid points.
    device : torch.device, optional
        Device to place the tensor on.
    
    Returns
    -------
    omegas : torch.Tensor
        1D tensor of shape [K], with three density regions in [-W, W].
    """
    # Three density regions configuration
    boundaries = [0.1, 0.4]  # Boundaries at 0.1W and 0.4W
    fractions = [0.5, 0.3, 0.2]  # 50% center, 30% middle, 20% tails
    
    n_regions = len(fractions)
    
    # Convert relative boundaries to absolute values
    abs_boundaries = [W * b for b in boundaries]
    
    # Build region edges: [0, w1, w2, ..., W]
    region_edges = [0.0] + abs_boundaries + [W]
    
    # Allocate points to each region
    K_per_region = []
    K_remaining = K
    for i, frac in enumerate(fractions[:-1]):
        k = max(2 if i > 0 else 3, int(frac * K))  # min 3 for center, 2 for others
        K_per_region.append(k)
        K_remaining -= k
    K_per_region.append(max(2, K_remaining))  # Last region gets remainder
    
    # Build grid segments for each region
    segments = []
    
    for i in range(n_regions):
        inner_edge = region_edges[i]
        outer_edge = region_edges[i + 1]
        k_region = K_per_region[i]
        
        if i == 0:
            # Center region: symmetric around 0
            center = np.linspace(-outer_edge, outer_edge, k_region)
            segments.append(center)
        else:
            # Non-center regions: split into left and right halves
            k_half = k_region // 2
            
            # Left side: [-outer_edge, -inner_edge) - exclude inner_edge (covered by center)
            left = np.linspace(-outer_edge, -inner_edge, k_half, endpoint=False)
            
            # Right side: (inner_edge, outer_edge] - exclude inner_edge (covered by center)
            if i == n_regions - 1:
                # Last region includes the endpoint W
                right = np.linspace(inner_edge, outer_edge, k_region - k_half + 1, endpoint=True)[1:]  # Skip first to avoid duplicate
            else:
                right = np.linspace(inner_edge, outer_edge, k_region - k_half + 1, endpoint=False)[1:]  # Skip first to avoid duplicate
            
            segments.insert(0, left)  # Add left to beginning
            segments.append(right)    # Add right to end
    
    # Combine all segments
    omegas = np.concatenate(segments)
    
    # Sort and remove any remaining duplicates
    omegas = np.sort(omegas)
    omegas = np.unique(omegas)
    
    # Ensure exactly K points by adding/removing points
    while len(omegas) < K:
        # Add points in the largest gaps
        gaps = np.diff(omegas)
        max_gap_idx = np.argmax(gaps)
        insert_point = (omegas[max_gap_idx] + omegas[max_gap_idx + 1]) / 2
        omegas = np.sort(np.append(omegas, insert_point))
    
    if len(omegas) > K:
        # Remove points from densest regions (smallest gaps)
        while len(omegas) > K:
            gaps = np.diff(omegas)
            min_gap_idx = np.argmin(gaps)
            # Remove the point that creates the smallest gap (keep one of the pair)
            omegas = np.delete(omegas, min_gap_idx + 1)
    
    # Convert to torch tensor
    omegas_tensor = torch.from_numpy(omegas.copy()).float()
    
    if device is not None:
        omegas_tensor = omegas_tensor.to(device)
    
    return omegas_tensor


# -----------------------------------------------------------------------------
# Phase Unwrapping (PyTorch implementation)
# -----------------------------------------------------------------------------

def unwrap_phase(phase: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Unwrap phase angles by changing absolute jumps greater than pi to their 
    2*pi complement along the given dimension.
    
    Parameters
    ----------
    phase : torch.Tensor
        Phase angles in radians.
    dim : int
        Dimension along which to unwrap.
    
    Returns
    -------
    unwrapped : torch.Tensor
        Unwrapped phase angles.
    """
    # Compute differences along the dimension
    diff = torch.diff(phase, dim=dim)
    
    # Wrap differences to [-pi, pi]
    diff_wrapped = (diff + torch.pi) % (2 * torch.pi) - torch.pi
    
    # Handle the case where diff is exactly -pi (should stay as pi)
    diff_wrapped = torch.where(
        (diff_wrapped == -torch.pi) & (diff > 0),
        torch.tensor(torch.pi, device=phase.device, dtype=phase.dtype),
        diff_wrapped
    )
    
    # Compute correction
    correction = diff_wrapped - diff
    
    # Cumulative sum of corrections
    correction_cumsum = torch.cumsum(correction, dim=dim)
    
    # Prepend zeros to match original shape
    pad_shape = list(phase.shape)
    pad_shape[dim] = 1
    zeros = torch.zeros(pad_shape, device=phase.device, dtype=phase.dtype)
    correction_cumsum = torch.cat([zeros, correction_cumsum], dim=dim)
    
    return phase + correction_cumsum


# -----------------------------------------------------------------------------
# CF Interpolation (Polar Method)
# -----------------------------------------------------------------------------

def interpolate_cf_polar(
    target_omegas: torch.Tensor,
    grid_omegas: torch.Tensor,
    cf: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate characteristic function at target frequencies using polar method.
    
    Interpolates magnitude and unwrapped phase separately, then reconstructs.
    This preserves the CF validity constraint |φ(ω)| ≤ 1 better than Cartesian.
    
    Parameters
    ----------
    target_omegas : torch.Tensor
        Target frequencies to interpolate at, shape [K] or [K_target].
    grid_omegas : torch.Tensor
        Original frequency grid, shape [K].
    cf : torch.Tensor
        Characteristic function values at grid_omegas.
        Shape: [K], [batch, K], [batch, n_actions, K], or [n_actions, K]
    
    Returns
    -------
    cf_interp : torch.Tensor
        Interpolated CF at target frequencies, same batch dimensions as input.
    """
    # Ensure inputs are on same device
    device = cf.device
    target_omegas = target_omegas.to(device)
    grid_omegas = grid_omegas.to(device)
    
    # Handle different input dimensions
    original_shape = cf.shape
    K = grid_omegas.shape[0]
    K_target = target_omegas.shape[0]
    
    # Flatten batch dimensions if present
    if cf.dim() == 1:
        # [K] -> [1, K]
        cf_flat = cf.unsqueeze(0)
    elif cf.dim() == 2:
        # [batch, K] -> [batch, K]
        cf_flat = cf
    elif cf.dim() == 3:
        # [batch, n_actions, K] -> [batch * n_actions, K]
        batch, n_actions, _ = cf.shape
        cf_flat = cf.view(batch * n_actions, K)
    else:
        raise ValueError(f"Unsupported CF shape: {cf.shape}")
    
    # Compute magnitude and phase
    magnitude = torch.abs(cf_flat)  # [batch_flat, K]
    phase = torch.angle(cf_flat)    # [batch_flat, K]
    
    # Unwrap phase along frequency dimension
    phase_unwrapped = unwrap_phase(phase, dim=-1)  # [batch_flat, K]
    
    # Linear interpolation using searchsorted
    # Find indices for interpolation
    # Clamp target_omegas to grid range for extrapolation handling
    target_clamped = torch.clamp(target_omegas, grid_omegas[0], grid_omegas[-1])
    
    # Find right indices
    indices_right = torch.searchsorted(grid_omegas, target_clamped)
    indices_right = torch.clamp(indices_right, 1, K - 1)
    indices_left = indices_right - 1
    
    # Get grid values at left and right indices
    omega_left = grid_omegas[indices_left]   # [K_target]
    omega_right = grid_omegas[indices_right] # [K_target]
    
    # Compute interpolation weights
    denom = omega_right - omega_left
    denom = torch.where(denom.abs() < 1e-10, torch.ones_like(denom), denom)
    t = (target_clamped - omega_left) / denom  # [K_target]
    t = torch.clamp(t, 0.0, 1.0)
    
    # Interpolate magnitude
    mag_left = magnitude[:, indices_left]    # [batch_flat, K_target]
    mag_right = magnitude[:, indices_right]  # [batch_flat, K_target]
    mag_interp = mag_left + t.unsqueeze(0) * (mag_right - mag_left)
    
    # Interpolate phase
    phase_left = phase_unwrapped[:, indices_left]    # [batch_flat, K_target]
    phase_right = phase_unwrapped[:, indices_right]  # [batch_flat, K_target]
    phase_interp = phase_left + t.unsqueeze(0) * (phase_right - phase_left)
    
    # Reconstruct complex CF
    cf_interp = mag_interp * torch.exp(1j * phase_interp)
    
    # Handle extrapolation (use boundary values)
    # For targets outside grid, use boundary CF values
    below_grid = target_omegas < grid_omegas[0]
    above_grid = target_omegas > grid_omegas[-1]
    
    if below_grid.any():
        cf_interp[:, below_grid] = cf_flat[:, 0:1].expand(-1, below_grid.sum())
    if above_grid.any():
        cf_interp[:, above_grid] = cf_flat[:, -1:].expand(-1, above_grid.sum())
    
    # Reshape back to original batch dimensions
    if len(original_shape) == 1:
        return cf_interp.squeeze(0)
    elif len(original_shape) == 2:
        return cf_interp
    elif len(original_shape) == 3:
        batch, n_actions, _ = original_shape
        return cf_interp.view(batch, n_actions, K_target)
    
    return cf_interp


# -----------------------------------------------------------------------------
# CF Collapse (Gaussian Method)
# -----------------------------------------------------------------------------

def collapse_cf_to_mean( #! THIS IS WHAT CAUSES THE INSTABILITY PROBLEM
    omegas: torch.Tensor,
    cf: torch.Tensor,
    max_w: float = 2.0,
) -> torch.Tensor:
    """
    Extract the mean E[G] from characteristic function using the Gaussian method.
    
    Assumes locally Gaussian CF: log φ(ω) ≈ iμω - 0.5σ²ω²
    Fits a line to the unwrapped phase: phase(φ(ω)) ≈ μω
    
    Parameters
    ----------
    omegas : torch.Tensor
        Frequency grid, shape [K].
    cf : torch.Tensor
        Characteristic function values.
        Shape: [K], [batch, K], [batch, n_actions, K], or [n_actions, K]
    max_w : float
        Maximum |ω| to use for fitting. Focus on low frequencies where signal is strong.
    
    Returns
    -------
    mean : torch.Tensor
        Estimated mean(s). Shape depends on input:
        - [K] -> scalar
        - [batch, K] -> [batch]
        - [batch, n_actions, K] -> [batch, n_actions]
        - [n_actions, K] -> [n_actions]
    """
    device = cf.device
    omegas = omegas.to(device)
    
    # Handle different input dimensions
    original_shape = cf.shape
    K = omegas.shape[0]
    
    # Flatten batch dimensions if present
    if cf.dim() == 1:
        # [K] -> [1, K]
        cf_flat = cf.unsqueeze(0)
        output_shape = ()
    elif cf.dim() == 2:
        # [batch, K] or [n_actions, K] -> [batch, K]
        cf_flat = cf
        output_shape = (cf.shape[0],)
    elif cf.dim() == 3:
        # [batch, n_actions, K] -> [batch * n_actions, K]
        batch, n_actions, _ = cf.shape
        cf_flat = cf.view(batch * n_actions, K)
        output_shape = (batch, n_actions)
    else:
        raise ValueError(f"Unsupported CF shape: {cf.shape}")
    
    # Create mask for frequencies within [-max_w, max_w]
    mask = torch.abs(omegas) <= max_w
    
    if mask.sum() < 2:
        # Fallback to all frequencies
        mask = torch.ones(K, dtype=torch.bool, device=device)
    
    # Extract subset
    w_sub = omegas[mask]  # [K_sub]
    cf_sub = cf_flat[:, mask]  # [batch_flat, K_sub]
    
    # Compute unwrapped phase
    phase = torch.angle(cf_sub)
    phase_unwrapped = unwrap_phase(phase, dim=-1)  # [batch_flat, K_sub]
    
    # Linear regression through origin: phase ≈ μ * ω
    # μ = sum(ω * phase) / sum(ω²)
    # Vectorized for batch dimension
    
    w_sub_expanded = w_sub.unsqueeze(0)  # [1, K_sub]
    
    numerator = (w_sub_expanded * phase_unwrapped).sum(dim=-1)  # [batch_flat]
    denominator = (w_sub_expanded ** 2).sum(dim=-1)  # [1] -> broadcast to [batch_flat]
    
    # Avoid division by zero
    denominator = torch.clamp(denominator, min=1e-12)
    
    mu = numerator / denominator  # [batch_flat]
    
    # Reshape to original batch dimensions
    if len(output_shape) == 0:
        return mu.squeeze(0)
    elif len(output_shape) == 1:
        return mu
    elif len(output_shape) == 2:
        return mu.view(output_shape)
    
    return mu

def complex_huber_loss(pred, target, delta=1.0):
    """
    Huber loss for complex-valued predictions.
    Less sensitive to outliers than MSE.
    
    Args:
        pred: predicted CF, shape [..., K]
        target: target CF, shape [..., K]
        delta: Huber loss threshold
    
    Returns:
        scalar loss
    """
    # Compute element-wise distance
    diff = pred - target
    abs_diff = torch.abs(diff)
    
    # Huber loss: quadratic below delta, linear above
    quadratic = 0.5 * (abs_diff ** 2)
    linear = delta * (abs_diff - 0.5 * delta)
    loss_per_element = torch.where(abs_diff <= delta, quadratic, linear)
    
    return torch.mean(loss_per_element)


# -----------------------------------------------------------------------------
# Reward CF
# -----------------------------------------------------------------------------

def reward_cf(
    omegas: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the characteristic function of immediate rewards: exp(iωr).
    
    Parameters
    ----------
    omegas : torch.Tensor
        Frequency grid, shape [K].
    rewards : torch.Tensor
        Immediate rewards, shape [batch] or [batch, 1].
    
    Returns
    -------
    cf_r : torch.Tensor
        Reward characteristic function, shape [batch, K].
    """
    device = rewards.device
    omegas = omegas.to(device)
    
    # Ensure rewards is [batch, 1]
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)  # [batch, 1]
    
    # Ensure rewards is [batch, 1] not [batch, n] where n > 1
    if rewards.shape[-1] != 1:
        rewards = rewards.unsqueeze(-1)
    
    # omegas: [K] -> [1, K]
    omegas_expanded = omegas.unsqueeze(0)  # [1, K]
    
    # Compute exp(i * omega * reward)
    # rewards: [batch, 1], omegas: [1, K] -> [batch, K]
    cf_r = torch.exp(1j * omegas_expanded * rewards)
    
    return cf_r


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def complex_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error for complex tensors.
    
    |z|² = Re(z)² + Im(z)²
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted complex tensor.
    target : torch.Tensor
        Target complex tensor.
    
    Returns
    -------
    loss : torch.Tensor
        Scalar loss value.
    """
    diff = pred - target
    return (diff.real ** 2 + diff.imag ** 2).mean()

