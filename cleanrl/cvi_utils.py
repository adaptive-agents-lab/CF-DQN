import os
import torch
import math

def get_cleaned_target_cf(omega_grid, V_complex_target, q_min=0.0, q_max=100.0):
    """
    Cleans the target CF by projecting it onto a valid distribution with support in [q_min, q_max].
    Uses the EXACT reverse PyTorch DFT operations to prevent aliasing.
    Requires a uniform frequency grid.
    """
    K = V_complex_target.shape[-1]
    W = torch.abs(omega_grid[0]).item()
    dx = math.pi / W
    x_grid = torch.linspace(- (K // 2) * dx, (K // 2 - 1) * dx, K, device=V_complex_target.device)
    
    # 1. Frequency -> Spatial: CF uses +i convention, so PDF recovery needs -i (= fft)
    V_shifted = torch.fft.ifftshift(V_complex_target, dim=-1)
    pdf_complex = torch.fft.fft(V_shifted, dim=-1)
    pdf = torch.clamp(torch.fft.fftshift(pdf_complex.real, dim=-1), min=0.0)
    
    # 2. Apply spatial mask to destroy the accumulated noise
    valid_mask = (x_grid >= q_min) & (x_grid <= q_max)
    pdf_clean = pdf * valid_mask.float()
    
    # 3. Normalize back to a valid probability distribution
    pdf_clean = pdf_clean / (pdf_clean.sum(dim=-1, keepdim=True) + 1e-8)
    
    # 4. Spatial -> Frequency: CF uses +i convention (= K * ifft)
    pdf_unshifted = torch.fft.ifftshift(pdf_clean.to(torch.complex64), dim=-1)
    V_clean_shifted = K * torch.fft.ifft(pdf_unshifted, dim=-1)
    cleaned_cf = torch.fft.fftshift(V_clean_shifted, dim=-1)
    
    return cleaned_cf

def create_uniform_grid(K: int, W: float, device="cpu"):
    """
    Constructs a uniform frequency grid required for Fast Fourier Transforms.
    Generates K points perfectly symmetric around 0, spanning [-W, W - dw].
    """
    assert K % 2 == 0, "K must be even for FFT symmetry."
    dw = 2.0 * W / K
    # Standard FFT grid: [-W, W)
    grid = torch.linspace(-W, W - dw, K, device=device)
    return grid

def ifft_collapse_q_values(omega_grid, V_complex, q_min=0.0, q_max=100.0, return_diagnostics=False):
    """
    Extracts Q-values by transforming the CF back into a Probability Density Function (PDF)
    using the Inverse Fast Fourier Transform, then computing the expected value E[x].
    Requires a uniform frequency grid.
    """
    K = V_complex.shape[-1]
    W = torch.abs(omega_grid[0]).item()
    
    # Mathematical relationship between frequency resolution and spatial resolution
    dx = math.pi / W
    
    # Construct the corresponding spatial grid x: [-K/2 * dx, (K/2 - 1) * dx]
    x_grid = torch.linspace(- (K // 2) * dx, (K // 2 - 1) * dx, K, device=V_complex.device)
    
    # Shift frequency 0 to index 0 for standard PyTorch FFT
    V_shifted = torch.fft.ifftshift(V_complex, dim=-1)
    
    # CF→PDF: the CF uses exp(+iωx), so recovery needs exp(-iωx) = fft convention
    pdf_complex = torch.fft.fft(V_shifted, dim=-1)
    
    # Shift the spatial zero back to the center and take the real component
    pdf_shifted = torch.fft.fftshift(pdf_complex.real, dim=-1)
    
    # 1. Unmasked Q-value (What the network ACTUALLY predicts)
    pdf_unmasked = torch.clamp(pdf_shifted, min=0.0)
    pdf_unmasked = pdf_unmasked / (pdf_unmasked.sum(dim=-1, keepdim=True) + 1e-8)
    q_unmasked = torch.sum(pdf_unmasked * x_grid, dim=-1)
    
    # 2. Masked Q-value (What the agent uses to act)
    valid_mask = (x_grid >= q_min) & (x_grid <= q_max)
    pdf_masked = pdf_unmasked * valid_mask.float()
    pdf_masked = pdf_masked / (pdf_masked.sum(dim=-1, keepdim=True) + 1e-8)
    q_masked = torch.sum(pdf_masked * x_grid, dim=-1)
    
    if return_diagnostics:
        return q_masked, q_unmasked, pdf_unmasked
    return q_masked

def create_three_density_grid(K: int, W: float, device="cpu"):
    """
    Constructs the Three-Density Region frequency grid.
    Allocates 50% of points to inner 10%, 30% to middle, and 20% to tails.
    
    Args:
        K: Total number of grid points (must be even for symmetry).
        W: Maximum frequency range [-W, W].
        device: PyTorch device.
        
    Returns:
        1D Tensor of shape (K + 1,) representing the frequency grid centered at 0.0.
    """
    assert K % 2 == 0, "K must be even to maintain perfect symmetry around 0."
    half_k = K // 2
    
    n_inner = int(half_k * 0.50)
    n_mid = int(half_k * 0.30)
    n_tail = half_k - n_inner - n_mid
    
    inner_bound = 0.1 * W
    mid_bound = 0.5 * W 
    
    omega_min = 1e-3 * W
    inner = torch.linspace(omega_min, inner_bound, n_inner, device=device)
    mid = torch.linspace(inner_bound + 1e-4, mid_bound, n_mid, device=device)
    tail = torch.linspace(mid_bound + 1e-4, W, n_tail, device=device)
    
    pos_grid = torch.cat([inner, mid, tail])
    
    # Mirror for the negative half, and explicitly insert 0.0 in the exact center
    neg_grid = -torch.flip(pos_grid, dims=[0])
    zero_point = torch.tensor([0.0], device=device)
    
    grid = torch.cat([neg_grid, zero_point, pos_grid])
    return grid

def unwrap_phase(phase, dim=-1):
    """
    ! PyTorch equivalent of numpy.unwrap.
    Removes 2*pi jumps in the phase angle to create a continuous line for interpolation.
    """
    diff = torch.diff(phase, dim=dim)
    diff_wrapped = (diff + math.pi) % (2 * math.pi) - math.pi
    
    first_element = phase.narrow(dim, 0, 1)
    unwrapped = torch.cat([first_element, first_element + torch.cumsum(diff_wrapped, dim=dim)], dim=dim)
    
    #TODO: to plot and see if the unwrapping is correct.
    #_plot_unwrap_phase_debug(phase, unwrapped, dim)
    return unwrapped

def polar_interpolation(omega_grid, target_V_complex, gammas):
    """
    Interpolates the target network's CF at the scaled frequencies (gamma * omega).
    Interpolates magnitude and phase separately to preserve the distribution's variance,
    using circular shortest-path math to prevent phase wrapping artifacts.
    """
    magnitudes = torch.abs(target_V_complex) 
    phases = torch.angle(target_V_complex) 
    
    gamma_target_omega = gammas * omega_grid.view(1, -1) 
    
    idx_right = torch.searchsorted(omega_grid, gamma_target_omega) 
    idx_right = torch.clamp(idx_right, 1, len(omega_grid) - 1) 
    idx_left = idx_right - 1
    
    omega_left = omega_grid[idx_left]
    omega_right = omega_grid[idx_right]
    
    t = (gamma_target_omega - omega_left) / (omega_right - omega_left + 1e-8)
    
    batch_idx = torch.arange(target_V_complex.shape[0], device=target_V_complex.device).unsqueeze(1)
    
    # 1. Linearly interpolate magnitude (preserves distribution structure)
    mag_left = magnitudes[batch_idx, idx_left]
    mag_right = magnitudes[batch_idx, idx_right]
    interp_mag = (1 - t) * mag_left + t * mag_right
    
    # 2. Circularly interpolate phase (shortest path on the unit circle)
    phase_left = phases[batch_idx, idx_left]
    phase_right = phases[batch_idx, idx_right]
    
    phase_diff = phase_right - phase_left
    phase_diff = (phase_diff + math.pi) % (2 * math.pi) - math.pi  # Wrap to [-pi, pi]
    
    interp_phase = phase_left + t * phase_diff
    
    return interp_mag * torch.complex(torch.cos(interp_phase), torch.sin(interp_phase))

def gaussian_collapse_q_values(omega_grid, V_complex, n_pairs=5):
    """
    Extracts Q-values from the CF via symmetric finite differences at ω→0.
    Uses circular shortest-path math to ensure the slope is calculated perfectly
    even if the phase wraps across the -pi/pi boundary.
    """
    zero_idx = len(omega_grid) // 2          
    k = torch.arange(1, n_pairs + 1, device=omega_grid.device)
    pos_idx = zero_idx + k                   
    neg_idx = zero_idx - k                   

    omega_pos = omega_grid[pos_idx]          
    phase = torch.angle(V_complex)          

    phase_pos = phase[..., pos_idx]          
    phase_neg = phase[..., neg_idx]          

    # Safe circular difference: calculates the true angular distance across the boundary
    phase_diff = phase_pos - phase_neg
    phase_diff = (phase_diff + math.pi) % (2 * math.pi) - math.pi

    # Per-pair slope
    slopes = phase_diff / (2.0 * omega_pos)   

    return slopes.mean(dim=-1)


def safe_collapse_q_values(omega_grid, V_complex, q_max_hint=600.0):
    """
    Adaptively extracts Q-values using only phase-safe frequency pairs.
    
    For each symmetric pair k, the constraint is: 2 * omega_k * |Q| < pi.
    This function automatically selects the maximum number of usable pairs
    given q_max_hint, which should be an upper bound on expected Q-values.
    
    Falls back to pair-1-only if no multi-pair configuration is safe.
    """
    zero_idx = len(omega_grid) // 2
    
    # Determine how many pairs are safe: 2 * omega_k * q_max_hint < pi
    max_omega = math.pi / (2.0 * q_max_hint)
    
    # Find how many pairs from center satisfy the constraint
    n_safe = 0
    for k in range(1, zero_idx):
        if omega_grid[zero_idx + k].item() < max_omega:
            n_safe = k
        else:
            break
    n_safe = max(n_safe, 1)  # Always use at least pair 1
    
    return gaussian_collapse_q_values(omega_grid, V_complex, n_pairs=n_safe)

def _plot_unwrap_phase_debug(phase_tensor: torch.Tensor, unwrapped_tensor: torch.Tensor, dim: int) -> None:
    import matplotlib.pyplot as plt

    with torch.no_grad():
        t_cpu = phase_tensor.detach().to("cpu")
        u_cpu = unwrapped_tensor.detach().to("cpu")
        if t_cpu.ndim == 0:
            wrapped_curve = t_cpu.unsqueeze(0).numpy()
            unwrapped_curve = u_cpu.unsqueeze(0).numpy()
            axis = 0
        else:
            axis = dim % t_cpu.ndim
            wrapped_curve = t_cpu.moveaxis(axis, -1).reshape(-1, t_cpu.shape[axis])[0].numpy()
            unwrapped_curve = u_cpu.moveaxis(axis, -1).reshape(-1, u_cpu.shape[axis])[0].numpy()

    x = range(len(wrapped_curve))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, wrapped_curve, label="wrapped phase", linewidth=1.2, alpha=0.7)
    ax.plot(x, unwrapped_curve, label="unwrapped phase", linewidth=1.2)
    ax.set_title(f"unwrap_phase debug (dim={axis})")
    ax.set_xlabel("index")
    ax.set_ylabel("phase (rad)")
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw_idle()
    plt.show(block=False)
    plt.pause(0.001)
    plt.close(fig)
