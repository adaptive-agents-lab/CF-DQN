import torch
import math

def get_cleaned_target_cf(omega_grid, V_complex_target, q_min=0.0, q_max=100.0):
    """
    The goal of this function is pretty simple, take the V_complex_target,
    convert it back to spatial domain to get the PDF, apply the the distributional mask to remove the incoherent probability mass,
    and then convert it back to the frequency domain to get a cleaned CF target.
    """
    K = V_complex_target.shape[-1]
    W = torch.abs(omega_grid[0]).item()
    dx = math.pi / W
    q_values_gird = torch.linspace(- (K // 2) * dx, (K // 2 - 1) * dx, K, device=V_complex_target.device)
    
    # 1. Frequency -> Spatial: CF uses +i convention, so PDF recovery needs -i (= fft)
    V_shifted = torch.fft.ifftshift(V_complex_target, dim=-1)
    pdf_complex = torch.fft.fft(V_shifted, dim=-1)
    pdf = torch.clamp(torch.fft.fftshift(pdf_complex.real, dim=-1), min=0.0)
    
    # 2. Apply spatial mask to destroy the accumulated noise
    valid_mask = (q_values_gird >= q_min) & (q_values_gird <= q_max)
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
    dx = math.pi / W
    
    #* Construct the corresponding spatial grid x: [-K/2 * dx, (K/2 - 1) * dx]
    q_values_gird = torch.linspace(- (K // 2) * dx, (K // 2 - 1) * dx, K, device=V_complex.device)
    
    #* Shift frequency 0 to index 0 for standard PyTorch FFT Implementation
    #* [-W, ..., -dw, 0, dw, ..., W - dw] -> [0, ..., W - dw, -W, ..., -dw]
    #* It cuts in half the array and swaps the right side with the left side
    V_shifted = torch.fft.ifftshift(V_complex, dim=-1)
    
    #* CF→PDF: the CF uses exp(+iωx), so recovery needs exp(-iωx) = fft convention
    pdf_complex = torch.fft.fft(V_shifted, dim=-1)
    
    #* Puts back the negative frequencies on the left and positive frequencies on the right, so that x=0 is in the center of the grid
    #* [0, ..., W - dw, -W, ..., -dw] -> [-W, ..., -dw, 0, dw, ..., W - dw]
    pdf_shifted = torch.fft.fftshift(pdf_complex.real, dim=-1)
    
    pdf_unmasked = torch.clamp(pdf_shifted, min=0.0)
    pdf_unmasked_normalized = pdf_unmasked / (pdf_unmasked.sum(dim=-1, keepdim=True) + 1e-8) #! re-normalize after masking to ensure it's a valid PDF
    expected_q_value_scalar_unmasked = torch.sum(pdf_unmasked_normalized * q_values_gird, dim=-1)
    
    valid_mask = (q_values_gird >= q_min) & (q_values_gird <= q_max)
    pdf_masked = pdf_unmasked * valid_mask.float()
    pdf_masked_normalized = pdf_masked / (pdf_masked.sum(dim=-1, keepdim=True) + 1e-8) #! re-normalize after masking to ensure it's a valid PDF
    expected_q_value_scalar = torch.sum(pdf_masked_normalized * q_values_gird, dim=-1)
    
    if return_diagnostics:
        return expected_q_value_scalar, expected_q_value_scalar_unmasked, pdf_unmasked

    return expected_q_value_scalar

def unwrap_phase(phase, dim=-1):
    """
    Removes 2*pi jumps in the phase angle to create a continuous line for interpolation.
    """
    diff = torch.diff(phase, dim=dim)
    diff_wrapped = (diff + math.pi) % (2 * math.pi) - math.pi
    
    first_element = phase.narrow(dim, 0, 1)
    unwrapped = torch.cat([first_element, first_element + torch.cumsum(diff_wrapped, dim=dim)], dim=dim)
    
    return unwrapped

def polar_interpolation(omega_grid, target_V_complex, gammas):
    """
    Interpolates the target network's CF at the scaled frequencies (gamma * omega).
    Interpolates magnitude and unwrapped phase separately to prevent phase wrapping artifacts.
    """
    magnitudes = torch.abs(target_V_complex) 
    phases = torch.angle(target_V_complex) 
    unwrapped_phases = unwrap_phase(phases, dim=-1)
    
    gamma_target_omega = gammas * omega_grid.view(1, -1) 
    
    idx_right = torch.searchsorted(omega_grid, gamma_target_omega) 
    idx_right = torch.clamp(idx_right, 1, len(omega_grid) - 1) 
    idx_left = idx_right - 1
    
    omega_left = omega_grid[idx_left]
    omega_right = omega_grid[idx_right]
    
    t = (gamma_target_omega - omega_left) / (omega_right - omega_left + 1e-8)
    
    batch_idx = torch.arange(target_V_complex.shape[0], device=target_V_complex.device).unsqueeze(1)
    
    #* Linearly interpolate magnitude
    mag_left = magnitudes[batch_idx, idx_left]
    mag_right = magnitudes[batch_idx, idx_right]
    interp_mag = (1 - t) * mag_left + t * mag_right
    
    #* Linearly interpolate the unwrapped phase
    phase_left = unwrapped_phases[batch_idx, idx_left]
    phase_right = unwrapped_phases[batch_idx, idx_right]
    interp_phase = phase_left + t * (phase_right - phase_left)
    
    return interp_mag * torch.complex(torch.cos(interp_phase), torch.sin(interp_phase))

def calculate_optimal_cf_grid_params(q_min: float, q_max: float, desired_resolution=1.0, safety_factor=2.0):
    """
    Calculates the mathematically optimal W and K for Distributional RL in the frequency domain.
    
    Args:
        q_min: Minimum possible return in the environment.
        q_max: Maximum possible return in the environment.
        desired_resolution: The spatial bin width (dx). 1.0 means bins at [..., 99, 100, 101, ...].
        safety_factor: Multiplier to prevent phase wrapping from neural network noise (Gibbs ringing).
                       2.0 means the grid can theoretically support 2x your max Q-value before aliasing.
    
    Returns:
        W (float): Maximum frequency.
        K (int): Number of frequency points (guaranteed to be even).
    """
    # 1. Find the absolute furthest point from 0 the grid needs to support
    q_bound = max(abs(q_min), abs(q_max))
    
    # 2. Lock down the spatial resolution
    W = math.pi / desired_resolution
    
    # 3. Calculate the minimum K to safely cover Q_bound without phase wrapping
    min_k_required = (2.0 * W * q_bound) / math.pi
    
    # 4. Apply the safety margin to absorb noise
    target_k = min_k_required * safety_factor
    
    # 5. Round up to the nearest even integer (required for FFT symmetry)
    K = int(math.ceil(target_k))
    if K % 2 != 0:
        K += 1
        
    return W, K

