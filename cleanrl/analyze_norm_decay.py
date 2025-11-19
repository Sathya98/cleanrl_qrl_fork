"""
Analyze why norm decays in URNN even without input embedding.
"""
import torch
import numpy as np
from urnn import URNN, ModReLU, complex_unit_norm
import torch.nn.functional as F

print("=" * 60)
print("Analyzing Norm Decay in URNN")
print("=" * 60)

# The issue: ModReLU activation is NOT norm-preserving!
print("\n1. MODRELU ACTIVATION:")
print("-" * 60)

# Create a test hidden state
hidden_size = 128
batch_size = 4
norm_scale = np.sqrt(128)

# Initial hidden state
h_init = np.sqrt(norm_scale / (2 * hidden_size)) + 1j * np.sqrt(norm_scale / (2 * hidden_size))
h = h_init * torch.ones(batch_size, hidden_size, dtype=torch.complex64)
initial_norm = torch.linalg.norm(h, dim=-1)
print(f"Initial norm: {initial_norm}")

# Apply ModReLU with beta=0 (zeros initializer)
modrelu = ModReLU(hidden_size, initializer="zeros")
h_after_modrelu = modrelu(h)
norm_after_modrelu = torch.linalg.norm(h_after_modrelu, dim=-1)
print(f"Norm after ModReLU (beta=0): {norm_after_modrelu}")
print(f"Norm change: {norm_after_modrelu - initial_norm}")

# Check what ModReLU does
print("\nModReLU formula: max(0, |z| + beta) * exp(1j * angle(z))")
print("When beta=0: max(0, |z|) * exp(1j * angle(z))")
print("This is: |z| * exp(1j * angle(z)) = z  (for |z| >= 0, which is always true)")
print("So ModReLU with beta=0 should preserve the complex number...")

# But wait, let's check if there's a numerical issue
magnitude = torch.abs(h)
angle = torch.angle(h)
new_magnitude = F.relu(magnitude + modrelu.beta)
reconstructed = new_magnitude * torch.exp(1j * angle)
print(f"\nDirect computation norm: {torch.linalg.norm(reconstructed, dim=-1)}")
print(f"Difference: {torch.linalg.norm(reconstructed - h_after_modrelu)}")

# The real issue: what if some magnitudes are negative after operations?
print("\n" + "=" * 60)
print("2. CHECKING IF NORM DECAY IS FROM UNITARY TRANSFORMATION:")
print("-" * 60)

# Simulate unitary transformation
hidden_size = 8  # Smaller for testing
batch_size = 4
input_size = 4

urnn = URNN(input_size, hidden_size, add_input_dense=False, norm_scale=1.0)
h = urnn.initial_hidden(batch_size)
initial_norm = torch.linalg.norm(h, dim=-1)
print(f"Initial norm: {initial_norm}")

# Create a random input
input_t = torch.randn(batch_size, input_size)

# Forward pass WITHOUT activation
diag_params = urnn.diag_embed(input_t)
rot_params = urnn.rot_embed(input_t.to(torch.complex64))
d1, d2, d3 = torch.chunk(torch.exp(1j * diag_params), 3, dim=-1)
r1, r2 = torch.chunk(rot_params, 2, dim=-1)

from urnn import householder_matrix
R1 = householder_matrix(complex_unit_norm(r1))
R2 = householder_matrix(complex_unit_norm(r2))

# Unitary transformation (should preserve norm)
h_unitary = torch.fft.fft(d1 * h, dim=-1)
h_unitary = torch.bmm(R1, h_unitary.unsqueeze(-1)).squeeze(-1)
h_unitary = torch.fft.ifft(d2 * h_unitary[:, urnn.permutation], dim=-1)
h_unitary = d3 * torch.bmm(R2, h_unitary.unsqueeze(-1)).squeeze(-1)

norm_after_unitary = torch.linalg.norm(h_unitary, dim=-1)
print(f"Norm after unitary transformation: {norm_after_unitary}")
print(f"Norm preservation error: {torch.abs(norm_after_unitary - initial_norm).max().item():.2e}")

# Now add activation
h_with_activation = urnn.activation(h_unitary)
norm_after_activation = torch.linalg.norm(h_with_activation, dim=-1)
print(f"\nNorm after ModReLU activation: {norm_after_activation}")
print(f"Norm change from activation: {norm_after_activation - norm_after_unitary}")
print(f"Relative change: {(norm_after_activation - norm_after_unitary) / norm_after_unitary * 100}%")

# Check if ModReLU beta is zero
print(f"\nModReLU beta values (first 10): {urnn.activation.beta[:10]}")
print(f"ModReLU beta mean: {urnn.activation.beta.mean():.6e}")

print("\n" + "=" * 60)
print("3. INVESTIGATING NUMERICAL STABILITY:")
print("-" * 60)

# Check if there are any small magnitude values that might cause issues
magnitudes = torch.abs(h_unitary)
print(f"Magnitude stats before activation:")
print(f"  Min: {magnitudes.min().item():.6e}")
print(f"  Max: {magnitudes.max().item():.6e}")
print(f"  Mean: {magnitudes.mean().item():.6e}")
print(f"  Number < 1e-8: {(magnitudes < 1e-8).sum().item()}")

# Check if division by abs(x) in angle computation causes issues
# Actually, ModReLU uses torch.angle(x) and torch.exp(1j * angle), which should be fine

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("The unitary transformation should preserve norm exactly.")
print("If norm is decaying, it's likely due to:")
print("1. ModReLU activation (but with beta=0, it should be identity)")
print("2. Numerical precision issues in FFT/IFFT")
print("3. Gradient updates changing the unitary parameters during training")
print("4. The reset logic in get_states() mixing states")


