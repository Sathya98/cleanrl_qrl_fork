"""
Deep comparison of the actual implementations.
"""
import torch
import numpy as np
import jax.numpy as jnp
import jax.random as random
from flax import nnx

print("=" * 60)
print("Deep Implementation Comparison")
print("=" * 60)

# Key insight: Check the actual input embedding behavior
torch.manual_seed(42)
jax_key = random.PRNGKey(42)

input_dim = 4
hidden_dim = 8

# PyTorch input embedding
pytorch_embed = torch.nn.Linear(input_dim, hidden_dim, bias=False, dtype=torch.complex64)
torch.nn.init.xavier_uniform_(pytorch_embed.weight)
pytorch_weight = pytorch_embed.weight.data
print(f"\nPyTorch input_embed weight shape: {pytorch_weight.shape}")
print(f"PyTorch input_embed weight mean: {pytorch_weight.mean().item():.6f}")
print(f"PyTorch input_embed weight std: {pytorch_weight.std().item():.6f}")
print(f"PyTorch input_embed weight abs mean: {pytorch_weight.abs().mean().item():.6f}")

# Test with real input
pytorch_input_real = torch.randn(input_dim)
pytorch_output_real = pytorch_embed(pytorch_input_real.to(torch.complex64))
print(f"\nPyTorch real input norm: {torch.linalg.norm(pytorch_input_real).item():.6f}")
print(f"PyTorch embedding output norm: {torch.linalg.norm(pytorch_output_real).item():.6f}")
print(f"PyTorch amplification factor: {torch.linalg.norm(pytorch_output_real).item() / torch.linalg.norm(pytorch_input_real).item():.6f}")

# Test with complex input (like JAX)
pytorch_input_complex = torch.randn(input_dim, dtype=torch.complex64)
pytorch_output_complex = pytorch_embed(pytorch_input_complex)
print(f"\nPyTorch complex input norm: {torch.linalg.norm(pytorch_input_complex).item():.6f}")
print(f"PyTorch embedding output norm: {torch.linalg.norm(pytorch_output_complex).item():.6f}")
print(f"PyTorch amplification factor: {torch.linalg.norm(pytorch_output_complex).item() / torch.linalg.norm(pytorch_input_complex).item():.6f}")

# Now check JAX - but we need to understand how Glorot works for complex
print("\n" + "=" * 60)
print("JAX Glorot Uniform for Complex Weights:")
print("=" * 60)
print("Glorot uniform: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))")
limit = np.sqrt(6.0 / (input_dim + hidden_dim))
print(f"For input_dim={input_dim}, hidden_dim={hidden_dim}: limit = {limit:.6f}")
print(f"This means weights are in range [-{limit:.6f}, {limit:.6f}]")
print(f"For complex weights, both real and imag parts are in this range")
print(f"Expected |weight| per element: ~{limit * np.sqrt(2/3):.6f} (for uniform in circle)")
print(f"Expected ||W||_F (Frobenius norm): ~{np.sqrt(input_dim * hidden_dim * limit**2 * 2/3):.6f}")

# The key insight: For a given input x, what's ||Wx||?
# For real x: ||Wx|| ≈ ||W||_F * ||x|| / sqrt(hidden_dim) (roughly)
# For complex x: ||Wx|| ≈ ||W||_F * ||x|| / sqrt(hidden_dim) (roughly)

print("\n" + "=" * 60)
print("CRITICAL DIFFERENCE:")
print("=" * 60)
print("JAX test uses: norm_scale = sqrt(cell_dim) = sqrt(8) ≈ 2.828")
print("PyTorch test uses: norm_scale = 1.0")
print()
print("Initial hidden state norm:")
jax_initial_norm = np.sqrt(8)  # sqrt(norm_scale / (2*cell_dim)) * sqrt(cell_dim) = sqrt(norm_scale)
pytorch_initial_norm = 1.0
print(f"  JAX: {jax_initial_norm:.6f}")
print(f"  PyTorch: {pytorch_initial_norm:.6f}")
print()
print("When you add input embedding with norm ~1.0-2.0:")
print(f"  JAX relative increase: ~{(1.5 / jax_initial_norm) * 100:.1f}%")
print(f"  PyTorch relative increase: ~{(1.5 / pytorch_initial_norm) * 100:.1f}%")
print()
print("→ JAX starts with 2.83x larger norm, so same absolute addition")
print("  causes ~2.83x smaller RELATIVE increase!")

# But wait, let's also check if there's a difference in how the unitary transformation
# preserves norm when starting from different scales
print("\n" + "=" * 60)
print("UNITARY TRANSFORMATION NORM PRESERVATION:")
print("=" * 60)
print("Both should preserve norm exactly (up to numerical precision)")
print("So the difference must be in:")
print("  1. Initial norm (JAX 2.83x larger)")
print("  2. Input embedding magnitude relative to hidden state")
print("  3. Possibly different input magnitudes")

# Let's simulate one step
print("\n" + "=" * 60)
print("SIMULATING ONE STEP:")
print("=" * 60)

# JAX-like: norm_scale = sqrt(8), complex input
jax_h_norm = np.sqrt(8)
jax_input_norm = np.sqrt(2 * input_dim)  # complex normal
jax_embed_norm = jax_input_norm * limit * np.sqrt(2/3) * np.sqrt(hidden_dim / input_dim)
jax_new_norm_sq = jax_h_norm**2 + jax_embed_norm**2  # approximate, ignoring inner product
jax_new_norm = np.sqrt(jax_new_norm_sq)
jax_growth = jax_new_norm - jax_h_norm

# PyTorch-like: norm_scale = 1.0, real input  
pytorch_h_norm = 1.0
pytorch_input_norm = np.sqrt(input_dim)  # real normal
pytorch_embed_norm = pytorch_input_norm * limit * np.sqrt(2/3) * np.sqrt(hidden_dim / input_dim)
pytorch_new_norm_sq = pytorch_h_norm**2 + pytorch_embed_norm**2
pytorch_new_norm = np.sqrt(pytorch_new_norm_sq)
pytorch_growth = pytorch_new_norm - pytorch_h_norm

print(f"JAX:")
print(f"  Initial h norm: {jax_h_norm:.6f}")
print(f"  Input norm: {jax_input_norm:.6f}")
print(f"  Embedding output norm (approx): {jax_embed_norm:.6f}")
print(f"  New norm (approx): {jax_new_norm:.6f}")
print(f"  Absolute growth: {jax_growth:.6f}")
print(f"  Relative growth: {(jax_growth/jax_h_norm)*100:.2f}%")
print()
print(f"PyTorch:")
print(f"  Initial h norm: {pytorch_h_norm:.6f}")
print(f"  Input norm: {pytorch_input_norm:.6f}")
print(f"  Embedding output norm (approx): {pytorch_embed_norm:.6f}")
print(f"  New norm (approx): {pytorch_new_norm:.6f}")
print(f"  Absolute growth: {pytorch_growth:.6f}")
print(f"  Relative growth: {(pytorch_growth/pytorch_h_norm)*100:.2f}%")
print()
print(f"→ JAX has {pytorch_growth/jax_growth:.2f}x smaller absolute growth")
print(f"→ JAX has {(pytorch_growth/pytorch_h_norm)/(jax_growth/jax_h_norm):.2f}x smaller relative growth")

