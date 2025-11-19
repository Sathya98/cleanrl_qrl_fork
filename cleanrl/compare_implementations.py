"""
Compare JAX and PyTorch implementations to identify differences.
"""
import torch
import numpy as np
import jax.numpy as jnp
import jax.random as random

print("=" * 60)
print("Key Differences Analysis")
print("=" * 60)

# 1. Input type difference
print("\n1. INPUT TYPE:")
print("-" * 60)
print("JAX test: complex64 inputs")
print("  a = random.normal(key, (num_steps, input_dim), dtype=jnp.complex64)")
print("  Input norm: ~sqrt(2*input_dim) for complex normal")
print()
print("PyTorch test: real inputs")
print("  inputs = torch.randn(sequence_length, batch_size, input_dim)")
print("  Input norm: ~sqrt(input_dim) for real normal")
print()
print("→ JAX inputs have ~sqrt(2) larger norm!")

# 2. norm_scale difference
print("\n2. NORM_SCALE:")
print("-" * 60)
cell_dim = 8
jax_norm_scale = np.sqrt(cell_dim)
pytorch_norm_scale = 1.0
print(f"JAX: norm_scale = sqrt({cell_dim}) = {jax_norm_scale:.6f}")
print(f"PyTorch: norm_scale = {pytorch_norm_scale:.6f}")
print(f"→ JAX initial hidden state norm is {jax_norm_scale:.2f}x larger")

# 3. Initial hidden state
print("\n3. INITIAL HIDDEN STATE:")
print("-" * 60)
jax_initial = jax_norm_scale / np.sqrt(2 * cell_dim) * (1 + 1j)
pytorch_initial = pytorch_norm_scale / np.sqrt(2 * cell_dim) * (1 + 1j)
print(f"JAX initial magnitude per element: {abs(jax_initial):.6f}")
print(f"PyTorch initial magnitude per element: {abs(pytorch_initial):.6f}")
print(f"JAX initial norm (for dim={cell_dim}): {abs(jax_initial) * np.sqrt(cell_dim):.6f}")
print(f"PyTorch initial norm (for dim={cell_dim}): {abs(pytorch_initial) * np.sqrt(cell_dim):.6f}")

# 4. Input embedding initialization
print("\n4. INPUT EMBEDDING INITIALIZATION:")
print("-" * 60)
print("Both use Glorot/Xavier uniform, but let's check the variance:")
# Glorot uniform: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
input_dim = 4
hidden_dim = 8
limit = np.sqrt(6.0 / (input_dim + hidden_dim))
print(f"For input_dim={input_dim}, hidden_dim={hidden_dim}:")
print(f"  Variance of Glorot uniform: {limit**2 / 3:.6f}")
print(f"  Expected norm of embedding output (for real input): ~{np.sqrt(hidden_dim * limit**2 / 3):.6f}")

# 5. Diagonal normalization
print("\n5. DIAGONAL MATRIX NORMALIZATION:")
print("-" * 60)
print("JAX: Explicitly normalizes: d1 = d1/jnp.abs(d1)")
print("PyTorch: Uses torch.exp(1j * diag_init) which is already unit amplitude")
print("→ Both should be unit amplitude, but JAX is more explicit")

# 6. Order of operations
print("\n6. ORDER OF OPERATIONS IN UNITARY TRANSFORMATION:")
print("-" * 60)
print("JAX:")
print("  carry = permute(r1 @ fft(d1 * carry))")
print("  carry = d3 * (r2 @ ifft(d2 * carry))")
print()
print("PyTorch:")
print("  h = fft(d1 * hx)")
print("  h = h @ R1")
print("  h = ifft(d2 * h[permutation])")
print("  h = d3 * (h @ R2)")
print()
print("→ Same operations, different order of matrix multiplication")
print("  (JAX uses @, PyTorch uses @ or indexing)")

# 7. Most critical: Input embedding magnitude
print("\n7. INPUT EMBEDDING MAGNITUDE (MOST CRITICAL):")
print("-" * 60)
print("The norm growth comes from: ||h + input_embed(x)||² = ||h||² + ||input_embed(x)||² + 2*Re(<h, input_embed(x)>)")
print()
print("For JAX (complex input):")
print("  - Input is complex, so input_embed processes complex input")
print("  - But wait... let me check if input_embed expects complex or real...")
print()
print("For PyTorch (real input):")
print("  - Input is real, converted to complex: input.to(torch.complex64)")
print("  - input_embed processes this complex input")
print()
print("The key question: What's the expected norm of input_embed(x)?")

# Simulate input embedding
torch.manual_seed(42)
jax_key = random.PRNGKey(42)

# PyTorch: real input
pytorch_input = torch.randn(4)
pytorch_embed = torch.nn.Linear(4, 8, bias=False, dtype=torch.complex64)
torch.nn.init.xavier_uniform_(pytorch_embed.weight)
pytorch_output = pytorch_embed(pytorch_input.to(torch.complex64))
pytorch_norm = torch.linalg.norm(pytorch_output).item()

# JAX: complex input  
jax_input = random.normal(jax_key, (4,), dtype=jnp.complex64)
# We can't easily create the exact same embedding, but we can estimate
# The embedding weight variance should be similar

print(f"PyTorch: real input norm = {torch.linalg.norm(pytorch_input).item():.6f}")
print(f"PyTorch: embedding output norm = {pytorch_norm:.6f}")
print(f"PyTorch: embedding amplifies by factor = {pytorch_norm / torch.linalg.norm(pytorch_input).item():.6f}")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("The slower norm growth in JAX is likely due to:")
print("1. Different input types (complex vs real) - but this shouldn't matter much")
print("2. Different norm_scale (sqrt(8) vs 1.0) - larger initial norm")
print("3. Potentially different input embedding scaling")
print("4. The key: Check if the input embedding weights have different scales!")


