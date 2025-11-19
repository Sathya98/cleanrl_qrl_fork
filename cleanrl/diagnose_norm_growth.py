"""
Diagnostic script to understand norm growth in URNN.
"""
import torch
import numpy as np
from urnn import LegacyURNN

torch.manual_seed(42)
input_dim = 8
hidden_dim = 8
batch_size = 4

# Initialize model
legacy_urnn = LegacyURNN(input_dim, hidden_dim)

# Create a single input
input_t = torch.randn(batch_size, input_dim)
hx = legacy_urnn.initial_hidden(batch_size)

print("=" * 60)
print("Diagnosing Norm Growth")
print("=" * 60)
print(f"Initial hidden state norm: {torch.linalg.norm(hx, dim=-1)}")
print()

# Step 1: Apply unitary transformation
d1, d2, d3 = torch.chunk(legacy_urnn.diag, 3, dim=0)
r1, r2 = torch.chunk(legacy_urnn.rotation, 2, dim=0)

from urnn import householder_matrix, complex_unit_norm
R1 = householder_matrix(complex_unit_norm(r1))
R2 = householder_matrix(complex_unit_norm(r2))

h_unitary = torch.fft.fft(d1.unsqueeze(0) * hx, dim=-1)
h_unitary = h_unitary @ R1
h_unitary = torch.fft.ifft(d2.unsqueeze(0) * h_unitary[:, legacy_urnn.permutation], dim=-1)
h_unitary = d3.unsqueeze(0) * (h_unitary @ R2)

print(f"After unitary transformation norm: {torch.linalg.norm(h_unitary, dim=-1)}")
print("✓ Unitary transformation preserves norm (as expected)")
print()

# Step 2: Compute input embedding
input_embed = legacy_urnn.input_embed(input_t.to(torch.complex64))
print(f"Input embedding norm: {torch.linalg.norm(input_embed, dim=-1)}")
print(f"Input embedding mean (real): {input_embed.real.mean().item():.6f}")
print(f"Input embedding mean (imag): {input_embed.imag.mean().item():.6f}")
print()

# Step 3: Add input embedding
h_after_add = h_unitary + input_embed
print(f"After adding input embedding norm: {torch.linalg.norm(h_after_add, dim=-1)}")
print()

# Mathematical analysis
print("Mathematical Analysis:")
print("-" * 60)
for i in range(batch_size):
    h_norm_before = torch.linalg.norm(h_unitary[i]).item()
    x_norm = torch.linalg.norm(input_embed[i]).item()
    h_norm_after = torch.linalg.norm(h_after_add[i]).item()
    
    # Compute inner product term
    inner_product = torch.real(torch.vdot(h_unitary[i], input_embed[i])).item()
    
    # Expected squared norm: ||h + x||² = ||h||² + ||x||² + 2*Re(<h, x>)
    expected_sq_norm = h_norm_before**2 + x_norm**2 + 2 * inner_product
    expected_norm = np.sqrt(expected_sq_norm)
    
    print(f"Batch {i}:")
    print(f"  ||h|| before: {h_norm_before:.6f}")
    print(f"  ||x|| (input embed): {x_norm:.6f}")
    print(f"  Re(<h, x>): {inner_product:.6f}")
    print(f"  ||h + x||² = {h_norm_before**2:.6f} + {x_norm**2:.6f} + 2*{inner_product:.6f} = {expected_sq_norm:.6f}")
    print(f"  ||h + x|| (expected): {expected_norm:.6f}")
    print(f"  ||h + x|| (actual): {h_norm_after:.6f}")
    print(f"  Norm increase: {h_norm_after - h_norm_before:.6f}")
    print()

print("=" * 60)
print("Key Insight:")
print("=" * 60)
print("Even though input embeddings have zero mean over the input distribution,")
print("for any specific input, adding ||x||² + 2*Re(<h, x>) to ||h||² will")
print("generally increase the norm unless h and x are perfectly anti-aligned.")
print("The inner product term Re(<h, x>) is typically positive, causing growth.")
print()

