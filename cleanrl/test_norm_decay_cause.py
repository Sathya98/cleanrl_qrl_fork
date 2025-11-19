"""
Test to identify the cause of norm decay.
"""
import torch
import numpy as np
from urnn import URNN

print("=" * 60)
print("Testing Norm Decay Causes")
print("=" * 60)

hidden_size = 128
input_size = 512
batch_size = 4
norm_scale = np.sqrt(128)

urnn = URNN(input_size, hidden_size, add_input_dense=False, norm_scale=norm_scale)
h = urnn.initial_hidden(batch_size)
print(f"Initial norm: {torch.linalg.norm(h, dim=-1)}")
print(f"Expected initial norm: {norm_scale:.6f}")

# Test: Does ModReLU beta learn negative values during training?
print("\n" + "=" * 60)
print("1. MODRELU BETA LEARNING:")
print("-" * 60)
print("If beta learns negative values, ModReLU will reduce norm!")
print(f"Initial beta mean: {urnn.activation.beta.mean().item():.6e}")
print(f"Initial beta min: {urnn.activation.beta.min().item():.6e}")
print(f"Initial beta max: {urnn.activation.beta.max().item():.6e}")

# Simulate beta becoming negative
urnn.activation.beta.data = torch.full_like(urnn.activation.beta, -0.1)
h_test = torch.ones(batch_size, hidden_size, dtype=torch.complex64) * (1 + 1j)
norm_before = torch.linalg.norm(h_test, dim=-1)
h_after = urnn.activation(h_test)
norm_after = torch.linalg.norm(h_after, dim=-1)
print(f"\nWith beta=-0.1:")
print(f"  Norm before: {norm_before}")
print(f"  Norm after: {norm_after}")
print(f"  Norm change: {norm_after - norm_before}")
print("  → Negative beta causes norm DECAY!")

# Reset beta to zero
urnn.activation.beta.data = torch.zeros_like(urnn.activation.beta)

print("\n" + "=" * 60)
print("2. RESET LOGIC TEST:")
print("-" * 60)

# Test the reset logic from get_states
done = torch.zeros(batch_size, dtype=torch.float32)
done[0] = 1.0  # First env is done

reset_mask = (1.0 - done).view(batch_size, 1)
print(f"Done flags: {done}")
print(f"Reset mask: {reset_mask.squeeze()}")

h_before_reset = h.clone()
h_after_reset = reset_mask * h + (1 - reset_mask) * urnn.initial_hidden(batch_size)

norm_before = torch.linalg.norm(h_before_reset, dim=-1)
norm_after = torch.linalg.norm(h_after_reset, dim=-1)
print(f"\nNorm before reset: {norm_before}")
print(f"Norm after reset: {norm_after}")
print("→ Reset preserves norm correctly")

print("\n" + "=" * 60)
print("3. CHECKING IF UNITARY PARAMS CHANGE NORM:")
print("-" * 60)

# Test with different input values
input1 = torch.randn(batch_size, input_size)
input2 = torch.randn(batch_size, input_size)

h1 = urnn.initial_hidden(batch_size)
h2 = urnn.initial_hidden(batch_size)

# Forward pass
_, h1_new = urnn(input1, h1)
_, h2_new = urnn(input2, h2)

norm1 = torch.linalg.norm(h1_new, dim=-1)
norm2 = torch.linalg.norm(h2_new, dim=-1)
print(f"Norm after input1: {norm1}")
print(f"Norm after input2: {norm2}")
print(f"Expected norm: {norm_scale:.6f}")

# Check if there's numerical drift
print(f"\nNumerical error: {torch.abs(norm1 - norm_scale).max().item():.6e}")

print("\n" + "=" * 60)
print("4. CHECKING GRADIENT UPDATES:")
print("-" * 60)
print("During training, gradient updates to:")
print("  - diag_embed.weight (affects d1, d2, d3)")
print("  - rot_embed.weight (affects r1, r2)")
print("  - activation.beta (affects ModReLU)")
print()
print("Even though d1, d2, d3 are normalized to unit amplitude with exp(1j * ...),")
print("the embedding weights can change, which affects the unitary transformation.")
print()
print("However, the unitary transformation itself should still preserve norm")
print("for any values of d1, d2, d3 (as long as they're unit amplitude).")
print()
print("The REAL issue might be ModReLU beta learning negative values!")

# Let's check what happens if we manually set beta to negative
print("\n" + "=" * 60)
print("5. SIMULATING MODRELU WITH NEGATIVE BETA:")
print("-" * 60)

urnn.activation.beta.data = torch.full_like(urnn.activation.beta, -0.05)
h_init = urnn.initial_hidden(batch_size)
norm_init = torch.linalg.norm(h_init, dim=-1)

# Multiple steps
h = h_init
for step in range(10):
    input_t = torch.randn(batch_size, input_size)
    _, h = urnn(input_t, h)
    norm = torch.linalg.norm(h, dim=-1)
    if step % 2 == 0:
        print(f"Step {step}: norm = {norm.mean().item():.6f}")

print(f"\nFinal norm: {norm}")
print(f"Initial norm: {norm_init}")
print(f"Norm decay: {(norm_init - norm).mean().item():.6f}")
print("→ Negative beta causes cumulative norm decay!")


