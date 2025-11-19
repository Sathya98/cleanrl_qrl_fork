"""
Final analysis comparing actual growth rates.
"""
import numpy as np

print("=" * 60)
print("Actual Growth Rate Comparison")
print("=" * 60)

# From terminal outputs:
# JAX: initial=1.68, step0=2.60, step31=11.48 (32 steps total)
# PyTorch: initial=1.0, step0=3.87, step7=9.48 (8 steps total)

jax_initial = 1.68
jax_step0 = 2.60
jax_step31 = 11.48
jax_steps = 32

pytorch_initial = 1.0
pytorch_step0 = 3.87
pytorch_step7 = 9.48
pytorch_steps = 8

print(f"\nJAX:")
print(f"  Initial norm: {jax_initial:.2f}")
print(f"  After 1 step: {jax_step0:.2f} (growth: {jax_step0 - jax_initial:.2f})")
print(f"  After {jax_steps} steps: {jax_step31:.2f} (total growth: {jax_step31 - jax_initial:.2f})")
print(f"  Average growth per step: {(jax_step31 - jax_initial) / jax_steps:.3f}")
print(f"  Relative growth per step (first step): {((jax_step0 - jax_initial) / jax_initial) * 100:.1f}%")

print(f"\nPyTorch:")
print(f"  Initial norm: {pytorch_initial:.2f}")
print(f"  After 1 step: {pytorch_step0:.2f} (growth: {pytorch_step0 - pytorch_initial:.2f})")
print(f"  After {pytorch_steps} steps: {pytorch_step7:.2f} (total growth: {pytorch_step7 - pytorch_initial:.2f})")
print(f"  Average growth per step: {(pytorch_step7 - pytorch_initial) / pytorch_steps:.3f}")
print(f"  Relative growth per step (first step): {((pytorch_step0 - pytorch_initial) / pytorch_initial) * 100:.1f}%")

print(f"\nComparison:")
print(f"  Absolute growth per step: JAX {((jax_step31 - jax_initial) / jax_steps):.3f} vs PyTorch {((pytorch_step7 - pytorch_initial) / pytorch_steps):.3f}")
print(f"  Ratio: {((pytorch_step7 - pytorch_initial) / pytorch_steps) / ((jax_step31 - jax_initial) / jax_steps):.2f}x faster in PyTorch")
print(f"  Relative growth (first step): JAX {((jax_step0 - jax_initial) / jax_initial) * 100:.1f}% vs PyTorch {((pytorch_step0 - pytorch_initial) / pytorch_initial) * 100:.1f}%")
print(f"  Ratio: {((pytorch_step0 - pytorch_initial) / pytorch_initial) / ((jax_step0 - jax_initial) / jax_initial):.2f}x faster relative growth in PyTorch")

print("\n" + "=" * 60)
print("Key Differences:")
print("=" * 60)
print("1. Initial norm: JAX 1.68 vs PyTorch 1.0 (1.68x larger)")
print("2. First step absolute growth: JAX 0.92 vs PyTorch 2.87 (3.1x larger in PyTorch)")
print("3. First step relative growth: JAX 55% vs PyTorch 287% (5.2x larger in PyTorch)")
print()
print("The slower growth in JAX is due to:")
print("  a) Larger initial norm (1.68x) → smaller relative increases")
print("  b) Potentially different input embedding scaling")
print("  c) Different input types (complex vs real) affecting embedding output magnitude")


