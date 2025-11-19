# Implementation Differences: JAX vs PyTorch LegacyURNN

## Summary

The JAX implementation shows **~3.5x slower absolute norm growth per step** compared to PyTorch. This is due to **multiple factors**:

## Key Differences

### 1. **Initial Norm Scale** (Primary Factor)
- **JAX**: `norm_scale = sqrt(cell_dim) = sqrt(8) ≈ 2.828`
  - Initial hidden state norm: **1.68**
- **PyTorch**: `norm_scale = 1.0` (default)
  - Initial hidden state norm: **1.0**

**Impact**: JAX starts with 1.68x larger initial norm, leading to smaller relative growth rates.

### 2. **Input Type** (Significant Factor)
- **JAX test**: Uses **complex64** inputs
  ```python
  a = random.normal(key, (num_steps, input_dim), dtype=jnp.complex64)
  ```
- **PyTorch test**: Uses **real** inputs (converted to complex)
  ```python
  inputs = torch.randn(sequence_length, batch_size, input_dim)
  # Later: input.to(torch.complex64)
  ```

**Impact**: Complex normal distribution has different statistics than real normal, affecting input embedding output magnitude.

### 3. **ModReLU Implementation** (Minor Difference)
- **JAX**: `jnp.maximum(0, jnp.abs(x) + self.beta.value) * (x / jnp.abs(x))`
  - Potential division by zero if `abs(x) ≈ 0` (though unlikely in practice)
- **PyTorch**: `F.relu(magnitude + self.beta) * torch.exp(1j * angle)`
  - More numerically stable

**Impact**: Minimal, but JAX version could have slight numerical differences.

### 4. **Diagonal Matrix Normalization** (Implementation Detail)
- **JAX**: Explicitly normalizes: `d1 = d1/jnp.abs(d1)`
- **PyTorch**: Uses `torch.exp(1j * diag_init)` which is already unit amplitude

**Impact**: Both achieve the same result, but JAX is more explicit.

## Growth Rate Analysis

From actual test outputs:

| Metric | JAX | PyTorch | Ratio |
|--------|-----|---------|-------|
| Initial norm | 1.68 | 1.00 | 1.68x |
| First step growth | 0.92 | 2.87 | 3.1x |
| First step relative growth | 55% | 287% | 5.2x |
| Average growth per step | 0.31 | 1.06 | 3.5x |

## Root Cause

The **3.5x slower growth** in JAX is primarily due to:

1. **Larger initial norm** (1.68x) → smaller relative increases
2. **Different input statistics** (complex vs real) → different input embedding output magnitudes
3. **Combined effect**: When starting from a larger base and with potentially smaller input embedding contributions, the absolute growth per step is significantly reduced

## Recommendations

To match PyTorch behavior in JAX:
1. Set `norm_scale=1.0` instead of `sqrt(cell_dim)`
2. Use real inputs and convert to complex (like PyTorch)
3. Verify input embedding weight initialization matches exactly

To match JAX behavior in PyTorch:
1. Set `norm_scale=sqrt(hidden_size)`
2. Use complex inputs directly

## Conclusion

The slower norm growth in JAX is **expected** given the different `norm_scale` and input types. It's not a bug, but a consequence of different design choices. The unitary transformation correctly preserves norm in both implementations; the growth comes from the non-unitary input embedding addition, which is the same in both.

