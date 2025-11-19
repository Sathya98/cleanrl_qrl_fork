# Norm Growth Analysis: JAX vs PyTorch URNN Implementation

## Key Finding: Different `norm_scale` Values

The **primary reason** for slower norm growth in the JAX implementation is the different `norm_scale` parameter:

- **JAX**: `norm_scale = sqrt(cell_dim) = sqrt(8) ≈ 2.828`
- **PyTorch**: `norm_scale = 1.0` (default)

## Impact on Initial Hidden State

The initial hidden state is computed as:
```python
h = sqrt(norm_scale / (2 * cell_dim)) * (1 + 1j) * ones(cell_dim)
```

This gives:
- **JAX initial norm**: `sqrt(8) ≈ 2.828`
- **PyTorch initial norm**: `1.0`

## Why This Causes Slower Growth

When adding input embeddings, the norm grows according to:
```
||h_new||² = ||h_old||² + ||input_embed(x)||² + 2*Re(<h_old, input_embed(x)>)
```

For similar input embedding magnitudes (~1.5-2.0):

- **JAX**: Starting from norm 2.83, adding ~1.5 gives ~53% relative increase
- **PyTorch**: Starting from norm 1.0, adding ~1.5 gives ~150% relative increase

The **absolute growth** is similar, but the **relative growth** is ~3x smaller in JAX because the denominator (initial norm) is ~2.83x larger.

## Other Differences (Minor)

1. **Input type**: JAX test uses complex inputs, PyTorch uses real inputs
   - This affects input magnitude but not the core mechanism

2. **Diagonal normalization**: JAX explicitly normalizes (`d1 = d1/abs(d1)`), PyTorch uses `exp(1j * angle)` which is already unit amplitude
   - Both achieve the same result

3. **Unitary transformation order**: Slightly different syntax but mathematically equivalent
   - Both preserve norm correctly

## Conclusion

The slower norm growth in JAX is **expected behavior** due to the larger `norm_scale`. This is not a bug, but a design choice. If you want to match the PyTorch behavior, set `norm_scale=1.0` in the JAX implementation.


