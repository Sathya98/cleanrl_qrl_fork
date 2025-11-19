# Norm Decay Analysis: URNN with ModReLU

## Root Cause: ModReLU Beta Learning Negative Values

The norm decay in your URNN implementation is caused by **ModReLU's learnable beta parameter becoming negative during training**.

## The Problem

Even though `add_input_dense=False` (so no input embedding is added), the **ModReLU activation** is applied after the unitary transformation:

```python
# In URNN.forward():
h = d3 * torch.bmm(R2, h.unsqueeze(-1)).squeeze(-1)  # Unitary transformation (preserves norm)
h = self.activation(h)  # ModReLU - THIS CAUSES NORM DECAY!
```

## ModReLU Formula

ModReLU is defined as:
```
output = max(0, |z| + beta) * exp(1j * angle(z))
```

Where:
- `z` is the complex input
- `beta` is a learnable real-valued parameter

## Why Negative Beta Causes Norm Decay

When `beta < 0`:
1. `|z| + beta < |z|` (magnitude decreases)
2. If `|z| + beta < 0`, ReLU clips it to 0 (complete magnitude loss)
3. Even if `|z| + beta > 0`, the magnitude is reduced: `new_magnitude = |z| + beta < |z|`

**Result**: The norm of the hidden state decays over time as beta becomes more negative.

## Evidence from Terminal Output

Looking at your terminal output:
- Initial norm: ~3.3636 (stable)
- After training starts: norm gradually decreases
- Norm can decay to near-zero if beta becomes very negative

## Why Beta Learns Negative Values

During training, the gradient updates to `activation.beta` are unconstrained. If the optimizer finds that negative beta values improve the loss (perhaps by regularizing or reducing magnitude), it will learn negative values, causing cumulative norm decay.

## Solutions

### Option 1: Clamp Beta to Non-Negative
```python
# In ModReLU.forward():
# Clamp beta to be non-negative during forward pass
beta = F.relu(self.beta)
new_magnitude = F.relu(magnitude + beta)
```

### Option 2: Initialize Beta to Small Positive Value
```python
# In ModReLU.__init__():
if initializer == "zeros":
    beta = torch.ones(in_features) * 0.01  # Small positive instead of zero
```

### Option 3: Use ReLU on Beta During Updates
```python
# In training loop, after optimizer step:
with torch.no_grad():
    urnn.activation.beta.clamp_(min=0.0)
```

### Option 4: Remove ModReLU (Use Identity)
If you want true norm preservation, you could use an identity activation:
```python
# In URNN.__init__():
if add_input_dense:
    self.activation = ModReLU(hidden_size, initializer="zeros")
else:
    self.activation = nn.Identity()  # No activation for norm preservation
```

## Recommended Fix

For norm-preserving URNNs, I recommend **Option 1** (clamp beta in forward) or **Option 3** (clamp after updates), as they allow beta to learn while preventing norm decay.


