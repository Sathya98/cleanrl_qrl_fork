import torch
import numpy as np
from urnn import URNN, LegacyURNN, householder_matrix

# Test parameters
input_dim = 32
hidden_dim = 32
sequence_length = 16
batch_size = 4
torch.manual_seed(42)

def test_householder_matrix():
    """Test that householder_matrix produces unitary matrices."""
    print("=" * 60)
    print("Testing householder_matrix")
    print("=" * 60)
    
    hidden_size = hidden_dim
    # Create a random complex vector and normalize it
    v = torch.randn((1, hidden_size), dtype=torch.complex64)
    v = v / torch.linalg.norm(v)  # Normalize
    
    # Build Householder matrix
    H = householder_matrix(v)  # (1, hidden_size, hidden_size)
    
    # Verify unitarity: H @ H.conj().T should be close to I
    I_expected = torch.eye(hidden_size, dtype=torch.complex64).unsqueeze(0)  # (1, hidden_size, hidden_size)
    H_H_dag = torch.bmm(H, H.conj().transpose(-2, -1))  # (1, hidden_size, hidden_size)
    
    error = torch.abs(H_H_dag - I_expected).max().item()  # scalar
    print(f"Householder matrix shape: {H.shape}")
    print(f"Max error from identity (|H @ H^dagger - I|): {error:.2e}")
    
    if error < 1e-5:
        print("✓ Householder matrix is unitary (within tolerance)")
    else:
        print(f"✗ Householder matrix may not be unitary (error: {error:.2e})")
    
    # Also check determinant (should be -1 for reflection)
    det = torch.linalg.det(H)
    print(f"Determinant: {det.item()} (expected: -1.0 for reflection)")
    
    print()


def test_urnn():
    """Test URNN with a sequence of inputs."""
    print("=" * 60)
    print("Testing URNN")
    print("=" * 60)
    
    # Initialize URNN
    urnn = URNN(input_dim, hidden_dim)
    print(f"URNN initialized: input_size={input_dim}, hidden_size={hidden_dim}")
    
    # Create random input sequence
    inputs = torch.randn(sequence_length, batch_size, input_dim)
    print(f"Input sequence shape: {inputs.shape}")
    
    # Initialize hidden state
    hx = urnn.initial_hidden(batch_size)
    print(f"Initial hidden state shape: {hx.shape}")
    print(f"Initial hidden state norm: {torch.abs(hx).mean().item():.6f}")
    
    # Process sequence
    outputs = []
    norms = []
    h = hx
    
    print("\nProcessing sequence:")
    for t in range(sequence_length):
        input_t = inputs[t]  # (batch, input_dim)
        output, h = urnn(input_t, h)
        outputs.append(output)
        
        # Track norms
        norm = torch.linalg.norm(output).item()
        norms.append(norm)
        
        if t < 3 or t >= sequence_length - 3:
            print(f"  Step {t:2d}: output norm = {norm:.6f}, hidden norm = {torch.abs(h).mean().item():.6f}")
    
    outputs = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden_dim)
    
    print(f"\nOutput sequence shape: {outputs.shape}")
    print(f"Mean output norm over sequence: {np.mean(norms):.6f}")
    print(f"Std output norm over sequence: {np.std(norms):.6f}")
    print(f"Min output norm: {min(norms):.6f}")
    print(f"Max output norm: {max(norms):.6f}")
    
    # Check that outputs are complex
    assert torch.is_complex(outputs), "Outputs should be complex"
    print("✓ Outputs are complex-valued")
    
    # Check that hidden state is complex
    assert torch.is_complex(h), "Hidden state should be complex"
    print("✓ Final hidden state is complex-valued")
    
    print()


def test_legacy_urnn():
    """Test LegacyURNN with a sequence of inputs."""
    print("=" * 60)
    print("Testing LegacyURNN")
    print("=" * 60)
    
    # Initialize LegacyURNN
    legacy_urnn = LegacyURNN(input_dim, hidden_dim)
    print(f"LegacyURNN initialized: input_size={input_dim}, hidden_size={hidden_dim}")
    
    # Create random input sequence
    inputs = torch.randn(sequence_length, batch_size, input_dim)
    print(f"Input sequence shape: {inputs.shape}")
    
    # Initialize hidden state
    hx = legacy_urnn.initial_hidden(batch_size)
    print(f"Initial hidden state shape: {hx.shape}")
    print(f"Initial hidden state norm: {torch.abs(hx).mean().item():.6f}")
    
    # Process sequence
    outputs = []
    norms = []
    h = hx
    
    print("\nProcessing sequence:")
    for t in range(sequence_length):
        input_t = inputs[t]  # (batch, input_dim)
        output, h = legacy_urnn(input_t, h)
        outputs.append(output)
        
        # Track norms
        norm = torch.abs(output).mean().item()
        norms.append(norm)
        
        if t < 3 or t >= sequence_length - 3:
            print(f"  Step {t:2d}: output norm = {norm:.6f}, hidden norm = {torch.abs(h).mean().item():.6f}")
    
    outputs = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden_dim)
    
    print(f"\nOutput sequence shape: {outputs.shape}")
    print(f"Mean output norm over sequence: {np.mean(norms):.6f}")
    print(f"Std output norm over sequence: {np.std(norms):.6f}")
    print(f"Min output norm: {min(norms):.6f}")
    print(f"Max output norm: {max(norms):.6f}")
    
    # Check that outputs are complex
    assert torch.is_complex(outputs), "Outputs should be complex"
    print("✓ Outputs are complex-valued")
    
    # Check that hidden state is complex
    assert torch.is_complex(h), "Hidden state should be complex"
    print("✓ Final hidden state is complex-valued")
    
    print()


def test_gradient_flow():
    """Test that gradients flow through the URNN."""
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    urnn = URNN(input_dim, hidden_dim)
    inputs = torch.randn(sequence_length, batch_size, input_dim, requires_grad=True)
    hx = urnn.initial_hidden(batch_size)
    
    # Forward pass
    h = hx
    for t in range(sequence_length):
        _, h = urnn(inputs[t], h)
    
    # Compute a simple loss (mean of real part)
    loss = h.real.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = inputs.grad is not None
    print(f"Input gradients computed: {has_grad}")
    
    if has_grad:
        grad_norm = inputs.grad.norm().item()
        print(f"Input gradient norm: {grad_norm:.6f}")
        print("✓ Gradients flow through URNN")
    else:
        print("✗ No gradients computed")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("URNN Test Suite")
    print("=" * 60)
    print(f"Test configuration:")
    print(f"  input_dim: {input_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  sequence_length: {sequence_length}")
    print(f"  batch_size: {batch_size}")
    print()
    
    # Run tests
    test_householder_matrix()
    test_urnn()
    test_legacy_urnn()
    test_gradient_flow()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

