import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def householder_matrix(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Householder reflection matrix for a given vector v.
    Assumes v is already normalized.
    
    Args:
        v: (..., hidden_size) complex vector, already normalized
    Returns:
        H: (..., hidden_size, hidden_size) unitary matrix
    """
    v = v.unsqueeze(-1)  # (..., hidden_size, 1)
    v_dag = v.conj().transpose(-2, -1)  # (..., 1, hidden_size)

    if len(v.shape) == 2:
        I = torch.eye(v.shape[-2], dtype=v.dtype, device=v.device)
    else:
        I = torch.tile(torch.eye(v.shape[-2], dtype=v.dtype, device=v.device), (v.shape[0], 1, 1))
    
    return I - 2 * (v @ v_dag)


def complex_unit_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize complex vector to unit norm.
    
    Args:
        x: Complex tensor of any shape
        eps: Small epsilon for numerical stability
    Returns:
        Normalized complex tensor with same shape
    """
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / (norm + eps)


class ModReLU(nn.Module):
    """
    Modulated ReLU activation function for complex-valued inputs.
    
    Formula: max(0, |z| + beta) * exp(1j * angle(z))
    where z is complex and beta is a learnable real-valued bias.
    """
    
    def __init__(
        self,
        in_features: int,
        initializer: str = "zeros",
        scale: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        
        if initializer == "zeros":
            beta = torch.zeros(in_features)
        elif initializer == "ones":
            beta = torch.ones(in_features)
        elif initializer == "normal":
            beta = torch.randn(in_features) * scale
        else:
            raise ValueError(f"Invalid initializer: {initializer}")
        
        self.beta = nn.Parameter(beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of shape (..., in_features)
        Returns:
            Complex tensor of same shape
        """
        magnitude = torch.abs(x)
        angle = torch.angle(x)
        # ModReLU: max(0, |z| + beta) * exp(1j * angle(z))
        new_magnitude = F.relu(magnitude + self.beta)
        return new_magnitude * torch.exp(1j * angle)


class URNN(nn.Module):
    """
    Unitary RNN Cell based on Arjovsky et al. 2015.
    
    Implements the unitary matrix decomposition: W = D3 R2 F^(-1) D2 Π R1 F D1
    where:
    - D1, D2, D3: Diagonal matrices (unit amplitude complex numbers, input-dependent)
    - R1, R2: Householder reflection matrices (unitary rotations, input-dependent)
    - F, F^(-1): FFT and IFFT operations
    - Π: Permutation matrix (fixed at initialization)
    
    Takes real input and processes with complex hidden state.
    Uses broadcasting for efficient batch processing.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        add_input_dense: bool = False,
        norm_scale: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.add_input_dense = add_input_dense
        self.norm_scale = norm_scale
        
        self.diag_embed = nn.Linear(input_size, hidden_size * 3)
        self.rot_embed = nn.Linear(input_size, hidden_size * 2, dtype=torch.complex64)
        
        # Optional input feed (dense connection)
        if add_input_dense:
            self.input_embed = nn.Linear(input_size, hidden_size, bias=False, dtype=torch.complex64)
        else:
            self.input_embed = None
        
        # Fixed permutation matrix (not learnable)
        perm = torch.randperm(hidden_size)
        self.register_buffer('permutation', perm)
        self.activation = ModReLU(hidden_size, initializer="zeros")
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with orthogonal initialization."""
        nn.init.orthogonal_(self.diag_embed.weight, gain=1.0)
        nn.init.xavier_uniform_(self.rot_embed.weight)
        if self.input_embed is not None:
            nn.init.xavier_uniform_(self.input_embed.weight)
    
    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize hidden state to equal superposition."""
        device = next(self.parameters()).device
        h = np.sqrt(self.norm_scale / (2 * self.hidden_size)) + 1j * np.sqrt(self.norm_scale / (2 * self.hidden_size))
        return h * torch.ones(batch_size, self.hidden_size, dtype=torch.complex64, device=device)
    
    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of URNN cell with broadcasting.
        
        Args:
            input: Real tensor of shape (batch, input_size)
            hx: Optional hidden state of shape (batch, hidden_size) complex
        Returns:
            output: (batch, hidden_size) complex tensor
            h_new: (batch, hidden_size) complex tensor (same as output)
        """
        diag_params = self.diag_embed(input)
        rot_params = self.rot_embed(input.to(torch.complex64))
        d1, d2, d3 = torch.chunk(torch.exp(1j * diag_params), 3, dim=-1)  # (batch, hidden_size) complex
        r1, r2 = torch.chunk(rot_params, 2, dim=-1)  # (batch, hidden_size) complex
        R1 = householder_matrix(complex_unit_norm(r1))
        R2 = householder_matrix(complex_unit_norm(r2))
        
        h = torch.fft.fft(d1 * hx, dim=-1)  # (batch, hidden_size)
        h = torch.bmm(R1, h.unsqueeze(-1)).squeeze(-1)
        h = torch.fft.ifft(d2 * h[:, self.permutation], dim=-1)
        h = d3 * torch.bmm(R2, h.unsqueeze(-1)).squeeze(-1)
        
        # Optional input feed
        if self.add_input_dense and self.input_embed is not None:
            h = h + self.input_embed(input.to(torch.complex64))
        h = self.activation(h)
        
        return h, h


class LegacyURNN(nn.Module):
    """
    Legacy Unitary RNN Cell - original version from Arjovsky et al. 2015.
    
    Same unitary decomposition as URNN, but with learnable parameters for
    D1, D2, D3, R1, R2 (not input-dependent). Input is embedded and added
    to the hidden state after the unitary transformation.
    
    Implements: W = D3 R2 F^(-1) D2 Π R1 F D1
    where D1, D2, D3, R1, R2 are learnable parameters.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        norm_scale: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.norm_scale = norm_scale
        self.eps = eps
        
        # Learnable diagonal matrices (complex parameters)
        diag_init = torch.rand(hidden_size * 3) * 2 * np.pi - np.pi
        self.diag = nn.Parameter(torch.exp(1j * diag_init))
        
        # Learnable rotation vectors (complex parameters)
        self.rotation = nn.Parameter(torch.rand(hidden_size * 2, dtype=torch.complex64) * 2 - (1 + 1j))
        
        # Input embedding: real input -> complex hidden_size
        self.input_embed = nn.Linear(input_size, hidden_size, bias=False, dtype=torch.complex64)
        
        # Fixed permutation matrix
        perm = torch.randperm(hidden_size)
        self.register_buffer('permutation', perm)
        
        self.activation = ModReLU(hidden_size, initializer="zeros")
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.input_embed.weight)
    
    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize hidden state to equal superposition."""
        device = next(self.parameters()).device
        h = np.sqrt(self.norm_scale / (2 * self.hidden_size)) + 1j * np.sqrt(self.norm_scale / (2 * self.hidden_size))
        return h * torch.ones(batch_size, self.hidden_size, dtype=torch.complex64, device=device)
    
    def forward(
        self,
        input: torch.Tensor,
        hx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Legacy URNN cell with broadcasting.
        
        Args:
            input: Real tensor of shape (batch, input_size)
            hx: Hidden state of shape (batch, hidden_size) complex
        Returns:
            output: (batch, hidden_size) complex tensor
            h_new: (batch, hidden_size) complex tensor (same as output)
        """
        # Split and normalize learnable parameters
        d1, d2, d3 = torch.chunk(self.diag, 3, dim=0)
        r1, r2 = torch.chunk(self.rotation, 2, dim=0)
        # Build Householder matrices (same for all batch elements)
        R1 = householder_matrix(complex_unit_norm(r1))  # (hidden_size, hidden_size)
        R2 = householder_matrix(complex_unit_norm(r2))  # (hidden_size, hidden_size)
        
        # Apply unitary transformation: D3 R2 F^(-1) D2 Π R1 F D1
        h = torch.fft.fft(d1.unsqueeze(0) * hx, dim=-1)  # (batch, hidden_size)
        h = h @ R1  # (batch, hidden_size)
        h = torch.fft.ifft(d2.unsqueeze(0) * h[:, self.permutation], dim=-1)  # (batch, hidden_size)
        h = d3.unsqueeze(0) * (h @ R2)  # (batch, hidden_size)
        # Add input embedding
        h = h + self.input_embed(input.to(torch.complex64))
        h = self.activation(h)
        
        return h, h
