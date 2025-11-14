# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

"""
Spectral Transform Unit (STU) implementation for EfficientZero
Based on "Spectral State Space Models" (Agarwal et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    """Find the nearest power of 2 to x."""
    if not round_up:
        return 1 << math.floor(math.log2(x))
    else:
        return 1 << math.ceil(math.log2(x))


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    """Generate Hankel matrix for spectral filters."""
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generate spectral filters using Hankel matrix eigendecomposition."""
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k ** 0.25
    return phi_k.to(dtype=dtype)


def convolve(
    u: torch.Tensor,
    v: torch.Tensor,
    n: int,
    K: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convolve input with filters using FFT.

    Args:
        u: Input tensor [batch, seq_len, d_in]
        v: Filters [seq_len, K]
        n: FFT length
        K: Number of filters

    Returns:
        U_plus, U_minus: [batch, seq_len, K, d_in]
    """
    bsz, seq_len, d_in = u.shape
    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1

    v = v.view(1, v.shape[0], v.shape[1], 1).to(torch.float32).contiguous()
    assert v.shape[2] == K, f"v.shape[2] ({v.shape[2]}) != K ({K})"
    sgn = sgn.unsqueeze(-1)

    # Expand u to match v dimensions: [bsz, seq_len, K, d_in]
    u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)
    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32).contiguous()
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn
    return U_plus, U_minus


class MiniSTU(nn.Module):
    """
    Simplified STU implementation for sequence modeling.
    Handles batched inputs: x ∈ [B, L, I] -> y ∈ [B, L, O].

    Based on spectral filtering with fixed convolutional filters.
    """
    def __init__(
        self,
        seq_len: int,
        num_filters: int,
        input_dim: int,
        output_dim: int,
        use_hankel_L: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        default_filters: torch.Tensor = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_filters = num_filters  # K
        self.input_dim = input_dim      # I
        self.output_dim = output_dim    # O
        self.use_hankel_L = use_hankel_L
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Spectral filters: shape [L, K]. Register as buffer so moves with .to(device)
        # IMPORTANT: Initialize on CPU to avoid Ray serialization issues with CUDA tensors
        # The buffer will automatically move to CUDA when model.cuda() is called
        if default_filters is not None:
            phi = default_filters.to('cpu', self.dtype)
        else:
            phi = get_spectral_filters(seq_len, num_filters, use_hankel_L, torch.device('cpu'), self.dtype)
        self.register_buffer("phi", phi, persistent=False)

        # FFT length
        self.n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)

        # Learnable projections Φ⁺, Φ⁻ : [K, I, O]
        # IMPORTANT: Initialize on CPU to avoid Ray serialization issues
        scale = (num_filters * input_dim) ** -0.5
        self.M_phi_plus = nn.Parameter(
            torch.randn(num_filters, input_dim, output_dim, dtype=dtype) * scale
        )
        if not use_hankel_L:
            self.M_phi_minus = nn.Parameter(
                torch.randn(num_filters, input_dim, output_dim, dtype=dtype) * scale
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, I] or [L, I]
        returns: [B, L, O]
        """
        # Allow unbatched input [L, I]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # -> [1, L, I]
        assert x.dim() == 3, f"Expected x with shape [B, L, I] or [L, I]; got {tuple(x.shape)}"
        B, L, I = x.shape

        x = x.to(self.M_phi_plus.dtype)
        U_plus, U_minus = convolve(x, self.phi, self.n, self.num_filters)

        # Contract over K and I: [B, L, K, I] ⊗ [K, I, O] -> [B, L, O]
        spectral_plus = torch.einsum('blki,kio->blo', U_plus, self.M_phi_plus)
        if self.use_hankel_L:
            return spectral_plus
        spectral_minus = torch.einsum('blki,kio->blo', U_minus, self.M_phi_minus)

        print("MiniSTU input x.shape:", x.shape)

        return spectral_plus + spectral_minus


class SpectralValuePolicyLayer(nn.Module):
    """
    Enhanced value-policy network with spectral filtering.

    Processes hidden state sequences using STU to capture long-range
    temporal dependencies before value/policy prediction.
    """
    def __init__(
        self,
        num_blocks: int,
        num_channels: int,
        reduced_channels: int,
        flatten_size: int,
        fc_layers: list,
        value_output_size: int,
        policy_output_size: int,
        init_zero: bool,
        seq_len: int = 6,  # unroll_steps + 1
        num_spectral_filters: int = 16,
        use_spectral: bool = True,
        **kwargs
    ):
        """
        Args:
            seq_len: Expected sequence length (unroll_steps + 1 for initial state)
            num_spectral_filters: Number of spectral filters K
            use_spectral: Whether to use spectral filtering (for ablation)
        """
        super().__init__()
        from .base_model import ValuePolicyNetwork
        from .layer import ResidualBlock

        self.use_spectral = use_spectral
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.v_num = kwargs.get('v_num', 1)

        # Baseline value-policy network
        self.base_network = ValuePolicyNetwork(
            num_blocks, num_channels, reduced_channels, flatten_size,
            fc_layers, value_output_size, policy_output_size, init_zero,
            **kwargs
        )

        if self.use_spectral:
            # STU for processing state sequences
            # Input: flattened spatial hidden states
            # Output: enhanced features for value/policy prediction
            self.spectral_layer = MiniSTU(
                seq_len=seq_len,
                num_filters=num_spectral_filters,
                input_dim=flatten_size,
                output_dim=flatten_size,
                use_hankel_L=False,
                dtype=torch.float32
            )

            # Additional projection to combine spectral and spatial features
            self.spectral_proj = nn.Sequential(
                nn.Linear(flatten_size * 2, flatten_size),
                nn.LayerNorm(flatten_size),
                nn.ReLU()
            )

    def forward(self, x: torch.Tensor, state_sequence: torch.Tensor = None):
        """
        Forward pass with optional spectral enhancement.

        Args:
            x: Current hidden state [B, C, H, W]
            state_sequence: Optional sequence of states [B, seq_len, C, H, W]
                          If provided, will apply spectral filtering

        Returns:
            values: [v_num, B, value_output_size]
            policy: [B, policy_output_size]
        """
        # Process through residual blocks
        for block in self.base_network.resblocks:
            x = block(x)

        # Flatten spatial dimensions
        B = x.shape[0]
        x_flat = x.reshape(B, -1)  # [B, C*H*W]

        if self.use_spectral and state_sequence is not None:
            # Apply spectral filtering to state sequence
            # state_sequence: [B, seq_len, C, H, W]
            B_seq, L, C, H, W = state_sequence.shape
            state_seq_flat = state_sequence.reshape(B_seq, L, -1)  # [B, seq_len, C*H*W]

            # Apply STU
            spectral_features = self.spectral_layer(state_seq_flat)  # [B, seq_len, flatten_size]

            # Use the last timestep's spectral features
            spectral_features_last = spectral_features[:, -1, :]  # [B, flatten_size]

            # Combine spatial and temporal (spectral) features
            combined = torch.cat([x_flat, spectral_features_last], dim=-1)
            x_enhanced = self.spectral_proj(combined)
        else:
            x_enhanced = x_flat

        # Reshape back for value/policy heads
        x_enhanced = x_enhanced.reshape(B, -1, x.shape[2], x.shape[3])

        # Compute values (multi-head)
        values = []
        for i in range(self.v_num):
            value = self.base_network.conv1x1_values[i](x_enhanced)
            value = self.base_network.bn_values[i](value)
            value = F.relu(value)
            value = value.reshape(B, -1)
            value = self.base_network.fc_values[i](value)
            values.append(value)

        # Compute policy
        policy = self.base_network.conv1x1_policy(x_enhanced)
        policy = self.base_network.bn_policy(policy)
        policy = F.relu(policy)
        policy = policy.reshape(B, -1)
        policy = self.base_network.fc_policy(policy)

        return torch.stack(values), policy


def create_spectral_value_policy_network(config, use_spectral=True):
    """
    Factory function to create spectral-enhanced value-policy network.

    Args:
        config: Configuration object with model parameters
        use_spectral: Whether to enable spectral filtering
    """
    # Calculate flatten size
    if config.model.down_sample:
        state_shape = (config.model.num_channels,
                      math.ceil(config.env.obs_shape[1] / 16),
                      math.ceil(config.env.obs_shape[2] / 16))
    else:
        state_shape = (config.model.num_channels,
                      config.env.obs_shape[1],
                      config.env.obs_shape[2])

    flatten_size = config.model.reduced_channels * state_shape[1] * state_shape[2]
    seq_len = config.rl.unroll_steps + 1  # +1 for initial state

    return SpectralValuePolicyLayer(
        num_blocks=config.model.num_blocks,
        num_channels=config.model.num_channels,
        reduced_channels=config.model.reduced_channels,
        flatten_size=flatten_size,
        fc_layers=config.model.fc_layers,
        value_output_size=config.model.value_support.size,
        policy_output_size=config.env.action_space_size,
        init_zero=config.model.init_zero,
        seq_len=seq_len,
        num_spectral_filters=config.model.get('num_spectral_filters', 16),
        use_spectral=use_spectral,
        v_num=config.train.v_num
    )
