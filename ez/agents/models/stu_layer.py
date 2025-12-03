# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

from __future__ import annotations
import torch
import torch.nn as nn
import math

from ez.agents.models.history_layer import HistoryConcat


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

# Note this is slow (usually we just run this once and save the result)
def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device | None = None,
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
    use_approx: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve input with filters using FFT."""
    bsz, seq_len, d_in = u.shape
    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1

    if use_approx:
        # v expected shape [L, K] in general; when using approx, we still need K
        _, K = v.shape
        _, d_out = v.shape
        v = v.view(1, -1, d_out, 1).to(torch.float32).contiguous()
    else:
        _, K = v.shape
        sgn = sgn.unsqueeze(-1)
        v = v.view(1, -1, K, 1, 1).to(torch.float32).contiguous()

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
    Simplified STU implementation for integrating into RL value networks.
    Handles batched inputs: x ∈ [B, L, I] -> y ∈ [B, L, O].
    """
    def __init__(
        self,
        seq_len: int,
        num_filters: int,
        input_dim: int,
        output_dim: int,
        use_hankel_L: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        default_filters: torch.Tensor | None = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_filters = num_filters  # K
        self.input_dim = input_dim      # I
        self.output_dim = output_dim    # O
        self.use_hankel_L = use_hankel_L
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        # Spectral filters: shape [L, K]. Register as buffer so moves with .to(device)
        if default_filters is not None:
            phi = default_filters.to(self.device, self.dtype)
        else:
            phi = get_spectral_filters(seq_len, num_filters, use_hankel_L, self.device, self.dtype)
        self.register_buffer("phi", phi, persistent=False)

        # FFT length
        self.n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)

        # Learnable projections Φ⁺, Φ⁻ : [K, I, O]
        scale = (num_filters * input_dim) ** -0.5
        self.M_phi_plus = nn.Parameter(
            torch.randn(num_filters, input_dim, output_dim, dtype=dtype, device=self.device) * scale
        )
        if not use_hankel_L:
            self.M_phi_minus = nn.Parameter(
                torch.randn(num_filters, input_dim, output_dim, dtype=dtype, device=self.device) * scale
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
        U_plus, U_minus = convolve(x, self.phi, self.n, use_approx=False) # type: ignore

        # Contract over K and I: [B, L, K, I] ⊗ [K, I, O] -> [B, L, O]
        spectral_plus = torch.einsum('blki,kio->blo', U_plus, self.M_phi_plus)

        if self.use_hankel_L:
            return spectral_plus

        spectral_minus = torch.einsum('blki,kio->blo', U_minus, self.M_phi_minus)
        return spectral_plus + spectral_minus

class HistoryMiniSTU(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_filters: int,
        input_dim: int,
        output_dim: int,
        use_hankel_L: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        default_filters: torch.Tensor | None = None,
    ):
        super().__init__()
        # Defer STU parameter initialization until spatial size (H, W) is known.
        # For non-spatial sequence inputs, input_dim is the per-step feature size.
        self.seq_len = seq_len
        self.num_filters = num_filters
        self.output_dim = output_dim
        self.use_hankel_L = use_hankel_L
        self.dtype = dtype
        self.device = device
        self.default_filters = default_filters

        self.stu: MiniSTU | None = None
        # HistoryConcat operates on the channel/feature dimension of the raw input (e.g., C for images).
        self.history = HistoryConcat(input_dim, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] for spatial or [B, L, I] for sequence
        original_shape = x.shape
        is_spatial = x.dim() == 4
        
        if is_spatial:
            # For 4D input [B, C, H, W], history concat gives [B, C*seq_len, H, W]
            # We must interpret seq_len as temporal length, not H*W.
            # Build a sequence tensor [B, seq_len, C*H*W] for MiniSTU.
            x_hist = self.history.forward(x)
            B, C_hist, H, W = x_hist.shape
            C = C_hist // self.history.history_length
            # [B, C*seq_len, H, W] -> [B, seq_len, C, H, W]
            x_seq = x_hist.view(B, self.history.history_length, C, H, W)
            # -> [B, seq_len, C*H*W]
            x_seq = x_seq.view(B, self.history.history_length, C * H * W)
            # Lazy-init STU with correct per-frame feature size (C*H*W)
            if (self.stu is None) or (self.stu.input_dim != C * H * W):
                self.stu = MiniSTU(
                    self.seq_len,
                    self.num_filters,
                    C * H * W,
                    self.output_dim,
                    self.use_hankel_L,
                    self.dtype,
                    self.device,
                    self.default_filters,
                )
            # Apply STU over temporal dimension (length = seq_len)
            y_seq = self.stu.forward(x_seq)  # [B, seq_len, output_dim]
            # Take the last time step as current output and expand back to spatial map
            y = y_seq[:, -1, :]  # [B, output_dim]
            out = y.view(B, -1, 1, 1).expand(B, y.shape[-1], H, W)
        else:
            # For 2D/3D input, use as-is
            x_hist = self.history.forward(x)
            # For non-spatial input, initialize STU with given per-step feature size if needed
            if (self.stu is None) or (self.stu.input_dim != x_hist.shape[-1]):
                self.stu = MiniSTU(
                    self.seq_len,
                    self.num_filters,
                    x_hist.shape[-1],
                    self.output_dim,
                    self.use_hankel_L,
                    self.dtype,
                    self.device,
                    self.default_filters,
                )
            out = self.stu.forward(x_hist)
        
        return out