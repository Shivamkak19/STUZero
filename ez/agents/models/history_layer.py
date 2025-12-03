import torch
from torch import nn, Tensor

class HistoryConcat(nn.Module):
    """
    Keep the last `history_length` inputs per batch element.

    - Maintains a per-batch buffer with zero-padding at start.
    - Handles device/dtype automatically and supports single-example input.
    - Returns either time-major or channel-concatenated history for filtering.
    """
    def __init__(self, in_shape: tuple, history_length: int, *,
                 concat_channels: bool = True,
                 detach_past: bool = True):
        super().__init__()
        self.in_shape = tuple(in_shape)
        self.history_length = int(history_length)
        self.concat_channels = bool(concat_channels)
        self.detach_past = bool(detach_past)

        # Registered buffer is created lazily when batch size is known.
        # Set persistent=False so it's not saved in state_dict (avoids shape mismatch on load)
        self.register_buffer("buffer", torch.empty(0), persistent=False)

    def reset(self, batch_size: int, device=None, dtype=None) -> None:
        """Reset the buffer to zeros for a new episode or batch.
        Creates a `[B, T, *in_shape]` buffer with zeros.
        """
        if device is None:
            device = self.buffer.device if self.buffer.numel() > 0 else torch.device("cpu")
        if dtype is None:
            dtype = self.buffer.dtype if self.buffer.numel() > 0 else torch.float32
        shape = (int(batch_size), self.history_length, *self.in_shape)
        self.buffer = torch.zeros(shape, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # Normalize input to `[B, *in_shape]` by adding batch if needed.
        if x.dim() == len(self.in_shape):
            x = x.unsqueeze(0)

        assert tuple(x.shape[1:]) == self.in_shape, (
            f"Expected input shape [B, {self.in_shape}]; got {tuple(x.shape)}"
        )

        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # Lazily allocate buffer or reallocate if batch size changed.
        if self.buffer.numel() == 0 or self.buffer.shape[0] != B:
            self.reset(B, device=device, dtype=dtype)

        # Roll time dimension left and append current x at the end.
        # Buffer shape: [B, T, *in_shape]
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        current = x.detach() if self.detach_past else x
        self.buffer[:, -1] = current

        # Return for spectral filtering:
        # - If concat_channels: concatenate time along channel/features axis.
        # - Else: keep `[B, T, *in_shape]` for explicit temporal ops.
        if self.concat_channels:
            # Move time into channels/features: e.g., [B, T, C, H, W] -> [B, T*C, H, W]
            if len(self.in_shape) == 1:
                # Vector input: [B, T, D] -> [B, T*D]
                Bsz, T, D = self.buffer.shape[:3]
                return self.buffer.reshape(Bsz, T * D)
            elif len(self.in_shape) == 2:
                # 2D features: [B, T, C, H] -> [B, T*C, H]
                Bsz, T, C, H = self.buffer.shape[:4]
                return self.buffer.reshape(Bsz, T * C, H)
            elif len(self.in_shape) == 3:
                # Image-like: [B, T, C, H, W] -> [B, T*C, H, W]
                Bsz, T, C, H, W = self.buffer.shape[:5]
                return self.buffer.reshape(Bsz, T * C, H, W)
            else:
                # Generic: flatten feature dims, concat time, then restore flat feature.
                Bsz, T = self.buffer.shape[:2]
                feat = self.buffer.shape[2:]
                return self.buffer.reshape(Bsz, T, -1).reshape(Bsz, T * int(torch.prod(torch.tensor(feat))))
        else:
            return self.buffer
