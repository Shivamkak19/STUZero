import torch
from torch import nn, Tensor
from typing import Any, List


class HistoryConcat(nn.Module):
    """
    Torch-native history concatenation utility.

    Semantics match the old HistoryWrapper:
    - Keeps the last `history_length` outputs of the previous block, including the current x.
    - If not enough past items, pads with zeros at the front.
    - Concatenates along feature dimension for batched inputs [B, D] -> [B, D*k],
      or along the only dimension for vectors [D] -> [D*k].
    - Past features are stored detached (no BPTT), current x remains attached for gradients.

    Note: If batch size changes between calls, the buffer is reset.
    """
    def __init__(self, in_dim: int, history_length: int):
        super().__init__()
        self.in_dim = in_dim
        self.history_length = int(history_length)
        self.out_dim = in_dim * self.history_length
        # Python list buffer; we only store detached tensors here
        self._buffer: List[Tensor] = []
        self._last_batch: int | None = None

    @torch.no_grad()
    def reset(self):
        self._buffer.clear()
        self._last_batch = None

    def _pad_and_stack(self, items: List[Tensor], like: Tensor) -> Tensor:
        # items already sized to <= history_length; pad at front with zeros to reach exactly history_length
        needed = self.history_length - len(items)
        if needed > 0:
            if like.dim() == 2:
                pad = [like.new_zeros(like.size(0), like.size(1)) for _ in range(needed)]  # [B, D] x needed
            elif like.dim() == 1:
                pad = [like.new_zeros(like.size(0)) for _ in range(needed)]  # [D] x needed
            else:
                raise ValueError(f"HistoryConcat expects 1D or 2D tensors, got shape {tuple(like.shape)}")
            items = [*pad, *items]
        # Now select the last exactly history_length
        items = items[-self.history_length:]

        if like.dim() == 2:
            # items: list of [B, D] -> stack along new dim to [B, Hist_length, D]
            stacked = torch.stack(items, dim=1)  # [B, Hist_length, D]
            return stacked
        else:
            # items: list of [D] -> stack along new dim to [Hist_length, D]
            stacked = torch.stack(items, dim=0)  # [Hist_length, D]
            return stacked

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() not in (1, 2):
            raise ValueError(f"HistoryConcat expects 1D or 2D input, got {x.dim()}D")

        # Reset buffer if batch size changed (for 2D inputs)
        if x.dim() == 2:
            bsz, d = x.shape
            if d != self.in_dim:
                raise ValueError(f"HistoryConcat expected feature dim {self.in_dim}, got {d}")
            if self._last_batch is None:
                self._last_batch = bsz
            elif self._last_batch != bsz:
                # batch size changed; reset to avoid shape mismatch
                self.reset()
                self._last_batch = bsz
        else:
            if x.numel() != self.in_dim:
                raise ValueError(f"HistoryConcat expected vector of dim {self.in_dim}, got {x.numel()}")

        # Compose a history list including current x, with past entries detached
        past = [t.detach() for t in self._buffer]
        hist_with_current = [*past, x]  # current x kept attached

        # Build output by padding/stacking
        y = self._pad_and_stack(hist_with_current, like=x)

        # Update buffer for the next call (store detached current x)
        with torch.no_grad():
            self._buffer.append(x.detach())
            if len(self._buffer) > self.history_length:
                self._buffer.pop(0)

        return y

