import torch.nn as nn

import torch
from torch import nn, Tensor
from typing import Any, List

class HistoryConcat(nn.Module):
    """
    Simplified history concatenation utility.

    Keeps track of the last N inputs to the layer and concatenates them:
    - Stores the last `history_length` inputs (including current x)
    - If not enough past items, pads with zeros at the front
    - Concatenates along dim=1 (channel/feature dimension)
    - For 2D inputs [B, D]: concatenates to [B, D*history_length]
    - For 4D inputs [B, C, H, W]: concatenates to [B, C*history_length, H, W]
    - Past features are stored detached (no BPTT), current x remains attached
    """
    def __init__(self, in_dim: int, history_length: int):
        super().__init__()
        self.in_dim = in_dim
        self.history_length = int(history_length)
        self.out_dim = in_dim * self.history_length
        # Python list buffer; we only store detached tensors here
        self._buffer: List[Tensor] = []

    @torch.no_grad()
    def reset(self):
        self._buffer.clear()

    def forward(self, x: Tensor) -> Tensor:
        # Compose a history list including current x, with past entries detached
        past = [t.detach() for t in self._buffer]
        hist_with_current = [*past, x]  # current x kept attached

        # Pad with zeros at the front if we don't have enough history
        needed = self.history_length - len(hist_with_current)
        if needed > 0:
            zero_pad = x.new_zeros(x.shape)
            hist_with_current = [zero_pad] * needed + hist_with_current

        # Take only the last history_length items
        hist_with_current = hist_with_current[-self.history_length:]

        # Concatenate along dim=1 (channel/feature dimension)
        y = torch.cat(hist_with_current, dim=1)

        # Update buffer for the next call (store detached current x)
        with torch.no_grad():
            self._buffer.append(x.detach())
            if len(self._buffer) > self.history_length:
                self._buffer.pop(0)

        return y

