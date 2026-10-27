import torch
from .linear import Linear
from .silu import SiLU

class SwiGLU(torch.nn.Module):
    def _set_intermeidate_dim(self, d_ff: int | None):
        if d_ff is None:
            # heuristic
            d_ff = round(self.d_model * 8/3)
            d_ff = (d_ff + 64 - 1) // 64 * 64

        self.d_ff = d_ff

    def __init__(self, d_model: int, d_ff: int = None, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff: int

        self._set_intermeidate_dim(d_ff)
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
