import torch
from . import SDPA, Linear, RoPE

class CausalMHA(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads

        self.sdpa = SDPA()
        self.rope: RoPE | None = None
        if theta is not None:
            self.rope = RoPE(theta, d_model // n_heads, 2048, device=device)

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        q = self.q_proj(x).reshape(*x.shape[:-1], self.n_heads, self.d_model // self.n_heads)
        k = self.k_proj(x).reshape(*x.shape[:-1], self.n_heads, self.d_model // self.n_heads)
        v = self.v_proj(x).reshape(*x.shape[:-1], self.n_heads, self.d_model // self.n_heads)

        if token_positions is not None:
            assert self.rope is not None
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], dtype=torch.bool, device=x.device))
        sdpa_out = self.sdpa(q, k, v, mask=mask).transpose(-2, -3).reshape(x.shape)
        return self.output_proj(sdpa_out)
