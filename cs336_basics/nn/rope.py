import torch
import dataclasses

@dataclasses.dataclass
class RoPECache:
    cos: torch.Tensor
    sin: torch.Tensor

type RoPECacheKey = tuple[float, int, int, torch.device]

class RoPE(torch.nn.Module):
    cache : dict[RoPECacheKey, RoPECache] = {}
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device or torch.get_default_device()
        self._prepare_buffers()

    def _get_cache_key(self) -> RoPECacheKey:
        return (self.theta, self.d_k, self.max_seq_len, self.device)

    def _prepare_buffers(self):
        cache_key = self._get_cache_key()
        if cache_key in RoPE.cache:
            return
        with torch.device(self.device):
            half_d = self.d_k / 2
            angle = self.theta ** (torch.arange(0, half_d, dtype=torch.float32) / half_d)[None, :]
            angle = torch.arange(0, self.max_seq_len, dtype=torch.float32)[:, None] / angle

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        RoPE.cache[cache_key] = RoPECache(cos=cos, sin=sin)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert RoPE.cache
        seq_len = x.shape[-3]
        token_positions = token_positions[:seq_len]

        in_dtype = x.dtype
        x = x.to(torch.float32)
        cache_key = self._get_cache_key()
        cache = RoPE.cache[cache_key]

        curr_cos = cache.cos[token_positions]
        curr_sin = cache.sin[token_positions]
        if len(x.shape) == 4:
            curr_cos = curr_cos.unsqueeze(-2)
            curr_sin = curr_sin.unsqueeze(-2)
        elif len(x.shape) > 4:
            raise RuntimeError()

        first_half = curr_cos * x[..., :, ::2] - curr_sin * x[..., :, 1::2]
        second_half = curr_sin * x[..., :, ::2] + curr_cos * x[..., :, 1::2]
        return torch.stack((first_half, second_half), dim=-1).reshape(x.shape).to(in_dtype)
