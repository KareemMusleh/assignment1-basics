import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.float()
        rms = ((x ** 2).sum(-1, keepdim=True)/self.d_model + self.eps).rsqrt()
        return ((x * rms) * self.weight).to(in_dtype)
