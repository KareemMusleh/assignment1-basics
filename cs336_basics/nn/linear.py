import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # store it like this for NT GEMM
        data = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2 /(in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = torch.nn.Parameter(data)

    def forward(self, x: torch.Tensor):
        return x @ self.weight.mT
