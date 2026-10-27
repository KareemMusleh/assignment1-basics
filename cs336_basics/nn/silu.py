import torch

# This is stateless. There is no reason to make it a module.

class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor):
        return x * x.sigmoid()
