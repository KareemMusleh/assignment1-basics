import torch

class Softmax(torch.nn.Module):
    def forward(self, x: torch.Tensor, dim: int):
        exp = torch.exp(x - x.amax(dim, keepdim=True))
        return exp / exp.sum(dim, keepdim=True)
