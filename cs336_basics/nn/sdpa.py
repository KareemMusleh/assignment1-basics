import torch
from .softmax import Softmax

class SDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        scale = q.shape[-1] ** -0.5
        scores: torch.Tensor = (q @ k.mT) * scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))
        probs = self.softmax(scores, dim=-1)
        return probs @ v
