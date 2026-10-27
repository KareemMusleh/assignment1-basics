import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        data = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(data, mean=0.0, std=1, a=-3, b=3)
        self.weight = torch.nn.Parameter(data)

    def forward(self, x: torch.Tensor):
        return self.weight[x]
