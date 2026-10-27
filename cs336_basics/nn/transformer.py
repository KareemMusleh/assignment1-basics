import torch
from . import TransformerBlock, Embedding, RMSNorm, Linear

class Transformer(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size: int, num_layers: int, n_heads: int, d_ff: int, context_len: int, theta: float | None = None, device=None, dtype=None):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        token_positions = torch.arange(context_len, device=device, dtype=torch.int32)
        self.register_buffer("token_positions", token_positions, persistent=False)

        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model, n_heads, d_ff, context_len, theta, device=device, dtype=dtype))
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions=self.token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
