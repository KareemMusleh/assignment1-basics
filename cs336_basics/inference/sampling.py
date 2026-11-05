from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 128

    def check_params(self):
        if not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("temperature must be finite and >= 0.0")
        if type(self.top_k) is not int or (self.top_k != -1 and self.top_k < 1):
            raise ValueError("top_k must be -1 or a positive integer")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        if type(self.max_tokens) is not int or self.max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative integer")


def sampling_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """M1: [batch, vocab] logits -> float32 probabilities; see docs/inference_assignment.md.

    Do not mutate logits. Implement validation, greedy, temperature, top-k, top-p.
    """
    params.check_params()
    if logits.ndim != 2 or logits.shape[-1] == 0:
        raise ValueError("logits must have shape [batch, vocab] with a nonempty vocabulary")
    if params.top_k > logits.shape[-1]:
        raise ValueError("top_k must not exceed the vocabulary size")
    temp = params.temperature
    if temp == 0:
        indices = torch.argmax(logits, dim=-1, keepdim=True)
        filtered = torch.zeros_like(logits, dtype=torch.float32)
        filtered.scatter_(dim=-1, index=indices, value=1.0)
        return filtered

    logits = logits.float() / temp
    filtered = torch.full_like(logits, -float('inf'))
    if params.top_k != -1:
        values, indices = logits.topk(params.top_k)
        filtered.scatter_(dim=-1, index=indices, src=values)
    else:
        filtered = logits

    probs = filtered.softmax(dim=-1)
    if params.top_p != 1.0:
        values, indices = probs.sort(dim=-1, descending=True)
        cumulative_probs = values.cumsum(dim=-1)
        mask = (cumulative_probs - values) < params.top_p
        values.masked_fill_(~mask, 0.0)
        values = values / values.sum(dim=-1, keepdim=True)
        result = torch.zeros_like(probs)
        result.scatter_(dim=-1, index=indices, src=values)
        return result
    return probs

def sample_tokens(
    logits: torch.Tensor, params: SamplingParams, *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """M1: Return one int64 token per row, using the supplied RNG when sampling."""
    probs = sampling_probs(logits, params)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
