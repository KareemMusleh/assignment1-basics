from collections.abc import Callable

import torch

from .sampling import SamplingParams, sample_tokens


def generate_tokens(
    forward: Callable[[torch.Tensor], torch.Tensor],
    prompt: torch.Tensor,
    params: SamplingParams,
    *,
    context_length: int,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """M1: Single-request uncached reference generation.

    prompt: nonempty int64 [prompt_length]. forward: [1, length] -> [1, length, vocab].
    Return prompt plus generated IDs on the input device; include a generated EOS.
    Reject prompt_length + max_tokens > context_length. Disable gradients.
    The caller owns model.eval() and device placement. Do not crop the prompt.
    """
    len_prompt = min(prompt.shape[0] + params.max_tokens, context_length)
    prompt.unsqueeze_(0)
    for _ in range(len_prompt):
        logits = forward(prompt)
        assert logits is not None
        last_logits = logits[..., -1, :]
        new_token = sample_tokens(last_logits, params, generator=generator)
        prompt = torch.cat([prompt, new_token], dim=-1)
        if not params.ignore_eos:
            if new_token.item() == eos_token_id:
                break
    return prompt
