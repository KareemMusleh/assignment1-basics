"""Translate test contracts to your implementation; never put algorithms here."""
from cs336_basics.inference.generation import generate_tokens
from cs336_basics.inference.sampling import SamplingParams, sample_tokens, sampling_probs

__all__ = ['SamplingParams', 'sample_tokens', 'sampling_probs', 'generate_tokens', 'make_cache_manager']


def make_cache_manager(num_pages: int, page_size: int):
    """M4/M6: return a view of your allocator + prefix cache for upstream tests.

    Required surface: free_slots (page-aligned token offsets), _allocate(n_pages),
    _page_to_token(page_starts), check_integrity(), and
    prefix_cache.insert_prefix(input_ids, token_indices).
    Your internal implementation does not need these exact method names.
    """
    raise NotImplementedError('M4/M6: connect your cache manager in inference_tests/adapters.py')
