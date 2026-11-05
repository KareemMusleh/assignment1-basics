import pytest
import torch

from .adapters import SamplingParams, sample_tokens, sampling_probs


def test_greedy_and_ties():
    logits = torch.tensor([[1., 4., 2.], [3., 3., 0.]])
    actual = sampling_probs(logits, SamplingParams())
    torch.testing.assert_close(actual, torch.tensor([[0., 1., 0.], [1., 0., 0.]]))


def test_temperature_and_no_mutation():
    logits = torch.tensor([[1000., 1002., 998.]], dtype=torch.float64)
    original = logits.clone()
    actual = sampling_probs(logits, SamplingParams(temperature=2.0))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.softmax(torch.tensor([[0., 1., -1.]]), -1))
    assert torch.equal(logits, original)


def test_top_k():
    logits = torch.log(torch.tensor([[0.1, 0.6, 0.3]]))
    probs = sampling_probs(logits, SamplingParams(temperature=1., top_k=2))
    torch.testing.assert_close(probs, torch.tensor([[0., 2/3, 1/3]]))


def test_top_p_keeps_crossing_token():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    probs = sampling_probs(logits, SamplingParams(temperature=1., top_p=0.8))
    torch.testing.assert_close(probs, torch.tensor([[2/3, 1/3, 0.]]))


def test_top_p_always_keeps_one_token():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    probs = sampling_probs(logits, SamplingParams(temperature=1., top_p=0.01))
    torch.testing.assert_close(probs, torch.tensor([[1., 0., 0.]]))


@pytest.mark.parametrize('top_k', [-1, 2])
def test_top_p_preserves_token_ids(top_k):
    # Each row has a different ranking; outputs must remain in vocabulary order.
    logits = torch.log(torch.tensor([[0.1, 0.6, 0.3], [0.3, 0.1, 0.6]]))
    probs = sampling_probs(logits, SamplingParams(temperature=1., top_k=top_k, top_p=0.8))
    expected = torch.tensor([[0., 2/3, 1/3], [1/3, 0., 2/3]])
    torch.testing.assert_close(probs, expected)


def test_top_k_then_top_p():
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]]))
    probs = sampling_probs(logits, SamplingParams(temperature=1., top_k=2, top_p=0.5))
    torch.testing.assert_close(probs, torch.tensor([[1., 0., 0., 0.]]))


@pytest.mark.parametrize('kwargs', [
    dict(temperature=-1), dict(temperature=float('nan')), dict(temperature=float('inf')),
    dict(top_k=0), dict(top_k=-2), dict(top_k=4), dict(top_k=1.5), dict(top_k=True),
    dict(top_p=0), dict(top_p=-0.1), dict(top_p=1.1),
    dict(top_p=float('nan')), dict(top_p=float('inf')),
    dict(max_tokens=-1), dict(max_tokens=1.5), dict(max_tokens=True),
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        sampling_probs(torch.zeros(1, 3), SamplingParams(**kwargs))


@pytest.mark.parametrize('temperature', [0., 1.])
@pytest.mark.parametrize('top_k', [-1, 1, 3])
def test_valid_parameter_boundaries(temperature, top_k):
    probs = sampling_probs(
        torch.tensor([[0., 1., 2.]]),
        SamplingParams(temperature=temperature, top_k=top_k, top_p=1., max_tokens=0),
    )
    assert probs.shape == (1, 3)
    torch.testing.assert_close(probs.sum(-1), torch.ones(1))


@pytest.mark.parametrize('shape', [(3,), (1, 2, 3), (1, 0)])
def test_invalid_logits_shape(shape):
    with pytest.raises(ValueError, match='logits'):
        sampling_probs(torch.zeros(shape), SamplingParams())


def test_sampling_rng_shape_and_support():
    logits = torch.tensor([[0., 3., 2.]]).repeat(128, 1)
    params = SamplingParams(temperature=1., top_k=2)
    first = sample_tokens(logits, params, generator=torch.Generator().manual_seed(42))
    second = sample_tokens(logits, params, generator=torch.Generator().manual_seed(42))
    assert first.shape == (128,) and first.dtype == torch.int64
    assert torch.equal(first, second)
    assert set(first.tolist()) == {1, 2}
