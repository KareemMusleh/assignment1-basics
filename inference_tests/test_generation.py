import pytest
import torch

from .adapters import SamplingParams, generate_tokens


def test_last_position_and_full_prefix():
    calls = []

    def forward(ids):
        assert not torch.is_grad_enabled()
        calls.append(ids.clone())
        logits = torch.full((*ids.shape, 8), -100.)
        logits.scatter_(-1, ((ids + 1) % 8).unsqueeze(-1), 100.)
        return logits

    result = generate_tokens(forward, torch.tensor([1, 2]), SamplingParams(max_tokens=3), context_length=8)
    assert result.tolist() == [1, 2, 3, 4, 5]
    assert [x.tolist() for x in calls] == [[[1, 2]], [[1, 2, 3]], [[1, 2, 3, 4]]]


@pytest.mark.parametrize('ignore_eos,expected', [(False, [1, 2]), (True, [1, 2, 2, 2])])
def test_eos(ignore_eos, expected):
    def forward(ids):
        logits = torch.zeros(*ids.shape, 4)
        logits[..., 2] = 100
        return logits

    result = generate_tokens(forward, torch.tensor([1]), SamplingParams(max_tokens=3, ignore_eos=ignore_eos), context_length=4, eos_token_id=2)
    assert result.tolist() == expected


def test_zero_budget():
    def forward(_):
        pytest.fail('Zero new tokens must not invoke the model')

    assert generate_tokens(forward, torch.tensor([1]), SamplingParams(max_tokens=0), context_length=1).tolist() == [1]


def test_context_overflow():
    with pytest.raises(ValueError):
        generate_tokens(lambda _: None, torch.tensor([1, 2]), SamplingParams(max_tokens=3), context_length=4)
