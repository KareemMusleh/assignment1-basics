import io
import random

import attrs
import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.train import LMModelConfig, LRSchedulerConfig, OptimizerConfig, TrainConfig


def test_checkpoint_stream_and_rng():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    stream = io.BytesIO()
    save_checkpoint(model, optimizer, 7, stream)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    stream.seek(0)
    assert load_checkpoint(stream, model, optimizer) == 7
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    assert not stream.closed


def test_training_resume_matches_uninterrupted(tmp_path):
    data_path = tmp_path / 'tokens.npy'
    np.save(data_path, np.arange(100, dtype=np.uint16) % 16)
    config = TrainConfig(
        model=LMModelConfig(8, 16, 1, 2, 16, 4, 10000.0),
        optimizer=OptimizerConfig(0.001, 0.01),
        lr_scheduler=LRSchedulerConfig(0.001, 0.0001, 1, 4),
        base_dir=tmp_path / 'full', train_data=data_path, valid_data=data_path,
        num_steps=4, batch_size=2, device='cpu', eval_interval=2, eval_batches=1,
        save_checkpoint_interval=2,
    )

    def run(cfg):
        torch.manual_seed(42)
        np.random.seed(42)
        cfg.build_trainer().train()
        return cfg.base_dir / 'checkpoints' / f'step{cfg.num_steps}' / f'step{cfg.num_steps}.pth'

    full_path = run(config)
    half_path = run(attrs.evolve(config, base_dir=tmp_path / 'half', num_steps=2))
    resumed_path = run(attrs.evolve(config, base_dir=tmp_path / 'resumed', resume_from=half_path))
    full = torch.load(full_path, weights_only=True)
    resumed = torch.load(resumed_path, weights_only=True)
    for name, value in full['model'].items():
        assert torch.equal(value, resumed['model'][name]), name
    assert full['lr_scheduler'] == resumed['lr_scheduler']
    assert full['iteration'] == resumed['iteration'] == 4
    for param_id, state in full['optimizer']['state'].items():
        for name, value in state.items():
            actual = resumed['optimizer']['state'][param_id][name]
            assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
