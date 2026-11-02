import numpy.typing as npt
import numpy as np
import torch

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    indices = starts[:, None] + np.arange(context_length)
    input_ids = dataset[indices]
    targets = dataset[indices + 1]

    input_ids = torch.from_numpy(input_ids).to(device=device, dtype=torch.long).reshape(batch_size, context_length)
    targets = torch.from_numpy(targets).to(device=device, dtype=torch.long).reshape(batch_size, context_length)
    assert len(targets) == len(input_ids)
    return input_ids, targets
