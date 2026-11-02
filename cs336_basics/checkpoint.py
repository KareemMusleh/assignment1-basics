import os
import random
from typing import IO, BinaryIO

import numpy as np
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    *,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    # Store NumPy's array as a list so torch.load(weights_only=True) can read it.
    numpy_state = np.random.get_state()
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    # Load on CPU; load_state_dict copies model/optimizer tensors to their devices.
    state = torch.load(src, map_location="cpu", weights_only=True)
    if lr_scheduler is not None and state["lr_scheduler"] is None:
        raise ValueError("Checkpoint has no learning-rate scheduler state")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(state["lr_scheduler"])

    rng = state["rng"]
    random.setstate(rng["python"])
    numpy_state = rng["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state["iteration"]
