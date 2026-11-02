"""Run with --config configs/tinystories.yaml --dry-run."""

import argparse
import math
from pathlib import Path

import attrs
import cattrs
import torch
import tyro
import yaml
from torch import nn
import numpy as np

from cs336_basics.nn import CrossEntropyLoss, Transformer
from cs336_basics.optim.adamw import AdamW
from cs336_basics.get_batch import get_batch
from cs336_basics.grad_clip import clip_grad
from cs336_basics.lr_scheduler import LRCosineScheduler
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint

CHECKPOINT_DIR = "checkpoints"

@attrs.define
class LMModelConfig:
    d_model: int
    vocab_size: int
    num_layers: int
    n_heads: int
    d_ff: int
    context_len: int
    theta: float | None = None

    def build_module(self, device: str | None = None, dtype: torch.dtype | None = None) -> "LMModel":
        return LMModel(self, device=device, dtype=dtype)

@attrs.define
class OptimizerConfig:
    lr: float
    weight_decay: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    def build_optimizer(self, params: nn.Module) -> torch.optim.Optimizer:
        return AdamW(params, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay, eps=self.eps)

@attrs.define
class LRSchedulerConfig:
    max_lr: float
    min_lr: float
    warmup_iters: int
    cycle_iters: int

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> LRCosineScheduler:
        return LRCosineScheduler(
            optimizer, self.max_lr, self.min_lr, self.warmup_iters, self.cycle_iters,
        )


class LMModel(nn.Module):
    def __init__(self, config: LMModelConfig, device: str | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.transformer = Transformer(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            num_layers=config.num_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            context_len=config.context_len,
            theta=config.theta,
            device=device,
            dtype=dtype,
        )
        self.loss = CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor):
        logits = self.transformer(input_ids)
        return self.loss(logits, targets)


@attrs.define
class TrainConfig:
    model: LMModelConfig
    optimizer: OptimizerConfig
    # Relative paths are resolved from the working directory.
    base_dir: Path
    train_data: Path
    lr_scheduler: LRSchedulerConfig | None = None
    valid_data: Path | None = None
    num_steps: int = 1000
    batch_size: int = 32
    eval_interval: int = attrs.field(default=100, validator=attrs.validators.gt(0))
    eval_batches: int = attrs.field(default=20, validator=attrs.validators.gt(0))
    device: str = "cuda"
    dtype: torch.dtype | None = None
    seed: int = 42
    output_dir: Path = Path("artifacts/checkpoints")
    max_grad_norm: float | None = 3.0

    save_checkpoint_interval: int = attrs.field(default=100, validator=attrs.validators.gt(0))
    resume_from: Path | None = None


    def build_trainer(self):
        return Trainer(self)

class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        assert self.config.train_data.exists()
        assert self.config.valid_data is None or self.config.valid_data.exists()

        if not self.config.base_dir.exists():
            self.config.base_dir.mkdir(parents=True)
            print(f"Created base_dir {self.config.base_dir}")
        self.trained = False

    @torch.no_grad()
    def evaluate(self, model: LMModel, valid_data: np.ndarray) -> float:
        was_training = model.training
        random_state = np.random.get_state()
        try:
            model.eval()
            np.random.seed(self.config.seed)
            total_loss = 0.0
            for _ in range(self.config.eval_batches):
                input_ids, targets = get_batch(
                    valid_data, self.config.batch_size,
                    self.config.model.context_len, self.config.device,
                )
                total_loss += model(input_ids, targets).item()
            return total_loss / self.config.eval_batches
        finally:
            np.random.set_state(random_state)
            model.train(was_training)

    def train(self):
        if self.trained:
            raise RuntimeError("Tried to train a model that is already trained")

        model: LMModel = self.config.model.build_module(device=self.config.device, dtype=self.config.dtype)
        optimizer: torch.optim.Optimizer = self.config.optimizer.build_optimizer(model.parameters())
        scheduler = (
            self.config.lr_scheduler.build_scheduler(optimizer)
            if self.config.lr_scheduler is not None else None
        )
        # Build the scheduler before restoring the optimizer: construction sets its LR.
        completed_steps = 0
        if self.config.resume_from is not None:
            completed_steps = load_checkpoint(self.config.resume_from, model, optimizer, scheduler)
            if completed_steps > self.config.num_steps:
                raise ValueError("Checkpoint step exceeds configured num_steps")
            print(f"Resuming after {completed_steps} completed steps")

        train_data = np.load(self.config.train_data, mmap_mode="r")
        valid_data = (
            np.load(self.config.valid_data, mmap_mode="r")
            if self.config.valid_data is not None else None
        )
        model.train()
        train_loss_sum = 0.0
        train_loss_count = 0

        for step in range(completed_steps, self.config.num_steps):
            input_ids, targets = get_batch(train_data, self.config.batch_size, self.config.model.context_len, self.config.device)
            loss = model(input_ids, targets)
            loss.backward()
            if self.config.max_grad_norm is not None:
                clip_grad(model.parameters(), max_l2_norm=self.config.max_grad_norm)

            loss_value = loss.item()
            train_loss_sum += loss_value
            train_loss_count += 1
            print(f"Step: {step}, Loss: {loss_value}, LR: {optimizer.param_groups[0]['lr']:.6g}")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            completed_steps = step + 1
            if valid_data is not None and (
                completed_steps % self.config.eval_interval == 0
                or completed_steps == self.config.num_steps
            ):
                valid_loss = self.evaluate(model, valid_data)
                perplexity = math.exp(valid_loss) if valid_loss < 700 else float("inf")
                print(
                    f"Validation after {completed_steps} steps: "
                    f"train_loss={train_loss_sum / train_loss_count:.4f}, "
                    f"valid_loss={valid_loss:.4f}, perplexity={perplexity:.2f}"
                )
                train_loss_sum = 0.0
                train_loss_count = 0
            if (
                completed_steps % self.config.save_checkpoint_interval == 0
                and completed_steps != self.config.num_steps
            ):
                self.save_checkpoint(model, optimizer, scheduler, completed_steps)

        self.save_checkpoint(model, optimizer, scheduler, completed_steps)
        self.trained = True

    def save_checkpoint(self, model: LMModel, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler | None, completed_steps: int):
        checkpoint_dir = self.config.base_dir/CHECKPOINT_DIR/f"step{completed_steps}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir/f"step{completed_steps}.pth"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            iteration=completed_steps,
            out=checkpoint_path,
        )


def load_config(path: Path) -> TrainConfig:
    with path.open() as f:
        values = yaml.safe_load(f)
    assert isinstance(values, dict)
    converter = cattrs.Converter(forbid_extra_keys=True)
    converter.register_structure_hook(Path, lambda value, _: Path(value))
    converter.register_structure_hook(torch.dtype, lambda value, _: getattr(torch, value))
    return converter.structure(values, TrainConfig)


def train(model: LMModel, config: TrainConfig):
    raise NotImplementedError("The configuration is ready; the training loop is not implemented yet.")


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    config = tyro.cli(TrainConfig, default=load_config(args.config), args=overrides)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if args.dry_run:
        model = config.model.build_module(device=config.device)
        print(config)
        print(f"Model constructed: {sum(p.numel() for p in model.parameters()):,} parameters")
        return
    trainer = config.build_trainer()
    trainer.train()


if __name__ == "__main__":
    main()
