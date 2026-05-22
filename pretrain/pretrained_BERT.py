"""Original BERT pretraining.

Objective from BERT:
    1. masked language modeling with the 80/10/10 replacement rule
    2. next sentence prediction over sentence pairs
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from transformers import BertForPreTraining

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretrained_data_sampler import (
    BertPretrainingCollator,
    TextDataConfig,
    build_bert_pretraining_dataloaders,
    set_seed,
)
from pretrain.common import (
    OptimConfig,
    build_bert_base_config,
    device_from_config,
    make_optimizer,
    make_scheduler,
    move_to_device,
    save_checkpoint,
)


@dataclass
class BERTPretrainConfig:
    data: TextDataConfig = field(default_factory=TextDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "bert_pretraining")
    checkpoint_name: str = "bert_pretraining_latest.pt"
    mlm_probability: float = 0.15
    device: str = "auto"


class BERTPretrainer:
    def __init__(self, cfg: BERTPretrainConfig):
        self.cfg = cfg
        set_seed(cfg.data.seed)
        self.device = device_from_config(cfg.device)
        self.model = BertForPreTraining(
            build_bert_base_config(max_length=cfg.data.max_length)
        ).to(self.device)
        self.collator = BertPretrainingCollator(mlm_probability=cfg.mlm_probability)
        self.train_loader, self.validation_loader = build_bert_pretraining_dataloaders(
            cfg.data,
            batch_size=cfg.optim.batch_size,
            collate_fn=self.collator,
            num_workers=cfg.optim.num_workers,
        )
        self.optimizer = make_optimizer(self.model, cfg.optim)
        total_steps = len(self.train_loader) * cfg.optim.epochs
        if cfg.optim.max_steps > 0:
            total_steps = min(total_steps, cfg.optim.max_steps)
        self.scheduler = make_scheduler(self.optimizer, cfg.optim, total_steps)

    def train_step(self, batch: dict) -> torch.Tensor:
        batch = move_to_device(batch, self.device)
        output = self.model(**batch)
        loss = output.loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach()

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in self.validation_loader:
            batch = move_to_device(batch, self.device)
            total_loss += float(self.model(**batch).loss.item())
            total_batches += 1
        self.model.train()
        return total_loss / max(1, total_batches)

    def checkpoint_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.checkpoint_name)

    def train(self) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        step = 0
        self.model.train()
        for epoch in range(self.cfg.optim.epochs):
            pbar = tqdm(self.train_loader, desc=f"BERT MLM+NSP epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                loss = self.train_step(batch)
                step += 1
                pbar.set_postfix({"loss": f"{float(loss):.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                if self.cfg.optim.max_steps > 0 and step >= self.cfg.optim.max_steps:
                    self.save(epoch, step)
                    return

            val_loss = self.evaluate()
            print(f"epoch={epoch + 1} validation_pretraining_loss={val_loss:.4f}")
            self.save(epoch, step)

    def save(self, epoch: int, step: int) -> None:
        save_checkpoint(
            self.checkpoint_path(),
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            step,
            extra={"config": self.cfg},
        )


def main() -> None:
    trainer = BERTPretrainer(BERTPretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()
