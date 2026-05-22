"""VICReg text pretraining.

Two masked text views are encoded by a shared BERT encoder.  VICReg combines
invariance, variance, and covariance terms to align views while preventing
feature collapse without negative samples.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretrained_data_sampler import TextDataConfig, TwoViewSpanMaskCollator, build_pretrain_dataloaders, set_seed
from pretrain.common import (
    BertSentenceEncoder,
    OptimConfig,
    ProjectionMLP,
    device_from_config,
    make_optimizer,
    make_scheduler,
    move_to_device,
    save_checkpoint,
    vicreg_covariance_loss,
    vicreg_variance_loss,
)


@dataclass
class VICRegPretrainConfig:
    data: TextDataConfig = field(default_factory=TextDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "text_vicreg")
    checkpoint_name: str = "text_vicreg_latest.pt"
    projector_hidden_dim: int = 2048
    projector_out_dim: int = 256
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    max_span_length: int = 5
    device: str = "auto"


class TextVICReg(nn.Module):
    def __init__(self, cfg: VICRegPretrainConfig):
        super().__init__()
        self.encoder = BertSentenceEncoder(max_length=cfg.data.max_length)
        self.projector = ProjectionMLP(768, cfg.projector_hidden_dim, cfg.projector_out_dim)

    def forward_view(self, input_ids, attention_mask, token_type_ids):
        cls = self.encoder(input_ids, attention_mask, token_type_ids)
        return self.projector(cls)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = self.forward_view(
            batch["view1_input_ids"],
            batch["view1_attention_mask"],
            batch["view1_token_type_ids"],
        )
        z2 = self.forward_view(
            batch["view2_input_ids"],
            batch["view2_attention_mask"],
            batch["view2_token_type_ids"],
        )
        return z1, z2


class VICRegPretrainer:
    def __init__(self, cfg: VICRegPretrainConfig):
        self.cfg = cfg
        set_seed(cfg.data.seed)
        self.device = device_from_config(cfg.device)
        self.model = TextVICReg(cfg).to(self.device)
        collator = TwoViewSpanMaskCollator(max_span_length=cfg.max_span_length, seed=cfg.data.seed)
        self.train_loader, self.validation_loader = build_pretrain_dataloaders(
            cfg.data,
            batch_size=cfg.optim.batch_size,
            collate_fn=collator,
            num_workers=cfg.optim.num_workers,
        )
        self.optimizer = make_optimizer(self.model, cfg.optim)
        total_steps = len(self.train_loader) * cfg.optim.epochs
        if cfg.optim.max_steps > 0:
            total_steps = min(total_steps, cfg.optim.max_steps)
        self.scheduler = make_scheduler(self.optimizer, cfg.optim, total_steps)

    def objective(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, dict]:
        invariance = F.mse_loss(z1, z2)
        variance = vicreg_variance_loss(z1) + vicreg_variance_loss(z2)
        covariance = vicreg_covariance_loss(z1) + vicreg_covariance_loss(z2)
        loss = (
            self.cfg.sim_coeff * invariance
            + self.cfg.std_coeff * variance
            + self.cfg.cov_coeff * covariance
        )
        stats = {
            "loss": float(loss.detach()),
            "inv": float(invariance.detach()),
            "var": float(variance.detach()),
            "cov": float(covariance.detach()),
        }
        return loss, stats

    def train_step(self, batch: dict) -> dict:
        batch = move_to_device(batch, self.device)
        z1, z2 = self.model(batch)
        loss, stats = self.objective(z1, z2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return stats

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        totals = {"loss": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0}
        count = 0
        for batch in self.validation_loader:
            batch = move_to_device(batch, self.device)
            z1, z2 = self.model(batch)
            _, stats = self.objective(z1, z2)
            for key in totals:
                totals[key] += stats[key]
            count += 1
        self.model.train()
        return {key: value / max(1, count) for key, value in totals.items()}

    def checkpoint_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.checkpoint_name)

    def train(self) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        step = 0
        self.model.train()
        for epoch in range(self.cfg.optim.epochs):
            pbar = tqdm(self.train_loader, desc=f"VICReg epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                stats = self.train_step(batch)
                step += 1
                pbar.set_postfix({key: f"{value:.4f}" for key, value in stats.items()})
                if self.cfg.optim.max_steps > 0 and step >= self.cfg.optim.max_steps:
                    self.save(epoch, step)
                    return

            val = self.evaluate()
            print(f"epoch={epoch + 1} validation={val}")
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
    trainer = VICRegPretrainer(VICRegPretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()
