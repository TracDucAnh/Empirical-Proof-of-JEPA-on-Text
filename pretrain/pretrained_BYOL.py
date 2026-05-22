"""BYOL-style text pretraining.

Two independently span-masked views of a sentence are encoded by an online
network and a momentum target network.  The online predictor learns to match
the target projection of the opposite view.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretrained_data_sampler import TextDataConfig, TwoViewSpanMaskCollator, build_pretrain_dataloaders, set_seed
from pretrain.common import (
    BertSentenceEncoder,
    OptimConfig,
    PredictionMLP,
    ProjectionMLP,
    cosine_prediction_loss,
    device_from_config,
    make_optimizer,
    make_scheduler,
    move_to_device,
    save_checkpoint,
)


@dataclass
class BYOLPretrainConfig:
    data: TextDataConfig = field(default_factory=TextDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "text_byol")
    checkpoint_name: str = "text_byol_latest.pt"
    projector_hidden_dim: int = 2048
    projector_out_dim: int = 256
    predictor_hidden_dim: int = 1024
    ema_decay: float = 0.996
    max_span_length: int = 5
    device: str = "auto"


class TextBYOL(nn.Module):
    def __init__(self, cfg: BYOLPretrainConfig):
        super().__init__()
        self.online_encoder = BertSentenceEncoder(max_length=cfg.data.max_length)
        self.online_projector = ProjectionMLP(768, cfg.projector_hidden_dim, cfg.projector_out_dim)
        self.online_predictor = PredictionMLP(cfg.projector_out_dim, cfg.predictor_hidden_dim)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self._freeze_target()

    def _freeze_target(self) -> None:
        for module in [self.target_encoder, self.target_projector]:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def online_forward(self, input_ids, attention_mask, token_type_ids):
        cls = self.online_encoder(input_ids, attention_mask, token_type_ids)
        projection = self.online_projector(cls)
        prediction = self.online_predictor(projection)
        return prediction

    @torch.no_grad()
    def target_forward(self, input_ids, attention_mask, token_type_ids):
        cls = self.target_encoder(input_ids, attention_mask, token_type_ids)
        return self.target_projector(cls)

    @torch.no_grad()
    def update_target(self, decay: float) -> None:
        online_modules = [self.online_encoder, self.online_projector]
        target_modules = [self.target_encoder, self.target_projector]
        for online_module, target_module in zip(online_modules, target_modules):
            for online, target in zip(online_module.parameters(), target_module.parameters()):
                target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)

    def forward(self, batch: dict) -> dict:
        p1 = self.online_forward(
            batch["view1_input_ids"],
            batch["view1_attention_mask"],
            batch["view1_token_type_ids"],
        )
        p2 = self.online_forward(
            batch["view2_input_ids"],
            batch["view2_attention_mask"],
            batch["view2_token_type_ids"],
        )
        z1_target = self.target_forward(
            batch["view1_input_ids"],
            batch["view1_attention_mask"],
            batch["view1_token_type_ids"],
        )
        z2_target = self.target_forward(
            batch["view2_input_ids"],
            batch["view2_attention_mask"],
            batch["view2_token_type_ids"],
        )
        loss = 0.5 * (
            cosine_prediction_loss(p1, z2_target)
            + cosine_prediction_loss(p2, z1_target)
        )
        return {"loss": loss}


class BYOLPretrainer:
    def __init__(self, cfg: BYOLPretrainConfig):
        self.cfg = cfg
        set_seed(cfg.data.seed)
        self.device = device_from_config(cfg.device)
        self.model = TextBYOL(cfg).to(self.device)
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

    def train_step(self, batch: dict) -> float:
        batch = move_to_device(batch, self.device)
        output = self.model(batch)
        loss = output["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.model.update_target(self.cfg.ema_decay)
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in self.validation_loader:
            batch = move_to_device(batch, self.device)
            total_loss += float(self.model(batch)["loss"].item())
            count += 1
        self.model.train()
        return total_loss / max(1, count)

    def checkpoint_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.checkpoint_name)

    def train(self) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        step = 0
        self.model.train()
        for epoch in range(self.cfg.optim.epochs):
            pbar = tqdm(self.train_loader, desc=f"BYOL epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                loss = self.train_step(batch)
                step += 1
                pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                if self.cfg.optim.max_steps > 0 and step >= self.cfg.optim.max_steps:
                    self.save(epoch, step)
                    return

            val_loss = self.evaluate()
            print(f"epoch={epoch + 1} validation_byol_loss={val_loss:.4f}")
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
    trainer = BYOLPretrainer(BYOLPretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()
