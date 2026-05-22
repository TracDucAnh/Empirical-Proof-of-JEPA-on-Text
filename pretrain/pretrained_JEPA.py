"""Text JEPA pretraining.

This script keeps JEPA separate from other anti-collapse methods.  A context
encoder sees a span-masked sentence, a momentum target encoder sees the clean
sentence, and a predictor maps context latents to target latents.  The target
encoder is updated by EMA and receives no gradients.
"""

from __future__ import annotations

import os
import copy
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretrained_data_sampler import JEPASpanMaskCollator, TextDataConfig, build_pretrain_dataloaders, set_seed
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
class JEPAPretrainConfig:
    data: TextDataConfig = field(default_factory=TextDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "text_jepa")
    checkpoint_name: str = "text_jepa_latest.pt"
    hidden_dim: int = 768
    max_span_length: int = 5
    lambda_sent: float = 1.0
    lambda_span: float = 1.0
    ema_decay: float = 0.996
    use_span_loss: bool = True
    device: str = "auto"


class TextJEPA(nn.Module):
    def __init__(self, cfg: JEPAPretrainConfig):
        super().__init__()
        self.use_span_loss = cfg.use_span_loss
        self.context_encoder = BertModel(build_bert_base_config(max_length=cfg.data.max_length))
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self._freeze_target_encoder()

        self.sent_predictor = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        if cfg.use_span_loss:
            self.span_predictor = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )

    def _freeze_target_encoder(self) -> None:
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

    @staticmethod
    def span_mean_pool(hidden: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
        mask = span_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def encode(
        self,
        encoder: BertModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return output.last_hidden_state

    @torch.no_grad()
    def update_target_encoder(self, decay: float) -> None:
        for context, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(decay).add_(context.data, alpha=1.0 - decay)

    def forward(self, batch: dict) -> dict:
        masked_hidden = self.encode(
            self.context_encoder,
            batch["masked_input_ids"],
            batch["masked_attention_mask"],
            batch["masked_token_type_ids"],
        )
        with torch.no_grad():
            clean_hidden = self.encode(
                self.target_encoder,
                batch["clean_input_ids"],
                batch["clean_attention_mask"],
                batch["clean_token_type_ids"],
            )

        masked_cls = masked_hidden[:, 0]
        clean_cls = clean_hidden[:, 0]
        target_sent = clean_cls.detach()
        pred_sent = self.sent_predictor(masked_cls)
        sent_loss = F.mse_loss(pred_sent, target_sent)

        span_loss = masked_cls.new_zeros(())
        target_span = None
        pred_span = None
        if self.use_span_loss:
            pooled_span = self.span_mean_pool(clean_hidden, batch["span_mask"])
            target_span = pooled_span.detach()
            pred_span = self.span_predictor(masked_cls)
            span_loss = F.mse_loss(pred_span, target_span)

        return {
            "target_sent": target_sent,
            "pred_sent": pred_sent,
            "target_span": target_span,
            "pred_span": pred_span,
            "sent_loss": sent_loss,
            "span_loss": span_loss,
        }


class JEPAPretrainer:
    def __init__(self, cfg: JEPAPretrainConfig):
        self.cfg = cfg
        set_seed(cfg.data.seed)
        self.device = device_from_config(cfg.device)
        self.model = TextJEPA(cfg).to(self.device)
        collator = JEPASpanMaskCollator(max_span_length=cfg.max_span_length, seed=cfg.data.seed)
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

    def objective(self, output: dict) -> tuple[torch.Tensor, dict]:
        loss = self.cfg.lambda_sent * output["sent_loss"]
        if self.cfg.use_span_loss:
            loss = loss + self.cfg.lambda_span * output["span_loss"]
        stats = {
            "loss": float(loss.detach()),
            "sent": float(output["sent_loss"].detach()),
            "span": float(output["span_loss"].detach()),
        }
        return loss, stats

    def train_step(self, batch: dict) -> dict:
        batch = move_to_device(batch, self.device)
        output = self.model(batch)
        loss, stats = self.objective(output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.model.update_target_encoder(self.cfg.ema_decay)
        return stats

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        totals = {"loss": 0.0, "sent": 0.0, "span": 0.0}
        count = 0
        for batch in self.validation_loader:
            batch = move_to_device(batch, self.device)
            output = self.model(batch)
            _, stats = self.objective(output)
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
            pbar = tqdm(self.train_loader, desc=f"JEPA epoch {epoch + 1}", ncols=120)
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
    trainer = JEPAPretrainer(JEPAPretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()
