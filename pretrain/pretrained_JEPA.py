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
    LossPlotter,
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
    checkpoint_name: str = "text_jepa_best.pt"
    latest_checkpoint_name: str = "text_jepa_latest.pt"
    loss_plot_name: str = "text_jepa_loss.png"
    plot_every: int = 10
    resume_from_latest: bool = True
    hidden_dim: int = 768
    predictor_dim: int = 384
    predictor_layers: int = 4
    predictor_heads: int = 6
    predictor_ffn_dim: int = 1536
    max_span_length: int = 5
    lambda_sent: float = 1.0
    lambda_span: float = 1.0
    ema_decay: float = 0.996
    use_span_loss: bool = True
    device: str = "auto"


class NarrowTransformerPredictor(nn.Module):
    """Small transformer predictor, separate from the BERT context encoder."""

    def __init__(
        self,
        input_dim: int,
        predictor_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, predictor_dim)
        self.position_embeddings = nn.Embedding(max_length, predictor_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, input_dim)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.input_proj(hidden) + self.position_embeddings(positions)
        padding_mask = attention_mask.eq(0)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.output_proj(self.norm(x))


class TextJEPA(nn.Module):
    def __init__(self, cfg: JEPAPretrainConfig):
        super().__init__()
        self.use_span_loss = cfg.use_span_loss
        self.context_encoder = BertModel(build_bert_base_config(max_length=cfg.data.max_length))
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self._freeze_target_encoder()

        self.predictor = NarrowTransformerPredictor(
            input_dim=cfg.hidden_dim,
            predictor_dim=cfg.predictor_dim,
            num_heads=cfg.predictor_heads,
            num_layers=cfg.predictor_layers,
            ffn_dim=cfg.predictor_ffn_dim,
            max_length=cfg.data.max_length,
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

        predicted_hidden = self.predictor(masked_hidden, batch["masked_attention_mask"])
        pred_sent = predicted_hidden[:, 0]
        clean_cls = clean_hidden[:, 0]
        target_sent = clean_cls.detach()
        sent_loss = F.mse_loss(pred_sent, target_sent)

        span_loss = predicted_hidden.new_zeros(())
        target_span = None
        pred_span = None
        if self.use_span_loss:
            pooled_span = self.span_mean_pool(clean_hidden, batch["span_mask"])
            target_span = pooled_span.detach()
            pred_span = self.span_mean_pool(predicted_hidden, batch["span_mask"])
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
        self.best_validation_loss = float("inf")
        self.start_epoch = 0
        self.global_step = 0
        self.plotter = LossPlotter(self.loss_plot_path(), "JEPA Pretraining Loss")
        if cfg.resume_from_latest:
            self.load_latest_if_available()

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

    def latest_checkpoint_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.latest_checkpoint_name)

    def loss_plot_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.loss_plot_name)

    def load_latest_if_available(self) -> None:
        path = self.latest_checkpoint_path()
        if not os.path.exists(path):
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        self.global_step = int(checkpoint.get("step", 0))
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if "plotter_state" in checkpoint:
            self.plotter.load_state_dict(checkpoint["plotter_state"])
        print(
            f"resumed_from={path} start_epoch={self.start_epoch + 1} "
            f"global_step={self.global_step} best_validation_loss={self.best_validation_loss:.4f}"
        )

    def train(self) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        step = self.global_step
        if self.cfg.optim.max_steps > 0 and step >= self.cfg.optim.max_steps:
            print(f"checkpoint already reached max_steps={self.cfg.optim.max_steps}")
            return

        self.model.train()
        for epoch in range(self.start_epoch, self.cfg.optim.epochs):
            pbar = tqdm(self.train_loader, desc=f"JEPA epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                stats = self.train_step(batch)
                step += 1
                self.plotter.add_train(step, stats["loss"])
                if step % self.cfg.plot_every == 0:
                    self.plotter.save()
                pbar.set_postfix({key: f"{value:.4f}" for key, value in stats.items()})
                if self.cfg.optim.max_steps > 0 and step >= self.cfg.optim.max_steps:
                    val = self.evaluate()
                    self.plotter.add_validation(step, val["loss"])
                    self.plotter.save()
                    saved = self.save_if_best(epoch, step, val["loss"])
                    self.save_latest(epoch, step, val["loss"])
                    print(
                        f"step={step} validation={val} "
                        f"best_validation_loss={self.best_validation_loss:.4f} saved_best={saved}"
                    )
                    return

            val = self.evaluate()
            self.plotter.add_validation(step, val["loss"])
            self.plotter.save()
            saved = self.save_if_best(epoch, step, val["loss"])
            self.save_latest(epoch, step, val["loss"])
            print(
                f"epoch={epoch + 1} validation={val} "
                f"best_validation_loss={self.best_validation_loss:.4f} saved_best={saved}"
            )

    def save_if_best(self, epoch: int, step: int, validation_loss: float) -> bool:
        if validation_loss >= self.best_validation_loss:
            return False
        self.best_validation_loss = validation_loss
        self.save(self.checkpoint_path(), epoch, step, validation_loss, checkpoint_type="best")
        return True

    def save_latest(self, epoch: int, step: int, validation_loss: float) -> None:
        self.save(self.latest_checkpoint_path(), epoch, step, validation_loss, checkpoint_type="latest")

    def save(
        self,
        path: str,
        epoch: int,
        step: int,
        validation_loss: float,
        checkpoint_type: str,
    ) -> None:
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            step,
            extra={
                "config": self.cfg,
                "validation_loss": validation_loss,
                "best_validation_loss": self.best_validation_loss,
                "checkpoint_type": checkpoint_type,
                "plotter_state": self.plotter.state_dict(),
            },
        )


def main() -> None:
    trainer = JEPAPretrainer(JEPAPretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()
