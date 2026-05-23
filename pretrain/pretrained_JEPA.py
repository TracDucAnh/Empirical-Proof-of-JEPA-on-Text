"""Text JEPA pretraining — architecture faithful to I-JEPA (arXiv 2301.08243).

Key design decisions mirroring I-JEPA:
  - Predictor: narrow BERT with input_proj (D→d) and output_proj (d→D).
    Input and output are both in encoder space D; d=384 is an internal bottleneck.
  - Target: raw target-encoder output [B, L, D], NO projection.
  - Loss: token-level L2 over span positions, averaged over spans (not mean-pooled first).
    Formula matches paper: (1/M) * sum_i sum_{j in B_i} ||pred_j - target_j||^2
  - Target encoder updated by EMA only, no gradients.
"""

from __future__ import annotations

import os
import copy
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertConfig, BertModel

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
    data: TextDataConfig = field(default_factory=lambda: TextDataConfig(max_length=256))
    optim: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "text_jepa")
    checkpoint_name: str = "text_jepa_best.pt"
    latest_checkpoint_name: str = "text_jepa_latest.pt"
    loss_plot_name: str = "text_jepa_loss.png"
    plot_every: int = 10
    resume_from_latest: bool = True
    hidden_dim: int = 768        # D: encoder hidden size
    predictor_dim: int = 384     # d: predictor internal bottleneck
    predictor_layers: int = 4
    predictor_heads: int = 6
    predictor_ffn_dim: int = 1536
    max_span_length: int = 5
    max_num_spans: int = 5
    min_num_spans: int = 5
    mask_seed: Optional[int] = None
    ema_decay: float = 0.996
    device: str = "auto"


class SmallBertPredictor(nn.Module):
    """Narrow BERT predictor that mirrors the I-JEPA predictor design.

    Data flow:
        [B, L, D]
          → input_proj  Linear(D → d)
          → BERT layers (hidden dim = d)
          → output_proj Linear(d → D)
        [B, L, D]

    The bottleneck at d forces the predictor to compress context information,
    preventing it from trivially copying the encoder output.
    Input and output both live in encoder space D, so the loss is computed
    directly against the (unprojected) target encoder output.
    """

    def __init__(
        self,
        input_dim: int,       # D
        predictor_dim: int,   # d
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if predictor_dim % num_heads != 0:
            raise ValueError("predictor_dim must be divisible by predictor_heads.")

        self.input_proj = nn.Linear(input_dim, predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, input_dim)  # d → D

        predictor_config = BertConfig(
            vocab_size=1,
            hidden_size=predictor_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=ffn_dim,
            hidden_act="gelu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_length,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
        )
        self.bert = BertModel(predictor_config, add_pooling_layer=False)

    def forward(
        self,
        hidden: torch.Tensor,          # [B, L, D]
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:                 # [B, L, D]
        x = self.input_proj(hidden)    # [B, L, d]
        x = self.bert(
            inputs_embeds=x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state            # [B, L, d]
        return self.output_proj(x)     # [B, L, D]


class TextJEPA(nn.Module):
    def __init__(self, cfg: JEPAPretrainConfig):
        super().__init__()
        encoder_config = build_bert_base_config(max_length=cfg.data.max_length)
        if cfg.hidden_dim != encoder_config.hidden_size:
            raise ValueError("hidden_dim must match the BERT encoder hidden size.")
        if cfg.predictor_dim != encoder_config.hidden_size // 2:
            raise ValueError("predictor_dim must be D/2, where D is the BERT encoder hidden size.")
        if cfg.predictor_layers >= encoder_config.num_hidden_layers:
            raise ValueError("predictor_layers must be fewer than the BERT encoder layers.")

        self.context_encoder = BertModel(encoder_config, add_pooling_layer=False)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self._freeze_target_encoder()

        self.predictor = SmallBertPredictor(
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
    def span_jepa_loss(
        pred: torch.Tensor,       # [B, L, D]
        target: torch.Tensor,     # [B, L, D]
        span_mask: torch.Tensor,  # [B, L]  binary, 1 at span token positions
    ) -> torch.Tensor:
        """Token-level L2 loss over span positions, averaged over spans.

        Mirrors I-JEPA paper eq: (1/M) * sum_i sum_{j in B_i} ||pred_j - target_j||^2_2

        Here M*|B_i| is approximated by the total number of span tokens (sum of span_mask),
        so we compute mean L2 per span token — equivalent to the paper when spans are
        equal size, and a sensible normalisation otherwise.

        We do NOT pool tokens before computing loss (that would destroy per-token signal).
        """
        # squared L2 per token per dim: [B, L, D] -> [B, L]
        l2_per_token = ((pred - target) ** 2).sum(dim=-1)
        # zero out non-span positions
        masked = l2_per_token * span_mask.float()
        # mean over span tokens (total span tokens across batch)
        n_span_tokens = span_mask.float().sum().clamp(min=1.0)
        return masked.sum() / n_span_tokens

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
        return output.last_hidden_state  # [B, L, D]

    @torch.no_grad()
    def update_target_encoder(self, decay: float) -> None:
        for context, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(decay).add_(context.data, alpha=1.0 - decay)

    def forward(self, batch: dict) -> dict:
        # Context encoder sees the span-masked sentence
        masked_hidden = self.encode(
            self.context_encoder,
            batch["masked_input_ids"],
            batch["masked_attention_mask"],
            batch["masked_token_type_ids"],
        )  # [B, L, D]

        # Target encoder sees the clean sentence — no gradients, no projection
        with torch.no_grad():
            clean_hidden = self.encode(
                self.target_encoder,
                batch["clean_input_ids"],
                batch["clean_attention_mask"],
                batch["clean_token_type_ids"],
            )  # [B, L, D]

        # Predictor: D → d (internal) → D
        predicted_hidden = self.predictor(
            masked_hidden,
            batch["masked_attention_mask"],
            batch["masked_token_type_ids"],
        )  # [B, L, D]

        # Loss: token-level L2 on span positions, target is raw encoder output (D)
        span_loss = self.span_jepa_loss(
            predicted_hidden,
            clean_hidden.detach(),
            batch["span_mask"],
        )

        return {
            "predicted_hidden": predicted_hidden,
            "target_hidden": clean_hidden,
            "span_loss": span_loss,
        }


class JEPAPretrainer:
    def __init__(self, cfg: JEPAPretrainConfig):
        self.cfg = cfg
        set_seed(cfg.data.seed)
        self.device = device_from_config(cfg.device)
        self.model = TextJEPA(cfg).to(self.device)
        train_collator = JEPASpanMaskCollator(
            max_span_length=cfg.max_span_length,
            max_num_spans=cfg.max_num_spans,
            min_num_spans=cfg.min_num_spans,
            seed=cfg.mask_seed,
        )
        validation_collator = JEPASpanMaskCollator(
            max_span_length=cfg.max_span_length,
            max_num_spans=cfg.max_num_spans,
            min_num_spans=cfg.min_num_spans,
            seed=cfg.mask_seed,
        )
        self.train_loader, self.validation_loader = build_pretrain_dataloaders(
            cfg.data,
            batch_size=cfg.optim.batch_size,
            collate_fn=train_collator,
            validation_collate_fn=validation_collator,
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
        loss = output["span_loss"]
        stats = {
            "loss": float(loss.detach()),
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
        totals = {"loss": 0.0, "span": 0.0}
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