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

import copy
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import BertConfig, BertModel

# ── load .env ─────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretrained_data_sampler import JEPASpanMaskCollator, TextDataConfig, build_pretrain_dataloaders, set_seed
from pretrain.common import (
    LossPlotter,
    OptimConfig,
    device_from_config,
    make_optimizer,
    make_scheduler,
    move_to_device,
    save_checkpoint,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Encoder configs  (mirrors tjepa_architecture.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_bert_base_config(max_length: int = 256) -> BertConfig:
    return BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_length, type_vocab_size=2,
        initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0,
        position_embedding_type="absolute",
    )


def build_bert_large_config(max_length: int = 256) -> BertConfig:
    return BertConfig(
        vocab_size=30522, hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, intermediate_size=4096, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_length, type_vocab_size=2,
        initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0,
        position_embedding_type="absolute",
    )


_ENCODER_CONFIGS: dict[str, callable] = {
    "bert_base":  build_bert_base_config,
    "bert_large": build_bert_large_config,
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Config
# ══════════════════════════════════════════════════════════════════════════════

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
    # ── architecture ──────────────────────────────────────────────────────────
    model_name: str = "bert_large"   # "bert_base" | "bert_large"
    hidden_dim: int = 1024           # D: must match model_name (bert_large=1024)
    predictor_dim: int = 512         # d: recommended D/2 = 512
    predictor_layers: int = 4
    predictor_heads: int = 8         # predictor_dim (512) % heads == 0
    predictor_ffn_dim: int = 2048    # 4 * predictor_dim
    # ── masking ───────────────────────────────────────────────────────────────
    max_span_length: int = 5
    max_num_spans: int = 5
    min_num_spans: int = 5
    mask_seed: Optional[int] = None
    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_decay: float = 0.996
    device: str = "auto"
    # ── HuggingFace Hub ───────────────────────────────────────────────────────
    hf_repo_id: str = "ducanhdinh/jepa_proof_tjepa"
    push_to_hub: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Predictor  (unchanged — identical to tjepa_architecture.py)
# ══════════════════════════════════════════════════════════════════════════════

class SmallBertPredictor(nn.Module):
    """Narrow BERT predictor: D → d (bottleneck) → D.
    Input/output live in encoder space D; loss computed against target encoder.
    """

    def __init__(self, input_dim=768, predictor_dim=384, num_heads=6,
                 num_layers=4, ffn_dim=1536, max_length=256, dropout=0.1):
        super().__init__()
        if predictor_dim % num_heads != 0:
            raise ValueError(
                f"predictor_dim ({predictor_dim}) must be divisible by "
                f"num_heads ({num_heads}).")

        self.input_proj  = nn.Linear(input_dim, predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, input_dim)

        predictor_config = BertConfig(
            vocab_size=1, hidden_size=predictor_dim, num_hidden_layers=num_layers,
            num_attention_heads=num_heads, intermediate_size=ffn_dim,
            hidden_act="gelu", hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_length, type_vocab_size=2,
            initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0,
            position_embedding_type="absolute",
        )
        self.bert = BertModel(predictor_config, add_pooling_layer=False)

    def forward(self, hidden, attention_mask, token_type_ids):
        x = self.input_proj(hidden)
        x = self.bert(
            inputs_embeds=x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        ).last_hidden_state
        return self.output_proj(x)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TextJEPA  (synced với tjepa_architecture.py)
# ══════════════════════════════════════════════════════════════════════════════

class TextJEPA(nn.Module):
    """
    Text JEPA: context encoder + EMA target encoder + predictor.

    Parameters
    ──────────
    model_name       : "bert_base" (768-d, 12L) | "bert_large" (1024-d, 24L)
    hidden_dim       : must match model_name (768 or 1024)
    predictor_dim    : bottleneck dim d < D
    predictor_layers : transformer depth (< encoder num_layers)
    predictor_heads  : attention heads (predictor_dim % heads == 0)
    predictor_ffn_dim: FFN hidden dim
    max_length       : sequence length, must match dataloader
    """

    def __init__(self, cfg: JEPAPretrainConfig):
        super().__init__()

        if cfg.model_name not in _ENCODER_CONFIGS:
            raise ValueError(f"model_name '{cfg.model_name}' not recognised. "
                             f"Choose from: {list(_ENCODER_CONFIGS.keys())}")
        encoder_config = _ENCODER_CONFIGS[cfg.model_name](max_length=cfg.data.max_length)

        if cfg.hidden_dim != encoder_config.hidden_size:
            raise ValueError(
                f"hidden_dim ({cfg.hidden_dim}) must equal encoder hidden_size "
                f"({encoder_config.hidden_size}) for model_name='{cfg.model_name}'.")

        # Soft warning thay vì hard error (mirrors tjepa_architecture.py)
        recommended = encoder_config.hidden_size // 2
        if cfg.predictor_dim != recommended:
            warnings.warn(
                f"predictor_dim={cfg.predictor_dim} differs from recommended D/2={recommended}. "
                "This is allowed but may affect training dynamics.", UserWarning, stacklevel=2)

        if cfg.predictor_layers >= encoder_config.num_hidden_layers:
            raise ValueError(
                f"predictor_layers ({cfg.predictor_layers}) must be fewer than "
                f"encoder layers ({encoder_config.num_hidden_layers}).")

        self.hidden_dim      = encoder_config.hidden_size
        self.context_encoder = BertModel(encoder_config, add_pooling_layer=False)
        self.target_encoder  = copy.deepcopy(self.context_encoder)
        self._freeze_target_encoder()

        self.predictor = SmallBertPredictor(
            input_dim=self.hidden_dim,
            predictor_dim=cfg.predictor_dim,
            num_heads=cfg.predictor_heads,
            num_layers=cfg.predictor_layers,
            ffn_dim=cfg.predictor_ffn_dim,
            max_length=cfg.data.max_length,
        )

    # ── target encoder management ─────────────────────────────────────────────

    def _freeze_target_encoder(self):
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, decay: float) -> None:
        """EMA update: target ← decay·target + (1−decay)·context."""
        for ctx, tgt in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            tgt.data.mul_(decay).add_(ctx.data, alpha=1.0 - decay)

    # ── encoder helpers ───────────────────────────────────────────────────────

    def _encode(self, encoder, input_ids, attention_mask, token_type_ids):
        """Run encoder, return last_hidden_state [B, L, D]."""
        return encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, return_dict=True,
        ).last_hidden_state

    def encode_full_sequence(self, batch: dict, use_target: bool = False) -> torch.Tensor:
        """
        Encode the CLEAN (unmasked) sentence through context or target encoder.
        Returns all token embeddings [B, L, D] — mirrors I-JEPA's forward_all_patches().
        Used by training script for fair effective-rank computation.

        Parameters
        ----------
        batch      : 7-key batch dict from tjepa_dataloader
        use_target : if True, use target encoder (no grad); else context encoder
        """
        encoder = self.target_encoder if use_target else self.context_encoder
        return self._encode(
            encoder,
            batch["clean_input_ids"],
            batch["clean_attention_mask"],
            batch["clean_token_type_ids"],
        )

    # ── loss ──────────────────────────────────────────────────────────────────

    @staticmethod
    def span_jepa_loss(pred, target, span_mask):
        """
        Token-level L2 loss over span positions only, averaged over span tokens.
        Mirrors I-JEPA eq: (1/M) Σ_i Σ_{j∈B_i} ‖pred_j − target_j‖²
        """
        l2_per_token = ((pred - target) ** 2).sum(dim=-1)
        masked        = l2_per_token * span_mask.float()
        n_span_tokens = span_mask.float().sum().clamp(min=1.0)
        return masked.sum() / n_span_tokens

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        """
        Accepts 7-key batch dict from tjepa_dataloader.JEPASpanMaskCollator.

        Returns
        ───────
        dict:
            predicted_hidden  [B, L, D]
            target_hidden     [B, L, D]
            span_loss         scalar
        """
        context_hidden = self._encode(
            self.context_encoder,
            batch["masked_input_ids"],
            batch["masked_attention_mask"],
            batch["masked_token_type_ids"],
        )

        with torch.no_grad():
            target_hidden = self._encode(
                self.target_encoder,
                batch["clean_input_ids"],
                batch["clean_attention_mask"],
                batch["clean_token_type_ids"],
            )

        predicted_hidden = self.predictor(
            context_hidden,
            batch["masked_attention_mask"],
            batch["masked_token_type_ids"],
        )

        span_loss = self.span_jepa_loss(
            predicted_hidden, target_hidden.detach(), batch["span_mask"])

        return dict(
            predicted_hidden = predicted_hidden,
            target_hidden    = target_hidden,
            span_loss        = span_loss,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HuggingFace Hub push
# ══════════════════════════════════════════════════════════════════════════════

def push_context_encoder_to_hub(
    model: TextJEPA,
    repo_id: str,
    save_dir: str,
    token: str | None = None,
) -> None:
    """
    Lưu context_encoder ra disk rồi push lên HuggingFace Hub.

    Parameters
    ----------
    model    : TextJEPA instance đã train
    repo_id  : e.g. "ducanhdinh/jepa_proof_tjepa"
    save_dir : thư mục tạm để lưu weights trước khi push
    token    : HF token (lấy từ .env nếu None)
    """
    _token = token or HF_TOKEN
    if not _token:
        raise RuntimeError(
            "HF_TOKEN không tìm thấy. "
            "Thêm HF_TOKEN=<token> vào file .env hoặc truyền trực tiếp.")

    encoder_dir = os.path.join(save_dir, "context_encoder")
    os.makedirs(encoder_dir, exist_ok=True)

    # Lưu weights + config của context_encoder
    model.context_encoder.save_pretrained(encoder_dir)
    print(f"Đã lưu context_encoder vào {encoder_dir}")

    # Push lên Hub
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=_token, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=encoder_dir,
        repo_id=repo_id,
        repo_type="model",
        token=_token,
    )
    print(f"Đã push context_encoder lên https://huggingface.co/{repo_id}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Pretrainer
# ══════════════════════════════════════════════════════════════════════════════

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
                    if self.cfg.push_to_hub:
                        push_context_encoder_to_hub(
                            self.model, self.cfg.hf_repo_id, self.cfg.output_dir)
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

        # Push sau khi train xong toàn bộ
        if self.cfg.push_to_hub:
            push_context_encoder_to_hub(
                self.model, self.cfg.hf_repo_id, self.cfg.output_dir)

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


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    trainer = JEPAPretrainer(JEPAPretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()