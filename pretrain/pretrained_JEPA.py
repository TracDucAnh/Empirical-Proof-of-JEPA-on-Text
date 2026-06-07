"""Text JEPA pretraining — architecture faithful to I-JEPA (arXiv 2301.08243).

Key design decisions mirroring I-JEPA:
  - Predictor: narrow BERT with input_proj (D→d) and output_proj (d→D).
    Input and output are both in encoder space D; d=384 is an internal bottleneck.
  - Target: raw target-encoder output [B, L, D], NO projection.
  - Loss: squared L2 distance is computed for each masked token, then averaged
    over all masked tokens: (1/M) * sum_j ||pred_j - target_j||^2.
  - Target encoder updated by EMA only, no gradients.

Optimizer & LR scheduler — synced with tjepa_training.py:
  - Manual cosine LR schedule: linear warmup (warmup_epochs) → cosine decay
      start_lr  →  peak_lr  →  final_lr
  - Manual cosine weight-decay schedule: weight_decay → final_weight_decay
  - Both schedules pre-computed as numpy arrays; applied per-step via
    param_group assignment (identical to tjepa_training.py's approach).
  - make_scheduler() from common is NO LONGER used.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

from pretrained_data_sampler import (
    JEPASpanMaskCollator,
    TextDataConfig,
    build_pretrain_dataloaders,
    set_seed,
)
from pretrain.common import (
    LossPlotter,
    OptimConfig,
    device_from_config,
    move_to_device,
)

# BERT base supports up to 512 positions; hard-coded here because
# common.py does not export this constant.
BERT_BASE_MAX_POSITION_EMBEDDINGS: int = 512


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Encoder configs
# ══════════════════════════════════════════════════════════════════════════════

def build_bert_base_config(max_length: int = 256) -> BertConfig:
    return BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=max(BERT_BASE_MAX_POSITION_EMBEDDINGS, max_length),
        type_vocab_size=2,
        initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0,
        position_embedding_type="absolute",
    )


def build_bert_large_config(max_length: int = 256) -> BertConfig:
    return BertConfig(
        vocab_size=30522, hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, intermediate_size=4096, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=max(BERT_BASE_MAX_POSITION_EMBEDDINGS, max_length),
        type_vocab_size=2,
        initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0,
        position_embedding_type="absolute",
    )


_ENCODER_CONFIGS: dict[str, callable] = {
    "bert_base":  build_bert_base_config,
    "bert_large": build_bert_large_config,
}


# ══════════════════════════════════════════════════════════════════════════════
# 2b.  Checkpoint helpers  (common.py không có load; save yêu cầu scheduler)
# ══════════════════════════════════════════════════════════════════════════════

def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    Load a checkpoint saved by common.save_checkpoint.
    Returns the full payload dict so callers can read extra keys.
    """
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload


def _save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    extra: Optional[dict] = None,
) -> None:
    """
    Like common.save_checkpoint but without a scheduler argument.
    Saves: epoch, step, model state, optimizer state, + extra keys.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch":     epoch,
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class JEPAPretrainConfig:
    data: TextDataConfig = field(default_factory=lambda: TextDataConfig(max_length=256))
    # NOTE: OptimConfig.lr / weight_decay are used as INITIAL values for the
    # AdamW constructor only. The actual per-step lr and weight_decay are
    # overridden by the manual cosine schedules below.
    optim: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "text_jepa")
    checkpoint_name: str = "text_jepa_best.pt"
    latest_checkpoint_name: str = "text_jepa_latest.pt"
    loss_plot_name: str = "text_jepa_loss.png"
    plot_every: int = 10
    resume_from_latest: bool = True
    # ── architecture ──────────────────────────────────────────────────────────
    model_name: str = "bert_base"    # "bert_base" | "bert_large"
    hidden_dim: int = 768            # D: must match model_name (bert_base=768)
    predictor_dim: int = 384         # d: recommended D/2 = 384
    predictor_layers: int = 4
    predictor_heads: int = 6         # predictor_dim (384) % heads == 0
    predictor_ffn_dim: int = 1536    # 4 * predictor_dim
    # ── masking ───────────────────────────────────────────────────────────────
    max_span_length: int = 5
    max_num_spans: int = 5
    min_num_spans: int = 5
    mask_seed: Optional[int] = None
    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_decay: float = 0.996
    grad_clip_norm: float = 0.3
    device: str = "auto"
    # ── LR / WD schedules (mirrors tjepa_training.py) ─────────────────────────
    start_lr: float = 0.0002          # lr at step 0 (bottom of warmup ramp)
    peak_lr: float = 0.001            # lr at end of warmup / top of cosine
    final_lr: float = 1.0e-6          # lr at last step
    warmup_epochs: int = 10           # number of epochs for linear warmup
    weight_decay: float = 0.04        # wd at step 0
    final_weight_decay: float = 0.4   # wd at last step (cosine schedule)
    # ── HuggingFace Hub ───────────────────────────────────────────────────────
    hf_repo_id: str = "ducanhdinh/jepa_proof_tjepa"
    push_to_hub: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Manual LR + WD cosine schedules  (identical logic to tjepa_training.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_lr_wd_schedules(
    cfg: JEPAPretrainConfig,
    total_steps: int,
    steps_per_epoch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute per-step LR and weight-decay arrays.

    LR  : linear warmup for `warmup_epochs` epochs, then cosine decay.
          start_lr → peak_lr → final_lr
    WD  : cosine ramp from weight_decay → final_weight_decay.

    Returns
    -------
    lr_schedule : np.ndarray [total_steps]
    wd_schedule : np.ndarray [total_steps]
    """
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    lr_schedule = np.zeros(total_steps)
    wd_schedule = np.zeros(total_steps)

    for step in range(total_steps):
        # ── LR: linear warmup then cosine decay ───────────────────────────
        if step < warmup_steps:
            lr_schedule[step] = (
                cfg.start_lr
                + step * (cfg.peak_lr - cfg.start_lr) / max(1, warmup_steps)
            )
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr_schedule[step] = (
                cfg.final_lr
                + 0.5 * (cfg.peak_lr - cfg.final_lr) * (1 + math.cos(math.pi * progress))
            )

        # ── WD: cosine ramp ────────────────────────────────────────────────
        progress = step / total_steps
        wd_schedule[step] = (
            cfg.weight_decay
            + 0.5 * (cfg.final_weight_decay - cfg.weight_decay)
            * (1 - math.cos(math.pi * progress))
        )

    return lr_schedule, wd_schedule


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Predictor  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class SmallBertPredictor(nn.Module):
    """Narrow BERT predictor: D → d (bottleneck) → D."""

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
            max_position_embeddings=max(BERT_BASE_MAX_POSITION_EMBEDDINGS, max_length),
            type_vocab_size=2,
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
# 5.  TextJEPA  (unchanged architecture)
# ══════════════════════════════════════════════════════════════════════════════

class TextJEPA(nn.Module):
    """Text JEPA: context encoder + EMA target encoder + predictor."""

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

    def _freeze_target_encoder(self):
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, decay: float) -> None:
        """EMA update: target ← decay·target + (1−decay)·context."""
        for ctx, tgt in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            tgt.data.mul_(decay).add_(ctx.data, alpha=1.0 - decay)

    def _encode(self, encoder, input_ids, attention_mask, token_type_ids):
        return encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, return_dict=True,
        ).last_hidden_state

    def encode_full_sequence(self, batch: dict, use_target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if use_target else self.context_encoder
        return self._encode(
            encoder,
            batch["clean_input_ids"],
            batch["clean_attention_mask"],
            batch["clean_token_type_ids"],
        )

    @staticmethod
    def span_jepa_loss(pred, target, span_mask):
        squared_l2_per_token = ((pred - target) ** 2).sum(dim=-1)
        masked = squared_l2_per_token * span_mask.float()
        n_span_tokens = span_mask.float().sum().clamp(min=1.0)
        return masked.sum() / n_span_tokens

    def forward(self, batch: dict) -> dict:
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
# 6.  HuggingFace Hub push  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def push_context_encoder_to_hub(
    model: TextJEPA,
    repo_id: str,
    save_dir: str,
    token: str | None = None,
) -> None:
    _token = token or HF_TOKEN
    if not _token:
        raise RuntimeError(
            "HF_TOKEN không tìm thấy. "
            "Thêm HF_TOKEN=<token> vào file .env hoặc truyền trực tiếp.")

    encoder_dir = os.path.join(save_dir, "context_encoder")
    os.makedirs(encoder_dir, exist_ok=True)

    model.context_encoder.save_pretrained(encoder_dir)
    print(f"Đã lưu context_encoder vào {encoder_dir}")

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
# 7.  Pretrainer  — optimizer & scheduler synced with tjepa_training.py
# ══════════════════════════════════════════════════════════════════════════════

class JEPAPretrainer:
    """
    Trainer cho TextJEPA.

    Optimizer & LR/WD schedule — hoàn toàn giống tjepa_training.py:
      • AdamW chỉ nhận context_encoder + predictor params (target_encoder bị loại)
      • LR  : pre-computed numpy array, assign vào param_group mỗi step
      • WD  : pre-computed numpy array, assign vào param_group mỗi step
      • make_scheduler() từ common KHÔNG được dùng nữa
      • global_step luôn đồng bộ với schedule index
    """

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

        # ── Optimizer: only trainable params (mirrors tjepa_training.py) ──
        # target_encoder has requires_grad=False already, but we build the
        # explicit list to be unambiguous — identical to tjepa_training.py's
        #   trainable_params = list(context_encoder.params) + list(predictor.params)
        self._trainable_params = (
            list(self.model.context_encoder.parameters())
            + list(self.model.predictor.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            self._trainable_params,
            lr=cfg.start_lr,                # overridden every step by schedule
            weight_decay=cfg.weight_decay,  # overridden every step by schedule
        )

        # ── Pre-compute LR + WD schedule arrays ───────────────────────────
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * cfg.optim.epochs
        if cfg.optim.max_steps > 0:
            self.total_steps = min(self.total_steps, cfg.optim.max_steps)

        self.lr_schedule, self.wd_schedule = build_lr_wd_schedules(
            cfg, self.total_steps, self.steps_per_epoch
        )

        self.best_validation_loss = float("inf")
        self.start_epoch = 0
        self.global_step = 0
        self.plotter = LossPlotter(self.loss_plot_path(), "JEPA Pretraining Loss")

        if cfg.resume_from_latest:
            self.load_latest_if_available()

    # ── per-step schedule application (mirrors tjepa_training.py) ────────────

    def _apply_schedule(self, step: int) -> tuple[float, float]:
        """
        Assign lr and weight_decay to all param groups for the given step.
        Returns (current_lr, current_wd) for logging.
        """
        idx = min(step, self.total_steps - 1)   # clamp for safety on resume
        current_lr = float(self.lr_schedule[idx])
        current_wd = float(self.wd_schedule[idx])
        for pg in self.optimizer.param_groups:
            pg["lr"]           = current_lr
            pg["weight_decay"] = current_wd
        return current_lr, current_wd

    # ── objective ─────────────────────────────────────────────────────────────

    def objective(self, output: dict) -> tuple[torch.Tensor, dict]:
        loss = output["span_loss"]
        stats = {
            "loss": float(loss.detach()),
            "span": float(output["span_loss"].detach()),
        }
        return loss, stats

    # ── train step ────────────────────────────────────────────────────────────

    def train_step(self, batch: dict, step: int) -> dict:
        """
        One gradient step.

        Changes vs. original:
          • _apply_schedule(step) sets lr + wd BEFORE the forward pass
            (mirrors tjepa_training.py's schedule application order)
          • No self.scheduler.step() — schedule is fully manual
          • grad_clip applied only to _trainable_params (not all model params)
        """
        batch = move_to_device(batch, self.device)

        # Apply schedule for this step
        current_lr, current_wd = self._apply_schedule(step)

        output = self.model(batch)
        loss, stats = self.objective(output)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self._trainable_params,
                self.cfg.grad_clip_norm,
            )

        self.optimizer.step()
        # NOTE: no self.scheduler.step() — lr/wd are set manually above
        self.model.update_target_encoder(self.cfg.ema_decay)

        stats["lr"] = current_lr
        stats["wd"] = current_wd
        return stats

    # ── evaluation ────────────────────────────────────────────────────────────

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

    # ── paths ─────────────────────────────────────────────────────────────────

    def checkpoint_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.checkpoint_name)

    def latest_checkpoint_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.latest_checkpoint_name)

    def loss_plot_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.loss_plot_name)

    # ── checkpoint resume ─────────────────────────────────────────────────────

    def load_latest_if_available(self) -> None:
        path = self.latest_checkpoint_path()
        if not os.path.exists(path):
            return

        checkpoint = _load_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.device,
        )
        self.best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        self.global_step = int(checkpoint.get("step", 0))
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if "plotter_state" in checkpoint:
            self.plotter.load_state_dict(checkpoint["plotter_state"])
        print(
            f"resumed_from={path}  start_epoch={self.start_epoch + 1}  "
            f"global_step={self.global_step}  "
            f"best_validation_loss={self.best_validation_loss:.4f}"
        )

    # ── main training loop ────────────────────────────────────────────────────

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
                # train_step applies schedule internally and returns lr/wd
                stats = self.train_step(batch, step)
                step += 1

                self.plotter.add_train(step, stats["loss"])
                if step % self.cfg.plot_every == 0:
                    self.plotter.save()

                pbar.set_postfix({
                    "loss": f"{stats['loss']:.4f}",
                    "lr":   f"{stats['lr']:.6f}",
                    "wd":   f"{stats['wd']:.4f}",
                })

                # Early-stop at max_steps
                if self.cfg.optim.max_steps > 0 and step >= self.cfg.optim.max_steps:
                    val = self.evaluate()
                    self.plotter.add_validation(step, val["loss"])
                    self.plotter.save()
                    saved = self.save_if_best(epoch, step, val["loss"])
                    self.save_latest(epoch, step, val["loss"])
                    print(
                        f"step={step}  validation={val}  "
                        f"best_validation_loss={self.best_validation_loss:.4f}  "
                        f"saved_best={saved}"
                    )
                    if self.cfg.push_to_hub:
                        push_context_encoder_to_hub(
                            self.model, self.cfg.hf_repo_id, self.cfg.output_dir)
                    return

            # End-of-epoch validation
            val = self.evaluate()
            self.plotter.add_validation(step, val["loss"])
            self.plotter.save()
            saved = self.save_if_best(epoch, step, val["loss"])
            self.save_latest(epoch, step, val["loss"])
            print(
                f"epoch={epoch + 1}  validation={val}  "
                f"best_validation_loss={self.best_validation_loss:.4f}  "
                f"saved_best={saved}"
            )

        # Push after full training
        if self.cfg.push_to_hub:
            push_context_encoder_to_hub(
                self.model, self.cfg.hf_repo_id, self.cfg.output_dir)

    # ── checkpoint helpers ────────────────────────────────────────────────────

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
        _save_checkpoint(
            path,
            self.model,
            self.optimizer,
            epoch,
            step,
            extra={
                "config":               self.cfg,
                "validation_loss":      validation_loss,
                "best_validation_loss": self.best_validation_loss,
                "checkpoint_type":      checkpoint_type,
                "plotter_state":        self.plotter.state_dict(),
                # Save schedule arrays so a resumed run can inspect them
                "lr_schedule":          self.lr_schedule.tolist(),
                "wd_schedule":          self.wd_schedule.tolist(),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    trainer = JEPAPretrainer(JEPAPretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()