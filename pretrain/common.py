"""Shared model and optimization utilities for pretraining scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertConfig, BertModel


@dataclass
class OptimConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 0
    num_workers: int = 2
    log_every: int = 20
    eval_every: int = 1


def device_from_config(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_bert_base_config(max_length: int, vocab_size: int = 30522) -> BertConfig:
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=max_length,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
    )


def make_optimizer(model: nn.Module, cfg: OptimConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )


def make_scheduler(optimizer: AdamW, cfg: OptimConfig, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - cfg.warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    step: int,
    extra: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


class LossPlotter:
    def __init__(self, path: str, title: str):
        self.path = path
        self.title = title
        self.train_steps: list[int] = []
        self.train_losses: list[float] = []
        self.validation_steps: list[int] = []
        self.validation_losses: list[float] = []

    def add_train(self, step: int, loss: float) -> None:
        self.train_steps.append(step)
        self.train_losses.append(loss)

    def add_validation(self, step: int, loss: float) -> None:
        self.validation_steps.append(step)
        self.validation_losses.append(loss)

    def state_dict(self) -> dict:
        return {
            "train_steps": self.train_steps,
            "train_losses": self.train_losses,
            "validation_steps": self.validation_steps,
            "validation_losses": self.validation_losses,
        }

    def load_state_dict(self, state: dict) -> None:
        self.train_steps = list(state.get("train_steps", []))
        self.train_losses = list(state.get("train_losses", []))
        self.validation_steps = list(state.get("validation_steps", []))
        self.validation_losses = list(state.get("validation_losses", []))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.train_steps, self.train_losses, label="train loss", linewidth=1.2)
        if self.validation_steps:
            ax.plot(
                self.validation_steps,
                self.validation_losses,
                label="validation loss",
                marker="o",
                linewidth=1.8,
            )
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(self.title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.path, dpi=160)
        plt.close(fig)


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionMLP(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BertSentenceEncoder(nn.Module):
    def __init__(self, max_length: int):
        super().__init__()
        self.bert = BertModel(build_bert_base_config(max_length=max_length))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return output.last_hidden_state[:, 0]


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    if rows != cols:
        raise ValueError("off_diagonal expects a square matrix.")
    return x.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    batch_size, dim = z.shape
    z = z - z.mean(dim=0)
    cov = (z.t() @ z) / max(1, batch_size - 1)
    return off_diagonal(cov).pow(2).sum() / dim


def cosine_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (pred * target).sum(dim=-1).mean()
