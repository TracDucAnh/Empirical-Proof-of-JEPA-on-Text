"""Barlow Twins text pretraining.

Two masked views are encoded by a shared network.  The cross-correlation matrix
between projected views is driven to the identity: diagonal entries preserve
shared information and off-diagonal entries reduce redundancy.
"""

from __future__ import annotations

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
    LossPlotter,
    OptimConfig,
    ProjectionMLP,
    device_from_config,
    make_optimizer,
    make_scheduler,
    move_to_device,
    off_diagonal,
    save_checkpoint,
)

HF_REPO_ID = "ducanhdinh/jepa_proof_barlow_twins"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
HF_TOKEN = os.environ.get("HF_TOKEN")


@dataclass
class BarlowTwinsPretrainConfig:
    data: TextDataConfig = field(default_factory=TextDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "text_barlow_twins")
    checkpoint_name: str = "text_barlow_twins_best.pt"
    latest_checkpoint_name: str = "text_barlow_twins_latest.pt"
    loss_plot_name: str = "text_barlow_twins_loss.png"
    plot_every: int = 10
    resume_from_latest: bool = True
    projector_hidden_dim: int = 2048
    projector_out_dim: int = 8192
    offdiag_coeff: float = 0.005
    max_span_length: int = 5
    device: str = "auto"


class TextBarlowTwins(nn.Module):
    def __init__(self, cfg: BarlowTwinsPretrainConfig):
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


class BarlowTwinsPretrainer:
    def __init__(self, cfg: BarlowTwinsPretrainConfig):
        self.cfg = cfg
        set_seed(cfg.data.seed)
        self.device = device_from_config(cfg.device)
        self.model = TextBarlowTwins(cfg).to(self.device)
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
        self.best_validation_loss = float("inf")
        self.start_epoch = 0
        self.global_step = 0
        self.plotter = LossPlotter(self.loss_plot_path(), "Barlow Twins Pretraining Loss")
        if cfg.resume_from_latest:
            self.load_latest_if_available()

    def objective(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch_size = z1.size(0)
        z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0, unbiased=False).clamp(min=1e-6)
        z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0, unbiased=False).clamp(min=1e-6)
        correlation = (z1_norm.t() @ z2_norm) / batch_size

        on_diag = torch.diagonal(correlation).add(-1).pow(2).sum()
        off_diag = off_diagonal(correlation).pow(2).sum()
        loss = on_diag + self.cfg.offdiag_coeff * off_diag
        stats = {
            "loss": float(loss.detach()),
            "on_diag": float(on_diag.detach()),
            "off_diag": float(off_diag.detach()),
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
        totals = {"loss": 0.0, "on_diag": 0.0, "off_diag": 0.0}
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
            pbar = tqdm(self.train_loader, desc=f"Barlow Twins epoch {epoch + 1}", ncols=120)
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
                    self.push_to_hub()
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

        self.push_to_hub()

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

    # ── Hugging Face Hub ──────────────────────────────────────────────────────

    def push_to_hub(self, repo_id: str = HF_REPO_ID) -> None:
        """Push encoder + tokenizer lên Hugging Face Hub."""
        if not HF_TOKEN:
            print("⚠  HF_TOKEN không tìm thấy trong .env — bỏ qua push_to_hub.")
            return

        from huggingface_hub import HfApi, login
        from transformers import BertTokenizerFast

        print(f"\n🚀 Đang push model lên {repo_id} …")
        login(token=HF_TOKEN)
        api = HfApi(token=HF_TOKEN)

        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # Load best checkpoint trước khi push
        best_path = self.checkpoint_path()
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model"])
            print(f"   ✓ Loaded best checkpoint từ {best_path}")
        else:
            print("   ⚠  Không tìm thấy best checkpoint, push model weights hiện tại.")

        # Encoder BERT
        encoder_dir = os.path.join(self.cfg.output_dir, "encoder_export")
        os.makedirs(encoder_dir, exist_ok=True)
        self.model.encoder.bert.save_pretrained(encoder_dir)
        api.upload_folder(
            folder_path=encoder_dir,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="encoder",
            commit_message="Add BERT encoder weights",
        )
        print("   ✓ BERT encoder đã được push vào thư mục encoder/.")

        # Tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased",
            model_max_length=self.cfg.data.max_length,
        )
        tokenizer_dir = os.path.join(self.cfg.output_dir, "tokenizer_export")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)
        api.upload_folder(
            folder_path=tokenizer_dir,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=".",
            commit_message="Add tokenizer",
        )
        print("   ✓ Tokenizer đã được push.")

        self._push_model_card(repo_id, api)
        print(f"   ✅ Xong! Model đã có tại https://huggingface.co/{repo_id}\n")

    def _push_model_card(self, repo_id: str, api) -> None:
        cfg = self.cfg
        card_content = f"""\
---
language: en
license: apache-2.0
tags:
  - bert
  - barlow-twins
  - self-supervised-learning
  - contrastive-learning
  - span-masking
  - pretraining
  - text-embeddings
---

# {repo_id}

BERT encoder pretrained from scratch với **Barlow Twins**.

## Augmentation strategy

Hai view được tạo bằng **span masking** độc lập (non-overlapping):

| | Mô tả |
|---|---|
| **View 1** | Câu gốc với các span ngẫu nhiên bị mask |
| **View 2** | Câu gốc với các span ngẫu nhiên khác bị mask (không overlap) |

## Kiến trúc Barlow Twins

```
View 1 ──► Encoder (θ) ──► Projector (θ) ──► z1  ──┐
                                                     ├──► Cross-correlation C = Z1ᵀZ2 / N  ──► Loss
View 2 ──► Encoder (θ) ──► Projector (θ) ──► z2  ──┘

Loss = Σ(C_ii - 1)²  +  λ · Σ_{{i≠j}} C_ij²
```

## Thông số huấn luyện

| Tham số | Giá trị |
|---|---|
| Max sequence length | {cfg.data.max_length} |
| Batch size | {cfg.optim.batch_size} |
| Epochs | {cfg.optim.epochs} |
| Learning rate | {cfg.optim.lr} |
| Projector hidden dim | {cfg.projector_hidden_dim} |
| Projector out dim | {cfg.projector_out_dim} |
| Off-diagonal coeff (λ) | {cfg.offdiag_coeff} |
| Max span length | {cfg.max_span_length} |

## Cách dùng

```python
from transformers import BertModel, BertTokenizerFast
import torch

tokenizer = BertTokenizerFast.from_pretrained("{repo_id}")
bert      = BertModel.from_pretrained("{repo_id}/encoder")

encoded = tokenizer(
    ["Hello world!", "Barlow Twins pretraining."],
    return_tensors="pt",
    padding=True,
    truncation=True,
)
with torch.no_grad():
    out     = bert(**encoded)
    cls_emb = out.last_hidden_state[:, 0, :]   # [CLS] token → (B, 768)
```
"""
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print("   ✓ Model card đã được push.")


def main() -> None:
    trainer = BarlowTwinsPretrainer(BarlowTwinsPretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()