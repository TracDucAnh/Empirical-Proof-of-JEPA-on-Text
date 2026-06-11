"""VICReg text pretraining — Lexical Substitution views.

Giống hệt text_vicreg.py nhưng dùng lexical-substitution làm augmentation:
  - View 1 : câu gốc (không mask)
  - View 2 : 15–20% token được thay bằng token ngữ nghĩa gần (BERT MLM top-5)

Cache view2 được xây offline (1 lần) bởi lexical_substitution_augmentor.py.
Lúc train chỉ đọc cache, không gọi BERT thêm lần nào.

Run cache trước:
    python lexical_substitution_augmentor.py \
        --train-samples 3000000 \
        --validation-samples 10000 \
        --max-length 256 \
        --batch-size 256 \
        --device auto
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

# ── Load .env (HF_TOKEN, v.v.) ────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = "ducanhdinh/jepa_proof_vicreg_replace"   # ← repo mới
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretrained_data_sampler import TextDataConfig, set_seed
from pretrain.common import (
    LossPlotter,
    OptimConfig,
    build_bert_base_config,
    device_from_config,
    load_training_checkpoint,
    make_optimizer,
    make_scheduler,
    move_to_device,
    save_checkpoint,
    vicreg_covariance_loss,
    vicreg_variance_loss,
)

# ── Lexical substitution dataloader (thay thế TwoViewSpanMaskCollator) ────────
from lexical_substitution_augmentor import (
    LexicalSubstitutionConfig,
    build_lexsubst_dataloaders,
)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VICRegReplacePretrainConfig:
    data: TextDataConfig = field(default_factory=TextDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    lexsubst: LexicalSubstitutionConfig = field(default_factory=LexicalSubstitutionConfig)
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "text_vicreg_replace")
    checkpoint_name: str = "text_vicreg_replace_best.pt"
    latest_checkpoint_name: str = "text_vicreg_replace_latest.pt"
    loss_plot_name: str = "text_vicreg_replace_loss.png"
    plot_every: int = 10
    resume_from_latest: bool = True
    expander_dim: int = 3072
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    device: str = "auto"


# ---------------------------------------------------------------------------
# Model — giữ nguyên 100% so với text_vicreg.py
# ---------------------------------------------------------------------------

class BertMeanPoolEncoder(nn.Module):
    def __init__(self, max_length: int):
        super().__init__()
        self.bert = BertModel(build_bert_base_config(max_length=max_length))

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class VICRegExpander(nn.Module):
    def __init__(self, input_dim: int = 768, expander_dim: int = 3072):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, expander_dim),
            nn.BatchNorm1d(expander_dim),
            nn.ReLU(inplace=True),
            nn.Linear(expander_dim, expander_dim),
            nn.BatchNorm1d(expander_dim),
            nn.ReLU(inplace=True),
            nn.Linear(expander_dim, expander_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextVICRegReplace(nn.Module):
    def __init__(self, cfg: VICRegReplacePretrainConfig):
        super().__init__()
        self.encoder = BertMeanPoolEncoder(max_length=cfg.data.max_length)
        self.expander = VICRegExpander(768, cfg.expander_dim)

    def forward_view(self, input_ids, attention_mask, token_type_ids):
        pooled = self.encoder(input_ids, attention_mask, token_type_ids)
        return self.expander(pooled)

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


# ---------------------------------------------------------------------------
# Trainer — chỉ khác ở __init__ (dataloader) và push_to_hub (repo_id + card)
# ---------------------------------------------------------------------------

class VICRegReplacePretrainer:
    def __init__(self, cfg: VICRegReplacePretrainConfig):
        self.cfg = cfg
        set_seed(cfg.data.seed)
        self.device = device_from_config(cfg.device)
        self.model = TextVICRegReplace(cfg).to(self.device)

        # ── Dùng lexical substitution dataloaders thay vì span-mask ──────────
        self.train_loader, self.validation_loader = build_lexsubst_dataloaders(
            data_cfg=cfg.data,
            subst_cfg=cfg.lexsubst,
            batch_size=cfg.optim.batch_size,
            num_workers=cfg.optim.num_workers,
        )
        # ─────────────────────────────────────────────────────────────────────

        self.optimizer = make_optimizer(self.model, cfg.optim)
        total_steps = len(self.train_loader) * cfg.optim.epochs
        if cfg.optim.max_steps > 0:
            total_steps = min(total_steps, cfg.optim.max_steps)
        self.scheduler = make_scheduler(self.optimizer, cfg.optim, total_steps)
        self.best_validation_loss = float("inf")
        self.start_epoch = 0
        self.global_step = 0
        self.plotter = LossPlotter(self.loss_plot_path(), "VICReg-Replace Pretraining Loss")
        if cfg.resume_from_latest:
            self.load_latest_if_available()

    # ── Objective — giữ nguyên ────────────────────────────────────────────────

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
        self.model.train()  # giữ train mode để BN dùng batch stats
        totals = {"loss": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0}
        count = 0
        for batch in self.validation_loader:
            batch = move_to_device(batch, self.device)
            z1, z2 = self.model(batch)
            _, stats = self.objective(z1, z2)
            for key in totals:
                totals[key] += stats[key]
            count += 1
        return {key: value / max(1, count) for key, value in totals.items()}

    # ── Paths ─────────────────────────────────────────────────────────────────

    def checkpoint_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.checkpoint_name)

    def latest_checkpoint_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.latest_checkpoint_name)

    def loss_plot_path(self) -> str:
        return os.path.join(self.cfg.output_dir, self.cfg.loss_plot_name)

    # ── Checkpoint resume ─────────────────────────────────────────────────────

    def load_latest_if_available(self) -> None:
        path = self.latest_checkpoint_path()
        if not os.path.exists(path):
            return
        checkpoint = load_training_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.device,
        )
        self.best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        self.global_step = int(checkpoint.get("step", 0))
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if "plotter_state" in checkpoint:
            self.plotter.load_state_dict(checkpoint["plotter_state"])
        print(
            f"resumed_from={path} start_epoch={self.start_epoch + 1} "
            f"global_step={self.global_step} best_validation_loss={self.best_validation_loss:.4f}"
        )

    # ── Train loop — giữ nguyên ───────────────────────────────────────────────

    def train(self) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        step = self.global_step
        if self.cfg.optim.max_steps > 0 and step >= self.cfg.optim.max_steps:
            print(f"checkpoint already reached max_steps={self.cfg.optim.max_steps}")
            return

        self.model.train()
        for epoch in range(self.start_epoch, self.cfg.optim.epochs):
            pbar = tqdm(self.train_loader, desc=f"VICReg-Replace epoch {epoch + 1}", ncols=120)
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

    # ── Save helpers — giữ nguyên ─────────────────────────────────────────────

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
        """Push encoder + full model weights + model card lên Hugging Face Hub."""
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

        # Full model weights
        weights_path = os.path.join(self.cfg.output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), weights_path)
        api.upload_file(
            path_or_fileobj=weights_path,
            path_in_repo="pytorch_model.bin",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add TextVICReg-Replace full model weights (encoder + expander)",
        )
        print("   ✓ Model weights đã được push.")

        # Encoder BERT riêng
        encoder_dir = os.path.join(self.cfg.output_dir, "encoder_export")
        os.makedirs(encoder_dir, exist_ok=True)
        self.model.encoder.bert.save_pretrained(encoder_dir)
        api.upload_folder(
            folder_path=encoder_dir,
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="encoder",
            commit_message="Add BERT encoder (mean-pool) weights",
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
        """Tự sinh và push README.md (model card) lên Hub."""
        cfg = self.cfg
        card_content = f"""\
---
language: en
license: apache-2.0
tags:
  - bert
  - vicreg
  - self-supervised-learning
  - contrastive-learning
  - lexical-substitution
  - pretraining
  - text-embeddings
---

# {repo_id}

BERT encoder pretrained from scratch với **VICReg + Lexical Substitution augmentation**.

## Augmentation strategy

Thay vì span masking, model này dùng **lexical substitution** để tạo view 2:

| | Mô tả |
|---|---|
| **View 1** | Câu gốc (không thay đổi) |
| **View 2** | 15–20% token ngẫu nhiên được thay bằng token ngữ nghĩa gần, dự đoán bởi BERT MLM (top-5, loại trừ token gốc) |

Quá trình tạo view 2 được thực hiện **offline 1 lần** trước khi train, không gọi BERT thêm lần nào trong vòng lặp training.

## Kiến trúc VICReg

```
Text → BERT (mean-pool) → z ∈ R^768 → Expander MLP → z' ∈ R^{cfg.expander_dim}
                                                       ↑ VICReg loss áp dụng tại đây
```

| Loss term | Hệ số | Mô tả |
|---|---|---|
| Invariance | `{cfg.sim_coeff}` | MSE giữa z1 và z2 (căn chỉnh hai views) |
| Variance | `{cfg.std_coeff}` | Giữ std của mỗi chiều ≥ 1 (chống collapse) |
| Covariance | `{cfg.cov_coeff}` | Decorrelate các chiều embedding |

Expander gồm 3 lớp Linear-BatchNorm-ReLU (dim = `{cfg.expander_dim}`).

## Thông số huấn luyện

| Tham số | Giá trị |
|---|---|
| Max sequence length | {cfg.data.max_length} |
| Batch size | {cfg.optim.batch_size} |
| Epochs | {cfg.optim.epochs} |
| Learning rate | {cfg.optim.lr} |
| Expander dim | {cfg.expander_dim} |
| Mask ratio (lexsubst) | {cfg.lexsubst.mask_ratio_low}–{cfg.lexsubst.mask_ratio_high} |
| Top-k candidates | {cfg.lexsubst.top_k} |
| sim_coeff | {cfg.sim_coeff} |
| std_coeff | {cfg.std_coeff} |
| cov_coeff | {cfg.cov_coeff} |

## Cách dùng — BERT encoder (feature extraction)

```python
from transformers import BertModel, BertTokenizerFast
import torch

tokenizer = BertTokenizerFast.from_pretrained("{repo_id}")
bert      = BertModel.from_pretrained("{repo_id}/encoder")

encoded = tokenizer(
    ["Hello world!", "VICReg with lexical substitution."],
    return_tensors="pt",
    padding=True,
    truncation=True,
)
with torch.no_grad():
    out    = bert(**encoded)
    hidden = out.last_hidden_state          # (B, T, 768)
    mask   = encoded["attention_mask"].unsqueeze(-1).float()
    emb    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean-pool → (B, 768)
```

## Cách dùng — Full model (encoder + expander)

```python
import torch
from text_vicreg_replace import TextVICRegReplace, VICRegReplacePretrainConfig

cfg   = VICRegReplacePretrainConfig()
model = TextVICRegReplace(cfg)
state = torch.load(
    hf_hub_download("{repo_id}", "pytorch_model.bin"),
    map_location="cpu",
)
model.load_state_dict(state)
model.eval()
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
    trainer = VICRegReplacePretrainer(VICRegReplacePretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()