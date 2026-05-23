"""Original BERT pretraining.

Objective from BERT:
    1. masked language modeling with the 80/10/10 replacement rule
    2. next sentence prediction over sentence pairs
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import BertForPreTraining

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretrained_data_sampler import (
    BertPretrainingCollator,
    TextDataConfig,
    build_bert_pretraining_dataloaders,
    set_seed,
)
from pretrain.common import (
    OptimConfig,
    build_bert_base_config,
    device_from_config,
    make_optimizer,
    make_scheduler,
    move_to_device,
    save_checkpoint,
)


def bert_data_config() -> TextDataConfig:
    return TextDataConfig(max_length=256)


@dataclass
class BERTPretrainConfig:
    data: TextDataConfig = field(default_factory=bert_data_config)
    optim: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "bert_pretraining")
    checkpoint_name: str = "bert_pretraining_best.pt"
    latest_checkpoint_name: str = "bert_pretraining_latest.pt"
    loss_plot_name: str = "bert_pretraining_loss.png"
    plot_every: int = 10
    resume_from_latest: bool = True
    mlm_probability: float = 0.15
    device: str = "auto"


class LossPlotter:
    def __init__(self, path: str):
        self.path = path
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
        ax.set_title("BERT MLM+NSP Pretraining Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.path, dpi=160)
        plt.close(fig)


class BERTPretrainer:
    def __init__(self, cfg: BERTPretrainConfig):
        self.cfg = cfg
        set_seed(cfg.data.seed)
        self.device = device_from_config(cfg.device)
        self.model = BertForPreTraining(
            build_bert_base_config(max_length=cfg.data.max_length)
        ).to(self.device)
        self.collator = BertPretrainingCollator(mlm_probability=cfg.mlm_probability)
        self.train_loader, self.validation_loader = build_bert_pretraining_dataloaders(
            cfg.data,
            batch_size=cfg.optim.batch_size,
            collate_fn=self.collator,
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
        self.plotter = LossPlotter(self.loss_plot_path())
        if cfg.resume_from_latest:
            self.load_latest_if_available()

    def train_step(self, batch: dict) -> torch.Tensor:
        batch = move_to_device(batch, self.device)
        output = self.model(**batch)
        loss = output.loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach()

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in self.validation_loader:
            batch = move_to_device(batch, self.device)
            total_loss += float(self.model(**batch).loss.item())
            total_batches += 1
        self.model.train()
        return total_loss / max(1, total_batches)

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
            pbar = tqdm(self.train_loader, desc=f"BERT MLM+NSP epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                loss = self.train_step(batch)
                step += 1
                loss_value = float(loss)
                self.plotter.add_train(step, loss_value)
                if step % self.cfg.plot_every == 0:
                    self.plotter.save()
                pbar.set_postfix({"loss": f"{loss_value:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                if self.cfg.optim.max_steps > 0 and step >= self.cfg.optim.max_steps:
                    val_loss = self.evaluate()
                    self.plotter.add_validation(step, val_loss)
                    self.plotter.save()
                    saved = self.save_if_best(epoch, step, val_loss)
                    self.save_latest(epoch, step, val_loss)
                    print(
                        f"step={step} validation_pretraining_loss={val_loss:.4f} "
                        f"best_validation_loss={self.best_validation_loss:.4f} saved_best={saved}"
                    )
                    return

            val_loss = self.evaluate()
            self.plotter.add_validation(step, val_loss)
            self.plotter.save()
            saved = self.save_if_best(epoch, step, val_loss)
            self.save_latest(epoch, step, val_loss)
            print(
                f"epoch={epoch + 1} validation_pretraining_loss={val_loss:.4f} "
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
    trainer = BERTPretrainer(BERTPretrainConfig())
    trainer.train()


if __name__ == "__main__":
    main()
