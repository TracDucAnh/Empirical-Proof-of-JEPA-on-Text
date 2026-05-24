"""push_bert.py
─────────────────────────────────────────────────────────────────────────────
Load best BERT checkpoint và push lên HuggingFace Hub dưới tên proof_bert.

Cách dùng:
    python push_bert.py                        # dùng best checkpoint (default)
    python push_bert.py --checkpoint path/to/checkpoint.pt
    python push_bert.py --repo-name ducanhdinh/proof_bert

Yêu cầu:
    pip install python-dotenv huggingface_hub transformers
    File .env cùng cấp chứa:  HF_TOKEN=hf_xxxxxxxxxxxxxxxx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── load .env trước mọi thứ ──────────────────────────────────────────────────
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN không tìm thấy.\n"
        f"Hãy tạo file {_ENV_PATH} với nội dung:\n"
        "    HF_TOKEN=hf_xxxxxxxxxxxxxxxx"
    )

# ── stdlib / third-party ─────────────────────────────────────────────────────
import torch
from huggingface_hub import HfApi, create_repo
from transformers import BertConfig, BertForPreTraining, BertTokenizerFast

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.common import build_bert_base_config  # noqa: E402
from pretrain.pretrained_BERT import BERTPretrainConfig  # noqa: E402  — required to unpickle checkpoint


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CHECKPOINT = str(
    Path(__file__).parent.parent / "outputs" / "bert_pretraining" / "bert_pretraining_best.pt"
)
DEFAULT_REPO_NAME  = "ducanhdinh/proof_bert"


def load_model_from_checkpoint(checkpoint_path: str) -> BertForPreTraining:
    """Khởi tạo BertForPreTraining và load state_dict từ checkpoint."""
    print(f"[1/4] Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # checkpoint được lưu bởi save_checkpoint() trong pretrain/common.py
    # state_dict nằm ở key "model"
    state_dict = checkpoint["model"]

    config = build_bert_base_config(max_length=256)
    model  = BertForPreTraining(config)
    model.load_state_dict(state_dict)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    step  = checkpoint.get("step",  "?")
    val   = checkpoint.get("best_validation_loss", checkpoint.get("validation_loss", "?"))
    print(f"    epoch={epoch}  step={step}  best_val_loss={val}")
    return model


def push(checkpoint_path: str, repo_name: str) -> None:
    # ── 1. load model ────────────────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path)

    # ── 2. tạo repo (idempotent — không lỗi nếu đã tồn tại) ─────────────────
    print(f"[2/4] Creating / verifying HF repo: {repo_name}")
    api = HfApi(token=HF_TOKEN)
    create_repo(
        repo_id=repo_name,
        token=HF_TOKEN,
        exist_ok=True,
        private=False,
    )

    # ── 3. save model + config vào thư mục tạm ───────────────────────────────
    tmp_dir = Path("/tmp/proof_bert_upload")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] Saving model to {tmp_dir}")

    model.save_pretrained(str(tmp_dir))

    # tokenizer bert-base-uncased khớp với dataloader (BertTokenizerFast)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(str(tmp_dir))

    # ghi model card tối giản
    readme = tmp_dir / "README.md"
    readme.write_text(
        "# proof_bert\n\n"
        "BERT-base pretrained from scratch with MLM + NSP on a C4 subset.\n\n"
        "Trained as part of ICLR empirical evidence experiments.\n"
    )

    # ── 4. upload toàn bộ thư mục ────────────────────────────────────────────
    print(f"[4/4] Uploading to HuggingFace Hub ...")
    api.upload_folder(
        folder_path=str(tmp_dir),
        repo_id=repo_name,
        token=HF_TOKEN,
        commit_message=f"Upload proof_bert checkpoint (step={checkpoint_path})",
    )

    print(f"\n✓ Done! Model available at: https://huggingface.co/ducanhdinh/proof_bert")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push BERT checkpoint to HuggingFace Hub.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to .pt checkpoint file (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--repo-name",
        default=DEFAULT_REPO_NAME,
        help=f"HuggingFace repo name, without username prefix (default: {DEFAULT_REPO_NAME})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    push(checkpoint_path=args.checkpoint, repo_name=args.repo_name)


if __name__ == "__main__":
    main()