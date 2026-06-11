"""Offline lexical-substitution view builder.

Tạo view 2 bằng cách thay thế 15–20% token trong câu gốc bằng các token
được dự đoán bởi BERT MLM (top-5, loại trừ token gốc).

Quy trình (CHỈ chạy 1 lần trước khi train):
  1. Với mỗi câu, chọn ngẫu nhiên mask_ratio_low–mask_ratio_high % token
     không phải special token.
  2. Với từng vị trí được chọn:
       a. Clone input_ids, đặt [MASK] tại đúng vị trí đó.
       b. Chạy BERT forward một lần.
       c. Lấy top-5 token tại vị trí đó.
       d. Lọc bỏ token gốc, chọn ngẫu nhiên 1 token từ phần còn lại.
       e. Ghi token thay thế vào bản sao view2.
  3. Cache kết quả: với mỗi câu lưu (input_ids_gốc, input_ids_view2).

Usage:
    from lexical_substitution_augmentor import (
        LexicalSubstitutionConfig,
        build_or_load_substituted_views,
    )

    train_pairs, val_pairs = build_or_load_substituted_views(data_cfg, subst_cfg)
    # train_pairs: List[Tuple[List[int], List[int]]]  (original_ids, view2_ids)

    Sau đó dùng LexicalSubstitutionCollator để wrap vào DataLoader.

Run this code before Train
python lexical_substitution_augmentor.py \
    --train-samples 3000000 \
    --validation-samples 10000 \
    --max-length 256 \
    --batch-size 256 \
    --device auto
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizerFast

# Import từ module data gốc – chỉ cần build_or_load_tokenized
try:
    from pretrained_data_sampler import TextDataConfig, build_or_load_tokenized
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from pretrained_data_sampler import TextDataConfig, build_or_load_tokenized


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LexicalSubstitutionConfig:
    """Tham số điều khiển quá trình tạo view offline."""

    mask_ratio_low: float = 0.15
    """Tỉ lệ token thay thế tối thiểu (tính trên token hợp lệ của mỗi câu)."""

    mask_ratio_high: float = 0.20
    """Tỉ lệ token thay thế tối đa."""

    top_k: int = 5
    """Số lượng ứng viên BERT trả về tại mỗi vị trí."""

    bert_model_name: str = "bert-base-uncased"
    """BERT pretrained dùng để dự đoán ứng viên thay thế."""

    batch_size: int = 256
    """Batch size khi chạy BERT forward (càng lớn càng nhanh, cần VRAM)."""

    seed: int = 42
    """Seed cho random chọn vị trí và chọn ứng viên."""

    device: str = "auto"
    """'auto' → dùng CUDA nếu có, ngược lại CPU."""

    cache_suffix: str = "lexsubst"
    """Hậu tố thêm vào tên file cache để phân biệt với cache sentence gốc."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPECIAL_IDS = {0, 101, 102, 103}   # [PAD]=0, [CLS]=101, [SEP]=102, [MASK]=103


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _valid_positions(input_ids: List[int]) -> List[int]:
    """Trả về danh sách index của token không phải special."""
    return [i for i, tok in enumerate(input_ids) if tok not in _SPECIAL_IDS]


def _choose_positions(valid: List[int], ratio_low: float, ratio_high: float, rng: random.Random) -> List[int]:
    """Chọn ngẫu nhiên ratio_low–ratio_high % vị trí từ danh sách hợp lệ."""
    if not valid:
        return []
    ratio = rng.uniform(ratio_low, ratio_high)
    n = max(1, round(len(valid) * ratio))
    return rng.sample(valid, min(n, len(valid)))


# ---------------------------------------------------------------------------
# Core: xây view2 cho một batch câu bằng BERT
# ---------------------------------------------------------------------------

def _build_view2_for_batch(
    batch_ids: List[List[int]],
    max_length: int,
    model: BertForMaskedLM,
    device: torch.device,
    cfg: LexicalSubstitutionConfig,
    rng: random.Random,
) -> List[List[int]]:
    """
    Với mỗi câu trong batch_ids, tạo view2 bằng lexical substitution.

    Với mỗi vị trí được chọn trong câu, hàm tạo một "probe" tensor = bản clone
    của câu đó với đúng 1 [MASK] tại vị trí đó, sau đó thu thập tất cả probe
    tensor thành một mini-batch duy nhất, chạy BERT forward một lần, rồi đọc
    top-k tại từng vị trí và chọn token thay thế.

    Không gọi BERT nhiều lần: thay vào đó, với B câu mỗi câu có K_i vị trí
    cần thay, ta tạo tối đa sum(K_i) probe tensor và xử lý hết trong 1 forward.
    Nếu sum(K_i) > batch_limit thì chia nhỏ bên ngoài (hàm gọi đã chia batch).
    """
    cls_id = 101
    sep_id = 102
    pad_id = 0
    mask_id = 103

    # Pad tất cả về max_length
    def pad(ids: List[int]) -> List[int]:
        seq = ids[:max_length]
        return seq + [pad_id] * (max_length - len(seq))

    padded = [pad(ids) for ids in batch_ids]

    # Chọn vị trí thay thế cho mỗi câu
    chosen_per_sentence: List[List[int]] = []
    for ids in padded:
        valid = _valid_positions(ids)
        positions = _choose_positions(valid, cfg.mask_ratio_low, cfg.mask_ratio_high, rng)
        positions.sort()
        chosen_per_sentence.append(positions)

    # Tạo probe tensors: mỗi (câu i, vị trí j) → 1 row
    probe_rows: List[List[int]] = []      # input_ids có [MASK] tại vị trí j
    probe_attn: List[List[int]] = []      # attention_mask tương ứng
    probe_meta: List[Tuple[int, int, int]] = []  # (sent_idx, pos_in_sent, original_token)

    for sent_idx, (ids, positions) in enumerate(zip(padded, chosen_per_sentence)):
        attn = [1 if tok != pad_id else 0 for tok in ids]
        for pos in positions:
            orig_tok = ids[pos]
            probe = ids.copy()
            probe[pos] = mask_id
            probe_rows.append(probe)
            probe_attn.append(attn)
            probe_meta.append((sent_idx, pos, orig_tok))

    # Bắt đầu từ bản sao của padded, sẽ ghi token thay thế vào
    view2 = [ids.copy() for ids in padded]

    if not probe_rows:
        # Không có gì để thay → view2 = bản sao gốc
        return [ids[:len(orig)] for ids, orig in zip(view2, batch_ids)]

    # Chạy BERT forward theo mini-batch
    model.eval()
    probe_input_tensor = torch.tensor(probe_rows, dtype=torch.long, device=device)
    probe_attn_tensor = torch.tensor(probe_attn, dtype=torch.long, device=device)

    INFER_BATCH = 512   # forward batch tối đa để tránh OOM
    logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, probe_input_tensor.size(0), INFER_BATCH):
            chunk_ids = probe_input_tensor[start: start + INFER_BATCH]
            chunk_attn = probe_attn_tensor[start: start + INFER_BATCH]
            out = model(input_ids=chunk_ids, attention_mask=chunk_attn)
            logits_list.append(out.logits.cpu())   # [chunk, seq_len, vocab]

    logits_all = torch.cat(logits_list, dim=0)     # [n_probes, seq_len, vocab]

    # Đọc kết quả và ghi vào view2
    for probe_idx, (sent_idx, pos, orig_tok) in enumerate(probe_meta):
        pos_logits = logits_all[probe_idx, pos, :]          # [vocab]
        top_ids = torch.topk(pos_logits, cfg.top_k).indices.tolist()
        candidates = [t for t in top_ids if t != orig_tok and t not in _SPECIAL_IDS]
        if candidates:
            replacement = rng.choice(candidates)
        else:
            # Mọi top-k đều là token gốc hoặc special → giữ nguyên
            replacement = orig_tok
        view2[sent_idx][pos] = replacement

    # Cắt về độ dài gốc (bỏ padding)
    return [v2[:len(orig)] for v2, orig in zip(view2, batch_ids)]


# ---------------------------------------------------------------------------
# Xây cache offline
# ---------------------------------------------------------------------------

def _cache_path(data_cfg: TextDataConfig, subst_cfg: LexicalSubstitutionConfig, split: str) -> str:
    """Tạo đường dẫn cache duy nhất theo cấu hình."""
    dataset = data_cfg.dataset_name.replace("/", "_")
    config = data_cfg.dataset_config or "default"
    name = (
        f"{dataset}_{config}_{split}_ml{data_cfg.max_length}_"
        f"ratio{subst_cfg.mask_ratio_low}-{subst_cfg.mask_ratio_high}_"
        f"topk{subst_cfg.top_k}_seed{subst_cfg.seed}_{subst_cfg.cache_suffix}.pt"
    )
    return os.path.join(data_cfg.cache_dir, name)


def _build_substituted_split(
    tokenized_ids: List[List[int]],
    max_length: int,
    cfg: LexicalSubstitutionConfig,
    split_name: str,
) -> List[Tuple[List[int], List[int]]]:
    """
    Tạo danh sách (original_ids, view2_ids) cho toàn bộ split.

    original_ids: token ids gốc (có [CLS]/[SEP]/[PAD] từ TextChunkDataset)
    view2_ids:    sau khi lexical substitution
    """
    device = _resolve_device(cfg.device)
    rng = random.Random(cfg.seed)

    print(f"[LexSubst] Loading BERT model '{cfg.bert_model_name}' on {device}...")
    model = BertForMaskedLM.from_pretrained(cfg.bert_model_name)
    model = model.to(device)
    model.eval()

    pairs: List[Tuple[List[int], List[int]]] = []
    batch_size = cfg.batch_size
    total = len(tokenized_ids)

    pbar = tqdm(range(0, total, batch_size), desc=f"[LexSubst] {split_name}", ncols=110)
    for start in pbar:
        batch = tokenized_ids[start: start + batch_size]
        view2_batch = _build_view2_for_batch(batch, max_length, model, device, cfg, rng)
        for orig, v2 in zip(batch, view2_batch):
            pairs.append((orig, v2))
        pbar.set_postfix({"done": f"{min(start + batch_size, total)}/{total}"})

    # Giải phóng VRAM
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pairs


def build_or_load_substituted_views(
    data_cfg: TextDataConfig,
    subst_cfg: Optional[LexicalSubstitutionConfig] = None,
) -> Tuple[List[Tuple[List[int], List[int]]], List[Tuple[List[int], List[int]]]]:
    """
    Entry point chính.

    Trả về (train_pairs, val_pairs) trong đó mỗi phần tử là
    (original_input_ids, view2_input_ids) — đều là List[int] đã có
    [CLS]/[SEP]/[PAD] phù hợp với max_length.

    Nếu cache đã tồn tại thì load trực tiếp, không chạy BERT lại.
    """
    if subst_cfg is None:
        subst_cfg = LexicalSubstitutionConfig()

    os.makedirs(data_cfg.cache_dir, exist_ok=True)
    train_path = _cache_path(data_cfg, subst_cfg, "train")
    val_path = _cache_path(data_cfg, subst_cfg, "val")

    # Kiểm tra cache
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[LexSubst] Cache found → loading from:\n  {train_path}\n  {val_path}")
        train_payload = torch.load(train_path, map_location="cpu", weights_only=False)
        val_payload = torch.load(val_path, map_location="cpu", weights_only=False)
        return train_payload["pairs"], val_payload["pairs"]

    # Lấy tokenized sentences từ pipeline gốc
    print("[LexSubst] Building tokenized sentences (sẽ dùng cache nếu đã có)...")
    train_ids, val_ids = build_or_load_tokenized(data_cfg)

    # Xây view2 offline
    print("[LexSubst] ─── Train split ───")
    train_pairs = _build_substituted_split(train_ids, data_cfg.max_length, subst_cfg, "train")
    print("[LexSubst] ─── Validation split ───")
    val_pairs = _build_substituted_split(val_ids, data_cfg.max_length, subst_cfg, "val")

    # Lưu cache
    torch.save({"pairs": train_pairs, "config": asdict(subst_cfg)}, train_path)
    torch.save({"pairs": val_pairs, "config": asdict(subst_cfg)}, val_path)
    print(f"[LexSubst] Cache saved:\n  {train_path}\n  {val_path}")

    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Dataset + Collator tích hợp với DataLoader
# ---------------------------------------------------------------------------

class LexSubstDataset(Dataset):
    """
    Dataset nhận danh sách (original_ids, view2_ids).

    Mỗi item trả về dict sẵn sàng để collate:
        original_input_ids  : [max_length]  LongTensor
        view2_input_ids     : [max_length]  LongTensor
        attention_mask      : [max_length]  LongTensor  (dựa trên original)
        token_type_ids      : [max_length]  LongTensor  (toàn 0)
    """

    def __init__(
        self,
        pairs: List[Tuple[List[int], List[int]]],
        max_length: int,
        pad_token_id: int = 0,
    ):
        self.pairs = pairs
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def _pad(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        seq = ids[: self.max_length]
        pad_len = self.max_length - len(seq)
        padded = seq + [self.pad_token_id] * pad_len
        attn = [1] * len(seq) + [0] * pad_len
        return padded, attn

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        orig_ids, v2_ids = self.pairs[index]
        orig_padded, attn = self._pad(orig_ids)
        v2_padded, _ = self._pad(v2_ids)
        return {
            "original_input_ids": torch.tensor(orig_padded, dtype=torch.long),
            "view2_input_ids": torch.tensor(v2_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "token_type_ids": torch.zeros(self.max_length, dtype=torch.long),
        }


class LexSubstTwoViewCollator:
    """
    Collator dùng khi train các method cần 2 view (BYOL, Barlow Twins, VICReg).

    Output batch keys:
        view1_input_ids       – câu gốc
        view1_attention_mask
        view1_token_type_ids
        view2_input_ids       – câu đã lexical-substituted
        view2_attention_mask
        view2_token_type_ids
    """

    def __call__(self, examples: List[dict]) -> dict:
        return {
            "view1_input_ids": torch.stack([e["original_input_ids"] for e in examples]),
            "view1_attention_mask": torch.stack([e["attention_mask"] for e in examples]),
            "view1_token_type_ids": torch.stack([e["token_type_ids"] for e in examples]),
            "view2_input_ids": torch.stack([e["view2_input_ids"] for e in examples]),
            "view2_attention_mask": torch.stack([e["attention_mask"] for e in examples]),
            "view2_token_type_ids": torch.stack([e["token_type_ids"] for e in examples]),
        }


def build_lexsubst_dataloaders(
    data_cfg: TextDataConfig,
    subst_cfg: Optional[LexicalSubstitutionConfig] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Hàm tiện ích: build (hoặc load từ cache) rồi trả về DataLoader.

    Thay thế `build_pretrain_dataloaders` khi dùng lexical substitution view.
    """
    train_pairs, val_pairs = build_or_load_substituted_views(data_cfg, subst_cfg)

    train_dataset = LexSubstDataset(train_pairs, max_length=data_cfg.max_length)
    val_dataset = LexSubstDataset(val_pairs, max_length=data_cfg.max_length)
    collator = LexSubstTwoViewCollator()
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=collator,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI: chạy độc lập để tạo cache trước
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build offline lexical-substitution view cache.")
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mask-ratio-low", type=float, default=0.15)
    parser.add_argument("--mask-ratio-high", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    data_cfg = TextDataConfig(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        max_length=args.max_length,
    )
    if args.cache_dir:
        data_cfg.cache_dir = args.cache_dir

    subst_cfg = LexicalSubstitutionConfig(
        mask_ratio_low=args.mask_ratio_low,
        mask_ratio_high=args.mask_ratio_high,
        top_k=args.top_k,
        bert_model_name=args.bert_model,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )

    train_pairs, val_pairs = build_or_load_substituted_views(data_cfg, subst_cfg)
    print(
        f"\n✓ Done. train_pairs={len(train_pairs):,}  val_pairs={len(val_pairs):,}\n"
        f"  Sample pair[0]:\n"
        f"    original : {train_pairs[0][0][:12]}...\n"
        f"    view2    : {train_pairs[0][1][:12]}..."
    )