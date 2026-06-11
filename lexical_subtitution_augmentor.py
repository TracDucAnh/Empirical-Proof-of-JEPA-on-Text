"""Offline lexical-substitution view builder — OPTIMISED.

Key optimisations vs the original:
  ① Multi-mask-per-sentence: mask ALL chosen positions in ONE forward pass
    instead of one pass per position.  Speedup ≈ mask_ratio × seq_len ≈ 40-50×.
  ② Vectorised probe construction: build the probe tensor with broadcast ops
    (scatter_ on the whole batch at once) instead of Python-level list.copy()
    per position.
  ③ Larger inference chunks + torch.compile (PyTorch ≥ 2.0).
  ④ AMP (autocast) on CUDA for ~1.5× throughput on modern GPUs.
  ⑤ Pinned-memory host tensors so CPU→GPU transfers overlap with compute.

Quy trình (CHỈ chạy 1 lần trước khi train):
  1. Với mỗi câu, chọn ngẫu nhiên mask_ratio_low–mask_ratio_high % token
     không phải special token.
  2. Tạo MỘT probe cho câu đó với tất cả vị trí đã chọn đặt thành [MASK].
  3. Chạy BERT forward MỘT LẦN, thu logits tại TẤT CẢ vị trí mask cùng lúc.
  4. Với mỗi vị trí, lấy top-k, lọc token gốc & special, chọn ngẫu nhiên 1.
  5. Scatter toàn bộ token thay thế vào view2 bằng tensor ops.
  6. Cache kết quả: với mỗi câu lưu (input_ids_gốc, input_ids_view2).

Usage (không thay đổi API so với bản gốc):
    from lexical_substitution_augmentor import (
        LexicalSubstitutionConfig,
        build_or_load_substituted_views,
    )
    train_pairs, val_pairs = build_or_load_substituted_views(data_cfg, subst_cfg)
    # train_pairs: List[Tuple[List[int], List[int]]]  (original_ids, view2_ids)

Run this script before training:
    python lexical_substitution_augmentor.py \
        --train-samples 3000000 \
        --validation-samples 10000 \
        --max-length 256 \
        --batch-size 1024 \
        --device auto
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForMaskedLM

try:
    from pretrained_data_sampler import TextDataConfig, build_or_load_tokenized
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from pretrained_data_sampler import TextDataConfig, build_or_load_tokenized


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LexicalSubstitutionConfig:
    """Tham số điều khiển quá trình tạo view offline."""

    mask_ratio_low: float = 0.15
    mask_ratio_high: float = 0.20
    top_k: int = 5
    bert_model_name: str = "bert-base-uncased"

    # ① Tăng batch_size lên vì giờ mỗi câu chỉ cần 1 forward thay vì ~50.
    #   Với 24 GB VRAM có thể dùng 2048–4096 thoải mái.
    batch_size: int = 1024

    seed: int = 42
    device: str = "auto"
    cache_suffix: str = "lexsubst_v2"

    # ② Dùng torch.compile nếu PyTorch ≥ 2.0 (tắt nếu gặp lỗi).
    use_compile: bool = True

    # ③ AMP autocast (chỉ hiệu quả trên CUDA).
    use_amp: bool = True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_ID  = 0
CLS_ID  = 101
SEP_ID  = 102
MASK_ID = 103
_SPECIAL_IDS      = {PAD_ID, CLS_ID, SEP_ID, MASK_ID}
_SPECIAL_IDS_LIST = list(_SPECIAL_IDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _valid_positions_batch(input_ids_tensor: torch.Tensor) -> torch.Tensor:
    """
    Trả về boolean mask [B, L]: True tại vị trí không phải special token.
    Vectorised — không có vòng lặp Python.
    """
    spec = torch.tensor(_SPECIAL_IDS_LIST, dtype=torch.long,
                        device=input_ids_tensor.device)               # [S]
    # input_ids_tensor: [B, L]  →  is_special: [B, L]
    is_special = input_ids_tensor.unsqueeze(-1).eq(spec).any(-1)      # [B, L]
    return ~is_special                                                 # [B, L]


# ---------------------------------------------------------------------------
# ① OPTIMISED CORE: 1 forward pass per sentence (multi-mask)
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
    Tạo view2 cho cả batch, mỗi câu chỉ cần ĐÚNG 1 BERT forward.

    So với bản gốc (1 forward / vị trí mask):
      - n_probes giảm từ  B × (avg_masked_positions)  xuống còn  B
      - Ví dụ: batch=1024, avg_pos=40  →  40 960 probes → 1 024 probes  (~40× ít hơn)
    """
    B = len(batch_ids)

    # ------------------------------------------------------------------
    # Bước 1: Pad & tensor hoá tất cả câu (CPU)
    #   Dùng pinned memory để CPU→GPU overlap với lần forward sau.
    # ------------------------------------------------------------------
    def _pad(ids: List[int]) -> List[int]:
        s = ids[:max_length]
        return s + [PAD_ID] * (max_length - len(s))

    padded = [_pad(ids) for ids in batch_ids]

    # Dùng pin_memory để transfer nhanh hơn lên GPU
    input_tensor = torch.tensor(padded, dtype=torch.long)             # [B, L] CPU
    if device.type == "cuda":
        input_tensor = input_tensor.pin_memory()

    view2_tensor = input_tensor.clone()                               # [B, L] CPU

    # ------------------------------------------------------------------
    # Bước 2: Chọn vị trí mask cho từng câu – vectorised
    #   valid_mask[b, i] = True nếu token i không phải special.
    # ------------------------------------------------------------------
    valid_mask = _valid_positions_batch(input_tensor)                 # [B, L]

    # Đếm số token hợp lệ mỗi câu để tính n_mask
    valid_counts = valid_mask.sum(dim=1).tolist()                     # [B] Python ints

    # Xây probe tensor: bắt đầu từ input_tensor, thay [MASK] tại vị trí chọn
    probe_tensor = input_tensor.clone()                               # [B, L] CPU

    # Lưu metadata cho scatter về sau
    all_batch_indices : List[int] = []
    all_pos_indices   : List[int] = []
    all_orig_tokens   : List[int] = []

    for b in range(B):
        valid_pos = valid_mask[b].nonzero(as_tuple=False).squeeze(1).tolist()
        if not valid_pos:
            continue
        ratio = rng.uniform(cfg.mask_ratio_low, cfg.mask_ratio_high)
        n = max(1, round(len(valid_pos) * ratio))
        chosen = rng.sample(valid_pos, min(n, len(valid_pos)))

        for pos in chosen:
            orig_tok = padded[b][pos]
            probe_tensor[b, pos] = MASK_ID
            all_batch_indices.append(b)
            all_pos_indices.append(pos)
            all_orig_tokens.append(orig_tok)

    if not all_batch_indices:
        return [padded[i][:len(batch_ids[i])] for i in range(B)]

    # ------------------------------------------------------------------
    # Bước 3: Xây attention mask (vectorised)
    # ------------------------------------------------------------------
    attn_mask = (input_tensor != PAD_ID).long()                       # [B, L]

    # ------------------------------------------------------------------
    # Bước 4: BERT forward – CHỈ 1 LẦN cho cả batch
    #   Mỗi câu → 1 probe với tất cả [MASK] đã đặt.
    #   Chia nhỏ thành INFER_CHUNK để tránh OOM.
    # ------------------------------------------------------------------
    INFER_CHUNK = 512
    logits_list: List[torch.Tensor] = []
    model.eval()

    use_amp = cfg.use_amp and device.type == "cuda"

    with torch.no_grad():
        for start in range(0, B, INFER_CHUNK):
            chunk_ids  = probe_tensor[start: start + INFER_CHUNK].to(device, non_blocking=True)
            chunk_attn = attn_mask   [start: start + INFER_CHUNK].to(device, non_blocking=True)
            # ③ AMP: float16 trên CUDA → throughput cao hơn ~1.5×
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(input_ids=chunk_ids, attention_mask=chunk_attn)
            logits_list.append(out.logits.float().cpu())              # [chunk, L, vocab]

    logits_all = torch.cat(logits_list, dim=0)                        # [B, L, vocab]

    # ------------------------------------------------------------------
    # Bước 5: Thu logits tại đúng từng vị trí mask (vectorised gather)
    # ------------------------------------------------------------------
    n_total   = len(all_batch_indices)
    batch_t   = torch.tensor(all_batch_indices, dtype=torch.long)     # [N]
    pos_t     = torch.tensor(all_pos_indices,   dtype=torch.long)     # [N]
    orig_tok_t = torch.tensor(all_orig_tokens,  dtype=torch.long)     # [N]

    # logits_all[batch_t[i], pos_t[i], :]  cho mỗi i
    # Dùng advanced indexing thay cho expand+gather để tiết kiệm bộ nhớ
    pos_logits = logits_all[batch_t, pos_t, :]                        # [N, vocab]

    # ------------------------------------------------------------------
    # Bước 6: Top-k + valid_mask + multinomial sampling (fully vectorised)
    # ------------------------------------------------------------------
    topk_ids = torch.topk(pos_logits, cfg.top_k, dim=-1).indices      # [N, top_k]

    orig_mask  = topk_ids.eq(orig_tok_t.unsqueeze(1))                 # [N, top_k]
    spec_ids_t = torch.tensor(_SPECIAL_IDS_LIST, dtype=torch.long)    # [S]
    spec_mask  = topk_ids.unsqueeze(2).eq(
                     spec_ids_t.view(1, 1, -1)
                 ).any(dim=-1)                                         # [N, top_k]

    valid = ~orig_mask & ~spec_mask                                    # [N, top_k]
    has_valid = valid.any(dim=-1)                                      # [N]

    weights = valid.float()
    weights[~has_valid, 0] = 1.0                                       # fallback hàng toàn-zero

    sample_idx  = torch.multinomial(weights, num_samples=1)            # [N, 1]
    chosen_tok  = topk_ids.gather(1, sample_idx).squeeze(1)           # [N]
    replacement = torch.where(has_valid, chosen_tok, orig_tok_t)      # [N]

    # ------------------------------------------------------------------
    # Bước 7: Scatter tất cả replacements vào view2_tensor cùng lúc
    # ------------------------------------------------------------------
    view2_tensor.index_put_(
        indices=(batch_t, pos_t),
        values=replacement,
    )

    # ------------------------------------------------------------------
    # Bước 8: Cắt về độ dài gốc
    # ------------------------------------------------------------------
    view2_list = view2_tensor.tolist()
    return [v2[:len(orig)] for v2, orig in zip(view2_list, batch_ids)]


# ---------------------------------------------------------------------------
# Xây cache offline
# ---------------------------------------------------------------------------

def _cache_path(data_cfg: TextDataConfig, subst_cfg: LexicalSubstitutionConfig,
                split: str) -> str:
    dataset = data_cfg.dataset_name.replace("/", "_")
    config  = data_cfg.dataset_config or "default"
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
    device = _resolve_device(cfg.device)
    rng = random.Random(cfg.seed)

    print(f"[LexSubst] Loading BERT model '{cfg.bert_model_name}' on {device}...")
    model = BertForMaskedLM.from_pretrained(cfg.bert_model_name)
    model = model.to(device)
    model.eval()

    # ② torch.compile: fuse ops trong BERT forward → ~10-20% thêm trên GPU
    if cfg.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[LexSubst] torch.compile enabled.")
        except Exception as e:
            print(f"[LexSubst] torch.compile skipped: {e}")

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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pairs


def build_or_load_substituted_views(
    data_cfg: TextDataConfig,
    subst_cfg: Optional[LexicalSubstitutionConfig] = None,
) -> Tuple[List[Tuple[List[int], List[int]]], List[Tuple[List[int], List[int]]]]:
    """
    Entry point chính. API giữ nguyên so với bản gốc.

    Trả về (train_pairs, val_pairs):
        mỗi phần tử là (original_input_ids, view2_input_ids).
    Nếu cache tồn tại → load ngay, không chạy BERT lại.
    """
    if subst_cfg is None:
        subst_cfg = LexicalSubstitutionConfig()

    os.makedirs(data_cfg.cache_dir, exist_ok=True)
    train_path = _cache_path(data_cfg, subst_cfg, "train")
    val_path   = _cache_path(data_cfg, subst_cfg, "val")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[LexSubst] Cache found → loading from:\n  {train_path}\n  {val_path}")
        train_payload = torch.load(train_path, map_location="cpu", weights_only=False)
        val_payload   = torch.load(val_path,   map_location="cpu", weights_only=False)
        return train_payload["pairs"], val_payload["pairs"]

    print("[LexSubst] Building tokenized sentences (dùng cache nếu có)...")
    train_ids, val_ids = build_or_load_tokenized(data_cfg)

    print("[LexSubst] ─── Train split ───")
    train_pairs = _build_substituted_split(train_ids, data_cfg.max_length, subst_cfg, "train")
    print("[LexSubst] ─── Validation split ───")
    val_pairs = _build_substituted_split(val_ids, data_cfg.max_length, subst_cfg, "val")

    torch.save({"pairs": train_pairs, "config": asdict(subst_cfg)}, train_path)
    torch.save({"pairs": val_pairs,   "config": asdict(subst_cfg)}, val_path)
    print(f"[LexSubst] Cache saved:\n  {train_path}\n  {val_path}")

    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Dataset + Collator (API không đổi)
# ---------------------------------------------------------------------------

class LexSubstDataset(Dataset):
    """
    Dataset nhận danh sách (original_ids, view2_ids).
    Output keys: original_input_ids, view2_input_ids, attention_mask, token_type_ids.
    """

    def __init__(self, pairs: List[Tuple[List[int], List[int]]],
                 max_length: int, pad_token_id: int = 0):
        self.pairs = pairs
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def _pad(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        seq = ids[: self.max_length]
        pad_len = self.max_length - len(seq)
        return seq + [self.pad_token_id] * pad_len, [1] * len(seq) + [0] * pad_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        orig_ids, v2_ids = self.pairs[index]
        orig_padded, attn = self._pad(orig_ids)
        v2_padded, _      = self._pad(v2_ids)
        return {
            "original_input_ids": torch.tensor(orig_padded, dtype=torch.long),
            "view2_input_ids":    torch.tensor(v2_padded,   dtype=torch.long),
            "attention_mask":     torch.tensor(attn,        dtype=torch.long),
            "token_type_ids":     torch.zeros(self.max_length, dtype=torch.long),
        }


class LexSubstTwoViewCollator:
    """
    Collator cho train BYOL / Barlow Twins / VICReg.
    Output keys: view1_* và view2_*.
    """

    def __call__(self, examples: List[dict]) -> dict:
        stack = lambda key: torch.stack([e[key] for e in examples])
        return {
            "view1_input_ids":      stack("original_input_ids"),
            "view1_attention_mask": stack("attention_mask"),
            "view1_token_type_ids": stack("token_type_ids"),
            "view2_input_ids":      stack("view2_input_ids"),
            "view2_attention_mask": stack("attention_mask"),
            "view2_token_type_ids": stack("token_type_ids"),
        }


def build_lexsubst_dataloaders(
    data_cfg: TextDataConfig,
    subst_cfg: Optional[LexicalSubstitutionConfig] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_pairs, val_pairs = build_or_load_substituted_views(data_cfg, subst_cfg)

    train_ds = LexSubstDataset(train_pairs, max_length=data_cfg.max_length)
    val_ds   = LexSubstDataset(val_pairs,   max_length=data_cfg.max_length)
    collator = LexSubstTwoViewCollator()
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        drop_last=drop_last, collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        drop_last=False, collate_fn=collator,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build offline lexical-substitution view cache (optimised).")
    parser.add_argument("--train-samples",      type=int,   default=1_000_000)
    parser.add_argument("--validation-samples", type=int,   default=10_000)
    parser.add_argument("--max-length",         type=int,   default=256)
    parser.add_argument("--mask-ratio-low",     type=float, default=0.15)
    parser.add_argument("--mask-ratio-high",    type=float, default=0.20)
    parser.add_argument("--top-k",              type=int,   default=5)
    parser.add_argument("--batch-size",         type=int,   default=1024)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--device",             type=str,   default="auto")
    parser.add_argument("--bert-model",         type=str,   default="bert-base-uncased")
    parser.add_argument("--cache-dir",          type=str,   default=None)
    parser.add_argument("--no-compile",         action="store_true")
    parser.add_argument("--no-amp",             action="store_true")
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
        use_compile=not args.no_compile,
        use_amp=not args.no_amp,
    )

    train_pairs, val_pairs = build_or_load_substituted_views(data_cfg, subst_cfg)
    print(
        f"\n✓ Done. train_pairs={len(train_pairs):,}  val_pairs={len(val_pairs):,}\n"
        f"  Sample pair[0]:\n"
        f"    original : {train_pairs[0][0][:12]}...\n"
        f"    view2    : {train_pairs[0][1][:12]}..."
    )