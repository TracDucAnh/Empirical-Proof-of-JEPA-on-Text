"""Dataset downloading, tokenization, and view sampling for pretraining.

The pretraining scripts all consume the same sentence/chunk stream from
`allenai/c4`, English config.  The default training cache is capped at one
million collected sentences so every method sees the same pretraining budget.
This file owns the data download path so each method can focus on its objective:

    HuggingFace dataset -> WordPiece ids -> fixed-length BERT chunks -> collator

Collators build the method-specific views:
    - MLM labels for BERT
    - two independently masked views for BYOL/VICReg/Barlow Twins
    - clean + masked view and span mask for JEPA
"""

from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class TextDataConfig:
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    train_split: str = "train"
    validation_split: str = "validation"
    tokenizer_name: str = "bert-base-uncased"
    cache_dir: str = os.path.join(PROJECT_ROOT, "pretrained_text_cache")
    max_length: int = 128
    min_words: int = 8
    train_samples: int = 1_000_000
    validation_samples: int = 10_000
    seed: int = 42
    streaming: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(name: str) -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(name)


def split_sentences(text: str) -> List[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if part.strip()
    ]


def _cache_path(cfg: TextDataConfig) -> str:
    dataset = cfg.dataset_name.replace("/", "_")
    config = cfg.dataset_config or "default"
    name = (
        f"{dataset}_{config}_ml{cfg.max_length}_"
        f"tr{cfg.train_samples}_va{cfg.validation_samples}_seed{cfg.seed}.pt"
    )
    return os.path.join(cfg.cache_dir, name)


def _document_cache_path(cfg: TextDataConfig) -> str:
    dataset = cfg.dataset_name.replace("/", "_")
    config = cfg.dataset_config or "default"
    name = (
        f"{dataset}_{config}_docs_ml{cfg.max_length}_"
        f"tr{cfg.train_samples}_va{cfg.validation_samples}_seed{cfg.seed}.pt"
    )
    return os.path.join(cfg.cache_dir, name)


def load_cache(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _iter_dataset_rows(cfg: TextDataConfig, split: str) -> Iterable[dict]:
    if cfg.dataset_config:
        return load_dataset(
            cfg.dataset_name,
            cfg.dataset_config,
            split=split,
            streaming=cfg.streaming,
        )
    return load_dataset(cfg.dataset_name, split=split, streaming=cfg.streaming)


def collect_texts(cfg: TextDataConfig, split: str, limit: int, name: str) -> List[str]:
    rows = _iter_dataset_rows(cfg, split)
    texts: List[str] = []
    pbar = tqdm(total=limit, desc=f"Collecting {name}", ncols=100)

    for row in rows:
        raw = row.get("text", "")
        text = raw.strip() if isinstance(raw, str) else ""
        if not text or (text.startswith("=") and text.endswith("=")):
            continue
        for sentence in split_sentences(text):
            if len(sentence.split()) >= cfg.min_words:
                texts.append(sentence)
                pbar.update(1)
                if len(texts) >= limit:
                    pbar.close()
                    return texts

    pbar.close()
    return texts


def collect_text_documents(
    cfg: TextDataConfig,
    split: str,
    sentence_limit: int,
    name: str,
) -> List[List[str]]:
    rows = _iter_dataset_rows(cfg, split)
    documents: List[List[str]] = []
    collected_sentences = 0
    pbar = tqdm(total=sentence_limit, desc=f"Collecting {name}", ncols=100)

    for row in rows:
        raw = row.get("text", "")
        text = raw.strip() if isinstance(raw, str) else ""
        if not text or (text.startswith("=") and text.endswith("=")):
            continue

        document = [
            sentence
            for sentence in split_sentences(text)
            if len(sentence.split()) >= cfg.min_words
        ]
        if len(document) < 2:
            continue

        documents.append(document)
        collected_sentences += len(document)
        pbar.update(min(len(document), sentence_limit - pbar.n))
        if collected_sentences >= sentence_limit:
            pbar.close()
            return documents

    pbar.close()
    return documents


def tokenize_texts(tokenizer: BertTokenizerFast, texts: List[str]) -> List[List[int]]:
    tokenized: List[List[int]] = []
    for start in tqdm(range(0, len(texts), 1024), desc="Tokenizing", ncols=100):
        batch = texts[start : start + 1024]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        tokenized.extend([ids for ids in encoded if ids])
    return tokenized


def tokenize_documents(tokenizer: BertTokenizerFast, documents: List[List[str]]) -> List[List[List[int]]]:
    tokenized_documents: List[List[List[int]]] = []
    for document in tqdm(documents, desc="Tokenizing documents", ncols=100):
        encoded = tokenizer(document, add_special_tokens=False)["input_ids"]
        tokenized = [ids for ids in encoded if ids]
        if len(tokenized) >= 2:
            tokenized_documents.append(tokenized)
    return tokenized_documents


def build_or_load_tokenized(cfg: TextDataConfig) -> Tuple[List[List[int]], List[List[int]]]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = _cache_path(cfg)
    if os.path.exists(path):
        payload = load_cache(path)
        return payload["train_ids"], payload["validation_ids"]

    tokenizer = load_tokenizer(cfg.tokenizer_name)
    train_texts = collect_texts(cfg, cfg.train_split, cfg.train_samples, "train")
    validation_texts = collect_texts(
        cfg,
        cfg.validation_split,
        cfg.validation_samples,
        "validation",
    )
    train_ids = tokenize_texts(tokenizer, train_texts)
    validation_ids = tokenize_texts(tokenizer, validation_texts)
    torch.save(
        {
            "config": asdict(cfg),
            "train_ids": train_ids,
            "validation_ids": validation_ids,
        },
        path,
    )
    return train_ids, validation_ids


def build_or_load_tokenized_documents(
    cfg: TextDataConfig,
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = _document_cache_path(cfg)
    if os.path.exists(path):
        payload = load_cache(path)
        return payload["train_documents"], payload["validation_documents"]

    tokenizer = load_tokenizer(cfg.tokenizer_name)
    train_documents = collect_text_documents(
        cfg,
        cfg.train_split,
        cfg.train_samples,
        "train documents",
    )
    validation_documents = collect_text_documents(
        cfg,
        cfg.validation_split,
        cfg.validation_samples,
        "validation documents",
    )
    train_ids = tokenize_documents(tokenizer, train_documents)
    validation_ids = tokenize_documents(tokenizer, validation_documents)
    torch.save(
        {
            "config": asdict(cfg),
            "train_documents": train_ids,
            "validation_documents": validation_ids,
        },
        path,
    )
    return train_ids, validation_ids


class TextChunkDataset(Dataset):
    """Fixed-length BERT chunk dataset with [CLS] and [SEP] tokens."""

    def __init__(
        self,
        tokenized_sentences: List[List[int]],
        max_length: int,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        if max_length < 3:
            raise ValueError("max_length must fit [CLS], at least one token, and [SEP].")
        self.examples = [ids for ids in tokenized_sentences if ids]
        self.max_length = max_length
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        ids = self.examples[index][: self.max_length - 2]
        input_ids = [self.cls_token_id] + ids + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }


class BertSentencePairDataset(Dataset):
    """Sentence-pair examples for original BERT MLM + NSP pretraining."""

    def __init__(
        self,
        tokenized_documents: List[List[List[int]]],
        max_length: int,
        seed: int = 42,
        nsp_probability: float = 0.5,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        if max_length < 5:
            raise ValueError("max_length must fit [CLS] A [SEP] B [SEP].")
        self.documents = [document for document in tokenized_documents if len(document) >= 2]
        self.examples = [
            (doc_index, sent_index)
            for doc_index, document in enumerate(self.documents)
            for sent_index in range(len(document) - 1)
        ]
        if not self.examples:
            raise ValueError("BERT NSP needs at least one document with two sentences.")
        self.max_length = max_length
        self.seed = seed
        self.nsp_probability = nsp_probability
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.examples)

    def _truncate_pair(self, tokens_a: List[int], tokens_b: List[int], rng: random.Random) -> None:
        max_pair_tokens = self.max_length - 3
        while len(tokens_a) + len(tokens_b) > max_pair_tokens:
            target = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            if rng.random() < 0.5:
                del target[0]
            else:
                target.pop()

    def _sample_pair(self, index: int, rng: random.Random) -> Tuple[List[int], List[int], int]:
        doc_index, sent_index = self.examples[index]
        document = self.documents[doc_index]
        tokens_a = list(document[sent_index])

        if rng.random() < self.nsp_probability and len(self.documents) > 1:
            random_doc_index = rng.randrange(len(self.documents))
            while random_doc_index == doc_index:
                random_doc_index = rng.randrange(len(self.documents))
            random_document = self.documents[random_doc_index]
            tokens_b = list(random_document[rng.randrange(len(random_document))])
            next_sentence_label = 1
        else:
            tokens_b = list(document[sent_index + 1])
            next_sentence_label = 0

        self._truncate_pair(tokens_a, tokens_b, rng)
        return tokens_a, tokens_b, next_sentence_label

    def __getitem__(self, index: int) -> dict:
        rng = random.Random(self.seed + index)
        tokens_a, tokens_b, next_sentence_label = self._sample_pair(index, rng)
        input_ids = [self.cls_token_id] + tokens_a + [self.sep_token_id] + tokens_b + [self.sep_token_id]
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "next_sentence_label": torch.tensor(next_sentence_label, dtype=torch.long),
        }


class SpanMasker:
    """Contiguous span masking used to create text augmentations."""

    def __init__(
        self,
        mask_token_id: int = 103,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
        max_span_length: int = 5,
    ):
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.max_span_length = max_span_length

    def mask_one(self, input_ids: torch.Tensor, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
        masked = input_ids.clone()
        span_mask = torch.zeros_like(input_ids)
        special = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        valid = [idx for idx, token_id in enumerate(input_ids.tolist()) if token_id not in special]
        if not valid:
            return masked, span_mask

        max_len = min(self.max_span_length, len(valid))
        span_len = rng.randint(1, max_len)
        start_candidates = []
        for i in range(len(valid)):
            chunk = valid[i : i + span_len]
            if len(chunk) == span_len and chunk[-1] - chunk[0] + 1 == span_len:
                start_candidates.append(chunk[0])

        start = rng.choice(start_candidates) if start_candidates else rng.choice(valid)
        span_positions = list(range(start, min(start + span_len, len(input_ids))))
        for pos in span_positions:
            if int(input_ids[pos]) not in special:
                masked[pos] = self.mask_token_id
                span_mask[pos] = 1
        return masked, span_mask


class BertMLMCollator:
    """Dynamic BERT MLM corruption with the 80/10/10 replacement rule."""

    def __init__(
        self,
        vocab_size: int = 30522,
        mlm_probability: float = 0.15,
        mask_token_id: int = 103,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[dict]) -> dict:
        input_ids = torch.stack([item["input_ids"] for item in examples])
        attention_mask = torch.stack([item["attention_mask"] for item in examples])
        token_type_ids = torch.stack([item["token_type_ids"] for item in examples])
        labels = input_ids.clone()

        probability = torch.full(input_ids.shape, self.mlm_probability)
        special = (
            input_ids.eq(self.cls_token_id)
            | input_ids.eq(self.sep_token_id)
            | input_ids.eq(self.pad_token_id)
            | ~attention_mask.bool()
        )
        probability.masked_fill_(special, 0.0)
        masked = torch.bernoulli(probability).bool()
        labels[~masked] = -100

        replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked
        input_ids[replace_mask] = self.mask_token_id

        replace_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked
            & ~replace_mask
        )
        random_tokens = torch.randint(0, self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[replace_random] = random_tokens[replace_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


class BertPretrainingCollator(BertMLMCollator):
    """Dynamic MLM corruption for BERT sentence-pair NSP examples."""

    def __call__(self, examples: List[dict]) -> dict:
        batch = super().__call__(examples)
        batch["next_sentence_label"] = torch.stack(
            [item["next_sentence_label"] for item in examples]
        )
        return batch


class TwoViewSpanMaskCollator:
    """Create two independently span-masked views for Siamese objectives."""

    def __init__(self, max_span_length: int = 5, seed: int = 42):
        self.masker = SpanMasker(max_span_length=max_span_length)
        self.seed = seed
        self.calls = 0

    def __call__(self, examples: List[dict]) -> dict:
        input_ids = torch.stack([item["input_ids"] for item in examples])
        attention_mask = torch.stack([item["attention_mask"] for item in examples])
        token_type_ids = torch.stack([item["token_type_ids"] for item in examples])

        view1, view2 = [], []
        for row, ids in enumerate(input_ids):
            rng1 = random.Random(self.seed + self.calls * 2_000_003 + row)
            rng2 = random.Random(self.seed + self.calls * 2_000_003 + 1_000_003 + row)
            masked1, _ = self.masker.mask_one(ids, rng1)
            masked2, _ = self.masker.mask_one(ids, rng2)
            view1.append(masked1)
            view2.append(masked2)

        self.calls += 1
        return {
            "view1_input_ids": torch.stack(view1),
            "view1_attention_mask": attention_mask,
            "view1_token_type_ids": token_type_ids,
            "view2_input_ids": torch.stack(view2),
            "view2_attention_mask": attention_mask,
            "view2_token_type_ids": token_type_ids,
        }


class JEPASpanMaskCollator:
    """Create clean and multi-span masked views for Text-JEPA."""

    def __init__(
        self,
        max_span_length: int = 5,
        max_num_spans: int = 5,
        min_num_spans: int = 5,
        mask_token_id: int = 103,
        sep_token_id: int = 102,
        cls_token_id: int = 101,
        pad_token_id: int = 0,
        seed: Optional[int] = None,
    ):
        if min_num_spans < 1:
            raise ValueError("min_num_spans must be at least 1.")
        if max_num_spans < min_num_spans:
            raise ValueError("max_num_spans must be greater than or equal to min_num_spans.")
        if max_span_length < 1:
            raise ValueError("max_span_length must be at least 1.")

        self.max_span_length = max_span_length
        self.max_num_spans = max_num_spans
        self.min_num_spans = min_num_spans
        self.mask_token_id = mask_token_id
        self.sep_token_id = sep_token_id
        self.cls_token_id = cls_token_id
        self.pad_token_id = pad_token_id
        self.rng = random.Random(seed)

    def _get_valid_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[int]:
        special_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        return [
            idx
            for idx in range(input_ids.size(0))
            if int(input_ids[idx]) not in special_ids and int(attention_mask[idx]) == 1
        ]

    def _sample_spans(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        available = self._get_valid_positions(input_ids, attention_mask)
        if not available:
            return [(1, 1)]

        n_spans = self.rng.randint(self.min_num_spans, self.max_num_spans)
        available_set = set(available)
        masked_positions: set[int] = set()
        spans: List[Tuple[int, int]] = []

        for _ in range(n_spans):
            free = [position for position in available if position not in masked_positions]
            if not free:
                break

            span_len = self.rng.randint(1, min(self.max_span_length, len(free)))
            valid_starts = [
                position
                for position in free
                if all(
                    (position + offset) in available_set
                    and (position + offset) not in masked_positions
                    for offset in range(span_len)
                )
            ]
            if not valid_starts:
                break

            start = self.rng.choice(valid_starts)
            end = start + span_len - 1
            spans.append((start, end))
            masked_positions.update(range(start, end + 1))

        return spans if spans else [(available[0], available[0])]

    def __call__(self, examples: List[dict]) -> dict:
        clean_input_ids = torch.stack([item["input_ids"] for item in examples])
        clean_attention_mask = torch.stack([item["attention_mask"] for item in examples])
        clean_token_type_ids = torch.stack([item["token_type_ids"] for item in examples])

        masked_input_ids = clean_input_ids.clone()
        span_masks = torch.zeros_like(clean_input_ids)
        for row in range(clean_input_ids.size(0)):
            spans = self._sample_spans(clean_input_ids[row], clean_attention_mask[row])
            for start, end in spans:
                masked_input_ids[row, start : end + 1] = self.mask_token_id
                span_masks[row, start : end + 1] = 1

        return {
            "clean_input_ids": clean_input_ids,
            "clean_attention_mask": clean_attention_mask,
            "clean_token_type_ids": clean_token_type_ids,
            "masked_input_ids": masked_input_ids,
            "masked_attention_mask": clean_attention_mask,
            "masked_token_type_ids": clean_token_type_ids,
            "span_mask": span_masks,
        }


def build_pretrain_dataloaders(
    data_cfg: TextDataConfig,
    batch_size: int,
    collate_fn,
    validation_collate_fn=None,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_ids, validation_ids = build_or_load_tokenized(data_cfg)
    train_dataset = TextChunkDataset(train_ids, max_length=data_cfg.max_length)
    validation_dataset = TextChunkDataset(validation_ids, max_length=data_cfg.max_length)
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=validation_collate_fn or collate_fn,
    )
    return train_loader, validation_loader


def build_bert_pretraining_dataloaders(
    data_cfg: TextDataConfig,
    batch_size: int,
    collate_fn,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_documents, validation_documents = build_or_load_tokenized_documents(data_cfg)
    train_dataset = BertSentencePairDataset(
        train_documents,
        max_length=data_cfg.max_length,
        seed=data_cfg.seed,
    )
    validation_dataset = BertSentencePairDataset(
        validation_documents,
        max_length=data_cfg.max_length,
        seed=data_cfg.seed + 10_000_000,
    )
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, validation_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and tokenize C4 pretraining data.")
    parser.add_argument(
        "--mode",
        choices=("bert", "sentences", "all"),
        default="all",
        help="bert builds document pairs for MLM+NSP; sentences builds single-sentence chunks.",
    )
    parser.add_argument("--train-samples", type=int, default=1_000_000)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--cache-dir", type=str, default=os.path.join(PROJECT_ROOT, "pretrained_text_cache"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-streaming", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TextDataConfig:
    return TextDataConfig(
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        max_length=args.max_length,
        min_words=args.min_words,
        cache_dir=args.cache_dir,
        seed=args.seed,
        streaming=not args.no_streaming,
    )


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    if cfg.train_samples < 1 or cfg.validation_samples < 1:
        raise ValueError("train-samples and validation-samples must be positive.")

    if args.mode in {"bert", "all"}:
        train_documents, validation_documents = build_or_load_tokenized_documents(cfg)
        print(
            "bert_cache_ready "
            f"train_documents={len(train_documents)} "
            f"validation_documents={len(validation_documents)}"
        )

    if args.mode in {"sentences", "all"}:
        train_ids, validation_ids = build_or_load_tokenized(cfg)
        print(
            "sentence_cache_ready "
            f"train_sentences={len(train_ids)} "
            f"validation_sentences={len(validation_ids)}"
        )


if __name__ == "__main__":
    main()
