"""Frozen-backbone downstream evaluation.

Protocol:
    1. Choose a pretrained method checkpoint.
    2. Load that method's encoder/backbone.
    3. Freeze every backbone parameter.
    4. Train only the downstream task head on that task's train split.
    5. Evaluate on that task's test/eval split.

Results are saved to:
    outputs/results/<task>_<method>_<timestamp>.json   — single-run detail
    outputs/results/results.csv                        — appended summary row

FIXES (QA):
    Bug 1 — collator token_start loop: changed condition from
        offsets[token_start][0] < answer_start
      to
        offsets[token_start][1] <= answer_start
      so the loop advances past tokens whose char_end is still at or before
      answer_start, stopping at the token that CONTAINS answer_start.

    Bug 2 — collator token_end loop: changed condition from
        offsets[token_end][1] > answer_end
      to
        offsets[token_end][0] >= answer_end
      so the loop retreats past tokens whose char_start is at or beyond
      answer_end, stopping at the token that CONTAINS answer_end.

    Bug 3 — evaluate() span search: replaced independent argmax with a
      constrained best-span search that enforces end >= start and restricts
      both positions to valid context tokens (not None in offset_mapping),
      preventing the head from predicting spans across question/padding tokens.

    Bug 4 — evaluate() offset guard: seq_len is derived from len(offsets)
      (the Python list) not from tensor dimensions, preventing index errors
      on padded sequences.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertForPreTraining

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from downstream_data_downloader import (
    ClassificationDataConfig,
    ClassificationEvaluationDataModule,
    NLIEvaluationDataModule,
    QADataConfig,
    QAEvaluationDataModule,
    RetrievalDataConfig,
    RetrievalEvaluationDataModule,
    SNLIDataConfig,
)
from pretrain.common import build_bert_base_config, device_from_config, move_to_device
from pretrain.pretrained_Barlow_Twins import BarlowTwinsPretrainConfig, TextBarlowTwins
from pretrain.pretrained_BYOL import BYOLPretrainConfig, TextBYOL
from pretrain.pretrained_JEPA import JEPAPretrainConfig, TextJEPA
from pretrain.pretrained_VICReg import TextVICReg, VICRegPretrainConfig
from pretrain.pretrained_BERT import BERTPretrainConfig

METHODS = {"bert", "jepa", "byol", "vicreg", "barlow_twins"}

RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")


# ---------------------------------------------------------------------------
# Result persistence helpers
# ---------------------------------------------------------------------------

def save_results(
    metrics: dict,
    cfg: "DownstreamTrainConfig",
    timestamp: str,
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    payload = {
        "timestamp": timestamp,
        "method": cfg.method,
        "task": cfg.task,
        "checkpoint": cfg.checkpoint_path,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "metrics": metrics,
    }
    json_name = f"{cfg.task}_{cfg.method}_{timestamp}.json"
    json_path = os.path.join(RESULTS_DIR, json_name)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[results] JSON saved → {json_path}")

    row: dict = {
        "timestamp": timestamp,
        "method": cfg.method,
        "task": cfg.task,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        **{f"metric_{k}": v for k, v in metrics.items()},
        "checkpoint": cfg.checkpoint_path,
    }

    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[results] CSV  row  → {CSV_PATH}")


# ---------------------------------------------------------------------------
# Checkpoint / backbone utilities
# ---------------------------------------------------------------------------

@dataclass
class DownstreamTrainConfig:
    method: str
    checkpoint_path: str
    task: str
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    log_every: int = 50
    retrieval_temperature: float = 0.05
    retrieval_eval_k: int = 10
    max_eval_queries: int = 0
    max_eval_corpus: int = 0


def load_checkpoint_state(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model"] if isinstance(payload, dict) and "model" in payload else payload


def load_matching_state(model: nn.Module, checkpoint_state: dict) -> None:
    current = model.state_dict()
    filtered = {}
    for key, value in checkpoint_state.items():
        if key not in current:
            continue
        if current[key].shape == value.shape:
            filtered[key] = value
            continue
        if key.endswith("position_embeddings.weight") and current[key].dim() == value.dim() == 2:
            resized = current[key].clone()
            rows = min(resized.size(0), value.size(0))
            resized[:rows] = value[:rows]
            filtered[key] = resized
    model.load_state_dict(filtered, strict=False)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def load_pretrained_bert_backbone(
    method: str,
    checkpoint_path: str,
    max_length: int,
    device: torch.device,
) -> nn.Module:
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")

    state = load_checkpoint_state(checkpoint_path)

    if method == "bert":
        model = BertForPreTraining(build_bert_base_config(max_length=max_length))
        load_matching_state(model, state)
        backbone = model.bert
    elif method == "jepa":
        cfg = JEPAPretrainConfig()
        cfg.data.max_length = max_length
        model = TextJEPA(cfg)
        load_matching_state(model, state)
        backbone = model.context_encoder
    elif method == "byol":
        cfg = BYOLPretrainConfig()
        cfg.data.max_length = max_length
        model = TextBYOL(cfg)
        load_matching_state(model, state)
        backbone = model.online_encoder.bert
    elif method == "vicreg":
        cfg = VICRegPretrainConfig()
        cfg.data.max_length = max_length
        model = TextVICReg(cfg)
        load_matching_state(model, state)
        backbone = model.encoder.bert
    else:
        cfg = BarlowTwinsPretrainConfig()
        cfg.data.max_length = max_length
        model = TextBarlowTwins(cfg)
        load_matching_state(model, state)
        backbone = model.encoder.bert

    return freeze(backbone.to(device))


# ---------------------------------------------------------------------------
# Shared backbone wrapper
# ---------------------------------------------------------------------------

class FrozenBackbone(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def hidden_states(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return output.last_hidden_state

    @torch.no_grad()
    def mean(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        """Mean-pool over non-padding tokens: sum(h * mask) / sum(mask)."""
        h = self.hidden_states(input_ids, attention_mask, token_type_ids)  # [B, L, D]
        mask = attention_mask.unsqueeze(-1).float()                        # [B, L, 1]
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)    # [B, D]


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class LinearClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_labels: int = 2):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class LinearQAHead(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_dim, 2)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


class RetrievalProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int = 768, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(features), dim=-1)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class ClassificationEvaluator:
    def __init__(self, cfg: DownstreamTrainConfig):
        self.cfg = cfg
        self.data_cfg = ClassificationDataConfig()
        self.device = device_from_config(cfg.device)
        backbone = load_pretrained_bert_backbone(
            cfg.method, cfg.checkpoint_path, self.data_cfg.max_length, self.device,
        )
        self.backbone = FrozenBackbone(backbone)
        self.head = LinearClassificationHead(num_labels=2).to(self.device)
        self.data = ClassificationEvaluationDataModule(self.data_cfg)
        self.optimizer = AdamW(self.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train(self) -> None:
        loader = self.data.train_dataloader()
        self.head.train()
        for epoch in range(self.cfg.epochs):
            pbar = tqdm(loader, desc=f"classification head epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                batch = move_to_device(batch, self.device)
                features = self.backbone.mean(
                    batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"),
                )
                logits = self.head(features)
                loss = F.cross_entropy(logits, batch["labels"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"loss": f"{float(loss.detach()):.4f}"})

    @torch.no_grad()
    def evaluate(self) -> dict:
        loader = self.data.test_dataloader()
        self.head.eval()
        correct = total = 0
        total_loss = 0.0
        tp = fp = fn = 0
        for batch in loader:
            batch = move_to_device(batch, self.device)
            features = self.backbone.mean(
                batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"),
            )
            logits = self.head(features)
            total_loss += float(F.cross_entropy(logits, batch["labels"]).item())
            pred   = logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += int((pred == labels).sum().item())
            total   += int(labels.numel())
            tp += int(((pred == 1) & (labels == 1)).sum().item())
            fp += int(((pred == 1) & (labels == 0)).sum().item())
            fn += int(((pred == 0) & (labels == 1)).sum().item())
        precision = tp / max(1, tp + fp)
        recall    = tp / max(1, tp + fn)
        f1        = 2 * precision * recall / max(1e-8, precision + recall)
        return {
            "accuracy":  correct / max(1, total),
            "f1":        f1,
            "precision": precision,
            "recall":    recall,
            "loss":      total_loss / max(1, len(loader)),
        }


class NLIEvaluator:
    LABEL_NAMES = ("entailment", "neutral", "contradiction")

    def __init__(self, cfg: DownstreamTrainConfig):
        self.cfg = cfg
        self.data_cfg = SNLIDataConfig()
        self.device = device_from_config(cfg.device)
        backbone = load_pretrained_bert_backbone(
            cfg.method, cfg.checkpoint_path, self.data_cfg.max_length, self.device,
        )
        self.backbone = FrozenBackbone(backbone)
        self.head = LinearClassificationHead(hidden_dim=768, num_labels=self.data_cfg.num_labels).to(self.device)
        self.data = NLIEvaluationDataModule(self.data_cfg)
        self.optimizer = AdamW(self.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train(self) -> None:
        loader = self.data.train_dataloader()
        self.head.train()
        for epoch in range(self.cfg.epochs):
            pbar = tqdm(loader, desc=f"nli head epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                batch = move_to_device(batch, self.device)
                features = self.backbone.mean(
                    batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"),
                )
                logits = self.head(features)
                loss = F.cross_entropy(logits, batch["labels"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"loss": f"{float(loss.detach()):.4f}"})

    @torch.no_grad()
    def evaluate(self) -> dict:
        loader = self.data.test_dataloader()
        self.head.eval()
        num_labels = self.data_cfg.num_labels
        correct = total = 0
        total_loss = 0.0
        tp = [0] * num_labels
        fp = [0] * num_labels
        fn = [0] * num_labels
        for batch in loader:
            batch = move_to_device(batch, self.device)
            features = self.backbone.mean(
                batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"),
            )
            logits = self.head(features)
            total_loss += float(F.cross_entropy(logits, batch["labels"]).item())
            pred   = logits.argmax(dim=-1)
            labels = batch["labels"]
            correct += int((pred == labels).sum().item())
            total   += int(labels.numel())
            for cls_idx in range(num_labels):
                tp[cls_idx] += int(((pred == cls_idx) & (labels == cls_idx)).sum().item())
                fp[cls_idx] += int(((pred == cls_idx) & (labels != cls_idx)).sum().item())
                fn[cls_idx] += int(((pred != cls_idx) & (labels == cls_idx)).sum().item())
        metrics: dict = {
            "accuracy": correct / max(1, total),
            "loss": total_loss / max(1, len(loader)),
        }
        f1_scores = []
        for cls_idx, name in enumerate(self.LABEL_NAMES):
            precision = tp[cls_idx] / max(1, tp[cls_idx] + fp[cls_idx])
            recall    = tp[cls_idx] / max(1, tp[cls_idx] + fn[cls_idx])
            f1        = 2 * precision * recall / max(1e-8, precision + recall)
            metrics[f"precision_{name}"] = precision
            metrics[f"recall_{name}"]    = recall
            metrics[f"f1_{name}"]        = f1
            f1_scores.append(f1)
        metrics["f1_macro"] = sum(f1_scores) / len(f1_scores)
        return metrics


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens).intersection(gold_tokens)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if not common:
        return 0.0
    overlap = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    precision = overlap / len(pred_tokens)
    recall    = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


class QAEvaluator:
    def __init__(self, cfg: DownstreamTrainConfig):
        self.cfg = cfg
        self.data_cfg = QADataConfig()
        self.device = device_from_config(cfg.device)
        backbone = load_pretrained_bert_backbone(
            cfg.method, cfg.checkpoint_path, self.data_cfg.max_length, self.device,
        )
        self.backbone = FrozenBackbone(backbone)
        self.head = LinearQAHead().to(self.device)
        self.data = QAEvaluationDataModule(self.data_cfg)
        self.optimizer = AdamW(self.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train(self) -> None:
        loader = self.data.train_dataloader()
        self.head.train()
        for epoch in range(self.cfg.epochs):
            pbar = tqdm(loader, desc=f"qa head epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                batch = move_to_device(batch, self.device)
                hidden = self.backbone.hidden_states(
                    batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"),
                )
                start_logits, end_logits = self.head(hidden)
                start_loss = F.cross_entropy(start_logits, batch["start_positions"])
                end_loss   = F.cross_entropy(end_logits,   batch["end_positions"])
                loss = 0.5 * (start_loss + end_loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"loss": f"{float(loss.detach()):.4f}"})

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate QA on the SQuAD validation set.

        Bug 3 FIX — constrained best-span search:
            Instead of taking argmax of start and end logits independently
            (which can produce end < start or spans that cross into the
            question/padding), we now:
              1. Collect all valid context token indices (where offset != None).
              2. Find the best (start, end) pair with end >= start by maximising
                 start_logits[i] + end_logits[j] over valid context positions.
            This matches the standard SQuAD inference procedure and ensures
            predictions are always valid answer spans.

        Bug 4 FIX — seq_len from Python list length:
            len(offsets) gives the actual padded sequence length, not the
            tensor dimension, preventing index-out-of-range errors.
        """
        loader = self.data.test_dataloader()
        self.head.eval()
        exact = 0.0
        f1    = 0.0
        total = 0

        for batch in loader:
            tensor_batch = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items()
                if torch.is_tensor(v)
            }
            hidden = self.backbone.hidden_states(
                tensor_batch["input_ids"],
                tensor_batch["attention_mask"],
                tensor_batch.get("token_type_ids"),
            )
            start_logits, end_logits = self.head(hidden)
            start_logits_cpu = start_logits.cpu()
            end_logits_cpu   = end_logits.cpu()

            batch_size = start_logits_cpu.size(0)
            for index in range(batch_size):
                offsets = batch["offset_mapping"][index]
                context = batch["contexts"][index]
                answers = batch["answers"][index]["text"]
                seq_len = len(offsets)                     # Bug 4 FIX

                valid_positions = [
                    pos for pos in range(seq_len)
                    if offsets[pos] is not None
                ]

                if not valid_positions:
                    prediction = ""
                else:
                    s_logits = start_logits_cpu[index]
                    e_logits = end_logits_cpu[index]

                    best_score = float("-inf")
                    best_start = valid_positions[0]
                    best_end   = valid_positions[0]

                    for i, s in enumerate(valid_positions):
                        s_score = float(s_logits[s])
                        for j in range(i, len(valid_positions)):
                            e = valid_positions[j]
                            score = s_score + float(e_logits[e])
                            if score > best_score:
                                best_score = score
                                best_start = s
                                best_end   = e

                    char_start = offsets[best_start][0]
                    char_end   = offsets[best_end][1]
                    prediction = context[char_start:char_end]

                gold_scores = [f1_score(prediction, ans) for ans in answers]
                gold_exact  = [
                    normalize_answer(prediction) == normalize_answer(ans)
                    for ans in answers
                ]
                f1    += max(gold_scores) if gold_scores else 0.0
                exact += float(any(gold_exact))
                total += 1

        return {"exact_match": exact / max(1, total), "f1": f1 / max(1, total)}


class RetrievalEvaluator:
    def __init__(self, cfg: DownstreamTrainConfig):
        self.cfg = cfg
        self.data_cfg = RetrievalDataConfig()
        self.device = device_from_config(cfg.device)
        max_length = max(self.data_cfg.query_max_length, self.data_cfg.document_max_length)
        backbone = load_pretrained_bert_backbone(
            cfg.method, cfg.checkpoint_path, max_length, self.device,
        )
        self.backbone = FrozenBackbone(backbone)
        self.head = RetrievalProjectionHead().to(self.device)
        self.data = RetrievalEvaluationDataModule(self.data_cfg)
        self.optimizer = AdamW(self.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def encode_mean(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        return self.backbone.mean(input_ids, attention_mask, token_type_ids)

    def train(self) -> None:
        loader = self.data.train_pair_dataloader()
        self.head.train()
        for epoch in range(self.cfg.epochs):
            pbar = tqdm(loader, desc=f"retrieval head epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                batch = move_to_device(batch, self.device)
                q = self.head(self.encode_mean(
                    batch["query_input_ids"], batch["query_attention_mask"], batch.get("query_token_type_ids"),
                ))
                d = self.head(self.encode_mean(
                    batch["document_input_ids"], batch["document_attention_mask"], batch.get("document_token_type_ids"),
                ))
                logits = (q @ d.t()) / self.cfg.retrieval_temperature
                labels = torch.arange(logits.size(0), device=self.device)
                loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"loss": f"{float(loss.detach()):.4f}"})

    @torch.no_grad()
    def embed_texts(self, texts: list[str], max_length: int) -> torch.Tensor:
        tokenizer = self.data.collator.tokenizer
        embeddings = []
        for start in range(0, len(texts), self.data_cfg.batch_size):
            batch_texts = texts[start : start + self.data_cfg.batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            encoded = move_to_device(encoded, self.device)
            pooled = self.encode_mean(
                encoded["input_ids"], encoded["attention_mask"], encoded.get("token_type_ids"),
            )
            embeddings.append(self.head(pooled).detach().cpu())
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 256)

    @staticmethod
    def document_text(row: dict) -> str:
        title = row.get("title", "")
        text  = row.get("text", "")
        return f"{title}. {text}" if title else text

    @staticmethod
    def relevant_documents(qrels) -> dict[str, set[str]]:
        relevant: dict[str, set[str]] = {}
        for row in qrels:
            if float(row["score"]) <= 0.0:
                continue
            query_id  = str(row["query-id"]).strip('"')
            corpus_id = str(row["corpus-id"]).strip('"')
            relevant.setdefault(query_id, set()).add(corpus_id)
        return relevant

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.head.eval()
        dataset  = self.data.datasets()
        relevant = self.relevant_documents(dataset["test"])

        query_index = {str(row["_id"]).strip('"'): index for index, row in enumerate(dataset["queries"])}
        query_ids = [qid for qid in relevant.keys() if qid in query_index]
        if self.cfg.max_eval_queries > 0:
            query_ids = query_ids[: self.cfg.max_eval_queries]

        query_texts = [dataset["queries"][query_index[qid]]["text"] for qid in query_ids]
        query_embeddings = self.embed_texts(query_texts, self.data_cfg.query_max_length).to(self.device)

        k = self.cfg.retrieval_eval_k
        top_scores  = torch.full((len(query_ids), k), -float("inf"), device=self.device)
        top_indices = torch.full((len(query_ids), k), -1, dtype=torch.long, device=self.device)

        corpus      = dataset["corpus"]
        corpus_size = len(corpus)
        if self.cfg.max_eval_corpus > 0:
            corpus_size = min(corpus_size, self.cfg.max_eval_corpus)

        pbar = tqdm(range(0, corpus_size, self.data_cfg.batch_size), desc="retrieval eval corpus", ncols=120)
        for start in pbar:
            end  = min(start + self.data_cfg.batch_size, corpus_size)
            rows = [corpus[i] for i in range(start, end)]
            texts = [self.document_text(row) for row in rows]
            doc_embeddings = self.embed_texts(texts, self.data_cfg.document_max_length).to(self.device)
            scores  = query_embeddings @ doc_embeddings.t()
            local_k = min(k, scores.size(1))
            local_scores, local_positions = scores.topk(local_k, dim=1)
            local_indices = local_positions + start

            combined_scores  = torch.cat([top_scores,  local_scores],  dim=1)
            combined_indices = torch.cat([top_indices, local_indices], dim=1)
            top_scores, selected = combined_scores.topk(k, dim=1)
            top_indices = combined_indices.gather(1, selected)

        corpus_ids = [str(corpus[i]["_id"]).strip('"') for i in range(corpus_size)]
        hits = 0
        reciprocal_rank = 0.0
        evaluated = 0
        for row_idx, query_id in enumerate(query_ids):
            retrieved = [
                corpus_ids[idx]
                for idx in top_indices[row_idx].cpu().tolist()
                if 0 <= idx < len(corpus_ids)
            ]
            gold = relevant[query_id]
            hit_positions = [rank for rank, cid in enumerate(retrieved, start=1) if cid in gold]
            if hit_positions:
                hits += 1
                reciprocal_rank += 1.0 / min(hit_positions)
            evaluated += 1

        return {
            f"recall@{k}": hits / max(1, evaluated),
            f"mrr@{k}":    reciprocal_rank / max(1, evaluated),
            "queries":     evaluated,
            "corpus":      corpus_size,
        }


# ---------------------------------------------------------------------------
# Factory + entry point
# ---------------------------------------------------------------------------

def build_evaluator(cfg: DownstreamTrainConfig):
    if cfg.task == "classification":
        return ClassificationEvaluator(cfg)
    if cfg.task == "nli":
        return NLIEvaluator(cfg)
    if cfg.task == "qa":
        return QAEvaluator(cfg)
    if cfg.task == "retrieval":
        return RetrievalEvaluator(cfg)
    raise ValueError(f"Unsupported downstream task: {cfg.task}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-backbone downstream evaluation.")
    parser.add_argument("--task",     choices=["classification", "nli", "qa", "retrieval"], required=True)
    parser.add_argument("--method",   choices=sorted(METHODS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs",   type=int,   default=5)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--retrieval-eval-k",    type=int, default=10)
    parser.add_argument("--max-eval-queries",    type=int, default=0)
    parser.add_argument("--max-eval-corpus",     type=int, default=0)
    args = parser.parse_args()

    cfg = DownstreamTrainConfig(
        method=args.method,
        checkpoint_path=args.checkpoint,
        task=args.task,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        retrieval_eval_k=args.retrieval_eval_k,
        max_eval_queries=args.max_eval_queries,
        max_eval_corpus=args.max_eval_corpus,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluator = build_evaluator(cfg)
    evaluator.train()
    metrics = evaluator.evaluate()

    print(metrics)
    save_results(metrics, cfg, timestamp)


if __name__ == "__main__":
    main()