"""Frozen-backbone downstream evaluation.

Protocol:
    1. Choose a pretrained method checkpoint.
    2. Load that method's encoder/backbone.
    3. Freeze every backbone parameter.
    4. Train only the downstream task head on that task's train split.
    5. Evaluate on that task's test/eval split.

Pretraining data is never used here.  Each downstream task owns its train/test
data through `downstream_data_downloader.py`.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
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
    QADataConfig,
    QAEvaluationDataModule,
    RetrievalDataConfig,
    RetrievalEvaluationDataModule,
)
from pretrain.common import build_bert_base_config, device_from_config, move_to_device
from pretrain.pretrained_Barlow_Twins import BarlowTwinsPretrainConfig, TextBarlowTwins
from pretrain.pretrained_BYOL import BYOLPretrainConfig, TextBYOL
from pretrain.pretrained_JEPA import JEPAPretrainConfig, TextJEPA
from pretrain.pretrained_VICReg import TextVICReg, VICRegPretrainConfig
from pretrain.pretrained_BERT import BERTPretrainConfig

METHODS = {"bert", "jepa", "byol", "vicreg", "barlow_twins"}


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
    def cls(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        return self.hidden_states(input_ids, attention_mask, token_type_ids)[:, 0]


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


class ClassificationEvaluator:
    def __init__(self, cfg: DownstreamTrainConfig):
        self.cfg = cfg
        self.data_cfg = ClassificationDataConfig()
        self.device = device_from_config(cfg.device)
        backbone = load_pretrained_bert_backbone(
            cfg.method,
            cfg.checkpoint_path,
            self.data_cfg.max_length,
            self.device,
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
                features = self.backbone.cls(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("token_type_ids"),
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
        correct = 0
        total = 0
        total_loss = 0.0
        tp = 0  # true  positives  (pred=1, label=1)
        fp = 0  # false positives  (pred=1, label=0)
        fn = 0  # false negatives  (pred=0, label=1)
        for batch in loader:
            batch = move_to_device(batch, self.device)
            features = self.backbone.cls(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("token_type_ids"),
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
    overlap = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in common)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


class QAEvaluator:
    def __init__(self, cfg: DownstreamTrainConfig):
        self.cfg = cfg
        self.data_cfg = QADataConfig()
        self.device = device_from_config(cfg.device)
        backbone = load_pretrained_bert_backbone(
            cfg.method,
            cfg.checkpoint_path,
            self.data_cfg.max_length,
            self.device,
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
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch.get("token_type_ids"),
                )
                start_logits, end_logits = self.head(hidden)
                start_loss = F.cross_entropy(start_logits, batch["start_positions"])
                end_loss = F.cross_entropy(end_logits, batch["end_positions"])
                loss = 0.5 * (start_loss + end_loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"loss": f"{float(loss.detach()):.4f}"})

    @torch.no_grad()
    def evaluate(self) -> dict:
        loader = self.data.test_dataloader()
        self.head.eval()
        exact = 0.0
        f1 = 0.0
        total = 0
        for batch in loader:
            tensor_batch = {
                key: value.to(self.device, non_blocking=True)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            hidden = self.backbone.hidden_states(
                tensor_batch["input_ids"],
                tensor_batch["attention_mask"],
                tensor_batch.get("token_type_ids"),
            )
            start_logits, end_logits = self.head(hidden)
            starts = start_logits.argmax(dim=-1).cpu().tolist()
            ends = end_logits.argmax(dim=-1).cpu().tolist()

            for index, (start, end) in enumerate(zip(starts, ends)):
                offsets = batch["offset_mapping"][index]
                context = batch["contexts"][index]
                answers = batch["answers"][index]["text"]
                if end < start or start >= len(offsets) or end >= len(offsets):
                    prediction = ""
                elif offsets[start] is None or offsets[end] is None:
                    prediction = ""
                else:
                    char_start = offsets[start][0]
                    char_end = offsets[end][1]
                    prediction = context[char_start:char_end]

                gold_scores = [f1_score(prediction, answer) for answer in answers]
                gold_exact = [normalize_answer(prediction) == normalize_answer(answer) for answer in answers]
                f1 += max(gold_scores) if gold_scores else 0.0
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
            cfg.method,
            cfg.checkpoint_path,
            max_length,
            self.device,
        )
        self.backbone = FrozenBackbone(backbone)
        self.head = RetrievalProjectionHead().to(self.device)
        self.data = RetrievalEvaluationDataModule(self.data_cfg)
        self.optimizer = AdamW(self.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def encode_cls(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        return self.backbone.cls(input_ids, attention_mask, token_type_ids)

    def train(self) -> None:
        loader = self.data.train_pair_dataloader()
        self.head.train()
        for epoch in range(self.cfg.epochs):
            pbar = tqdm(loader, desc=f"retrieval head epoch {epoch + 1}", ncols=120)
            for batch in pbar:
                batch = move_to_device(batch, self.device)
                q = self.head(self.encode_cls(
                    batch["query_input_ids"],
                    batch["query_attention_mask"],
                    batch.get("query_token_type_ids"),
                ))
                d = self.head(self.encode_cls(
                    batch["document_input_ids"],
                    batch["document_attention_mask"],
                    batch.get("document_token_type_ids"),
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
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = move_to_device(encoded, self.device)
            cls = self.encode_cls(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded.get("token_type_ids"),
            )
            embeddings.append(self.head(cls).detach().cpu())
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 256)

    @staticmethod
    def document_text(row: dict) -> str:
        title = row.get("title", "")
        text = row.get("text", "")
        return f"{title}. {text}" if title else text

    @staticmethod
    def relevant_documents(qrels) -> dict[str, set[str]]:
        relevant: dict[str, set[str]] = {}
        for row in qrels:
            if float(row["score"]) <= 0.0:
                continue
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            relevant.setdefault(query_id, set()).add(corpus_id)
        return relevant

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.head.eval()
        dataset = self.data.datasets()
        relevant = self.relevant_documents(dataset["test"])
        query_ids = list(relevant.keys())
        if self.cfg.max_eval_queries > 0:
            query_ids = query_ids[: self.cfg.max_eval_queries]

        query_index = {str(row["_id"]): index for index, row in enumerate(dataset["queries"])}
        query_texts = [dataset["queries"][query_index[query_id]]["text"] for query_id in query_ids]
        query_embeddings = self.embed_texts(query_texts, self.data_cfg.query_max_length).to(self.device)

        k = self.cfg.retrieval_eval_k
        top_scores = torch.full((len(query_ids), k), -float("inf"), device=self.device)
        top_indices = torch.full((len(query_ids), k), -1, dtype=torch.long, device=self.device)

        corpus = dataset["corpus"]
        corpus_size = len(corpus)
        if self.cfg.max_eval_corpus > 0:
            corpus_size = min(corpus_size, self.cfg.max_eval_corpus)

        pbar = tqdm(range(0, corpus_size, self.data_cfg.batch_size), desc="retrieval eval corpus", ncols=120)
        for start in pbar:
            end = min(start + self.data_cfg.batch_size, corpus_size)
            rows = [corpus[index] for index in range(start, end)]
            texts = [self.document_text(row) for row in rows]
            doc_embeddings = self.embed_texts(texts, self.data_cfg.document_max_length).to(self.device)
            scores = query_embeddings @ doc_embeddings.t()
            local_k = min(k, scores.size(1))
            local_scores, local_positions = scores.topk(local_k, dim=1)
            local_indices = local_positions + start

            combined_scores = torch.cat([top_scores, local_scores], dim=1)
            combined_indices = torch.cat([top_indices, local_indices], dim=1)
            top_scores, selected = combined_scores.topk(k, dim=1)
            top_indices = combined_indices.gather(1, selected)

        corpus_ids = [str(corpus[index]["_id"]) for index in range(corpus_size)]
        hits = 0
        reciprocal_rank = 0.0
        evaluated = 0
        for row_idx, query_id in enumerate(query_ids):
            retrieved = [
                corpus_ids[index]
                for index in top_indices[row_idx].cpu().tolist()
                if 0 <= index < len(corpus_ids)
            ]
            gold = relevant[query_id]
            hit_positions = [rank for rank, corpus_id in enumerate(retrieved, start=1) if corpus_id in gold]
            if hit_positions:
                hits += 1
                reciprocal_rank += 1.0 / min(hit_positions)
            evaluated += 1

        return {
            f"recall@{k}": hits / max(1, evaluated),
            f"mrr@{k}": reciprocal_rank / max(1, evaluated),
            "queries": evaluated,
            "corpus": corpus_size,
        }


def build_evaluator(cfg: DownstreamTrainConfig):
    if cfg.task == "classification":
        return ClassificationEvaluator(cfg)
    if cfg.task == "qa":
        return QAEvaluator(cfg)
    if cfg.task == "retrieval":
        return RetrievalEvaluator(cfg)
    raise ValueError(f"Unsupported downstream task: {cfg.task}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-backbone downstream evaluation.")
    parser.add_argument("--task", choices=["classification", "qa", "retrieval"], required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrieval-eval-k", type=int, default=10)
    parser.add_argument("--max-eval-queries", type=int, default=0)
    parser.add_argument("--max-eval-corpus", type=int, default=0)
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
    evaluator = build_evaluator(cfg)
    evaluator.train()
    metrics = evaluator.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
