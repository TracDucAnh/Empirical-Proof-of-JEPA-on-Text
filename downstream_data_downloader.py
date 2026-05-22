"""Downstream evaluation data downloaders and dataloaders.

Tasks currently covered:
    1. Sentiment Classification:
       `stanfordnlp/imdb`.
       Splits: train -> train, test -> test.

    2. Extractive Question Answering:
       `rajpurkar/squad`, subset `plain_text`.
       Splits: train -> train, validation -> test/eval.

    3. Information Retrieval:
       `mteb/fever`.
       Configs: default qrels, corpus documents, queries.
       Splits: qrels train -> train, qrels test -> test/eval.
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ClassificationDataConfig:
    dataset_name: str = "stanfordnlp/imdb"
    subset_name: str = "plain_text"
    train_split: str = "train"
    test_split: str = "test"
    cache_dir: str = os.path.join(PROJECT_ROOT, "downstream_cache", "imdb_plain_text")
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 64
    num_workers: int = 2
    seed: int = 42


class IMDBDataDownloader:
    """Download and cache IMDB sentiment-classification train/test splits."""

    required_columns = {"text", "label"}
    label_names = ["neg", "pos"]

    def __init__(self, cfg: ClassificationDataConfig):
        self.cfg = cfg

    def download(self) -> DatasetDict:
        train = load_dataset(
            self.cfg.dataset_name,
            self.cfg.subset_name,
            split=self.cfg.train_split,
        )
        test = load_dataset(
            self.cfg.dataset_name,
            self.cfg.subset_name,
            split=self.cfg.test_split,
        )
        dataset = DatasetDict({"train": train, "test": test})
        self.validate(dataset)
        return dataset

    def validate(self, dataset: DatasetDict) -> None:
        for split_name, split in dataset.items():
            missing = self.required_columns.difference(split.column_names)
            if missing:
                raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")

    def load_or_download(self) -> DatasetDict:
        if os.path.isdir(self.cfg.cache_dir):
            dataset = load_from_disk(self.cfg.cache_dir)
            self.validate(dataset)
            return dataset

        dataset = self.download()
        os.makedirs(os.path.dirname(self.cfg.cache_dir), exist_ok=True)
        dataset.save_to_disk(self.cfg.cache_dir)
        return dataset


class TextClassificationCollator:
    """Tokenize single-text examples for a BERT-style classifier."""

    def __init__(self, tokenizer_name: str, max_length: int):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict:
        texts = [item["text"] for item in examples]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(item["label"]) for item in examples], dtype=torch.long)
        return encoded


class ClassificationEvaluationDataModule:
    """Data module for downstream IMDB sentiment classification."""

    def __init__(self, cfg: Optional[ClassificationDataConfig] = None):
        self.cfg = cfg or ClassificationDataConfig()
        self.downloader = IMDBDataDownloader(self.cfg)
        self.collator = TextClassificationCollator(
            tokenizer_name=self.cfg.tokenizer_name,
            max_length=self.cfg.max_length,
        )

    def datasets(self) -> DatasetDict:
        return self.downloader.load_or_download()

    def train_dataloader(self) -> DataLoader:
        dataset = self.datasets()["train"]
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collator,
            generator=generator,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.datasets()["test"]
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collator,
        )


@dataclass
class QADataConfig:
    dataset_name: str = "rajpurkar/squad"
    subset_name: str = "plain_text"
    train_split: str = "train"
    test_split: str = "validation"
    cache_dir: str = os.path.join(PROJECT_ROOT, "downstream_cache", "squad_plain_text")
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = 384
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42


class SquadDataDownloader:
    """Download and cache SQuAD train/validation splits for extractive QA."""

    required_columns = {"id", "title", "context", "question", "answers"}

    def __init__(self, cfg: QADataConfig):
        self.cfg = cfg

    def download(self) -> DatasetDict:
        train = load_dataset(
            self.cfg.dataset_name,
            self.cfg.subset_name,
            split=self.cfg.train_split,
        )
        test = load_dataset(
            self.cfg.dataset_name,
            self.cfg.subset_name,
            split=self.cfg.test_split,
        )
        dataset = DatasetDict({"train": train, "test": test})
        self.validate(dataset)
        return dataset

    def validate(self, dataset: DatasetDict) -> None:
        for split_name, split in dataset.items():
            missing = self.required_columns.difference(split.column_names)
            if missing:
                raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")

    def load_or_download(self) -> DatasetDict:
        if os.path.isdir(self.cfg.cache_dir):
            dataset = load_from_disk(self.cfg.cache_dir)
            self.validate(dataset)
            return dataset

        dataset = self.download()
        os.makedirs(os.path.dirname(self.cfg.cache_dir), exist_ok=True)
        dataset.save_to_disk(self.cfg.cache_dir)
        return dataset


class SquadQACollator:
    """Tokenize question/context pairs and map character answers to token spans."""

    def __init__(self, tokenizer_name: str, max_length: int, include_metadata: bool = False):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.include_metadata = include_metadata

    def __call__(self, examples: list[dict]) -> dict:
        questions = [item["question"].strip() for item in examples]
        contexts = [item["context"] for item in examples]
        encoded = self.tokenizer(
            questions,
            contexts,
            padding=True,
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping")
        start_positions, end_positions = self.answer_token_positions(examples, encoded, offset_mapping)
        encoded["start_positions"] = torch.tensor(start_positions, dtype=torch.long)
        encoded["end_positions"] = torch.tensor(end_positions, dtype=torch.long)

        if self.include_metadata:
            encoded["example_ids"] = [item["id"] for item in examples]
            encoded["questions"] = questions
            encoded["contexts"] = contexts
            encoded["answers"] = [item["answers"] for item in examples]
            encoded["offset_mapping"] = self.context_offsets(encoded, offset_mapping)

        return encoded

    def answer_token_positions(
        self,
        examples: list[dict],
        encoded,
        offset_mapping: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        start_positions: list[int] = []
        end_positions: list[int] = []

        for index, item in enumerate(examples):
            cls_index = int((encoded["input_ids"][index] == self.tokenizer.cls_token_id).nonzero()[0])
            answers = item["answers"]
            answer_texts = answers["text"]
            answer_starts = answers["answer_start"]

            if not answer_texts:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            answer_start = int(answer_starts[0])
            answer_end = answer_start + len(answer_texts[0])
            sequence_ids = encoded.sequence_ids(index)
            context_start = self.first_context_token(sequence_ids)
            context_end = self.last_context_token(sequence_ids)

            if context_start is None or context_end is None:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            offsets = offset_mapping[index]
            if offsets[context_start][0] > answer_start or offsets[context_end][1] < answer_end:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start:
                token_start += 1
            start_positions.append(token_start - 1)

            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= answer_end:
                token_end -= 1
            end_positions.append(token_end + 1)

        return start_positions, end_positions

    @staticmethod
    def first_context_token(sequence_ids: list[Optional[int]]) -> Optional[int]:
        for index, sequence_id in enumerate(sequence_ids):
            if sequence_id == 1:
                return index
        return None

    @staticmethod
    def last_context_token(sequence_ids: list[Optional[int]]) -> Optional[int]:
        for index in range(len(sequence_ids) - 1, -1, -1):
            if sequence_ids[index] == 1:
                return index
        return None

    def context_offsets(self, encoded, offset_mapping: torch.Tensor) -> list[list[Optional[tuple[int, int]]]]:
        all_offsets: list[list[Optional[tuple[int, int]]]] = []
        for index in range(offset_mapping.size(0)):
            sequence_ids = encoded.sequence_ids(index)
            row_offsets: list[Optional[tuple[int, int]]] = []
            for token_index, offset in enumerate(offset_mapping[index].tolist()):
                if sequence_ids[token_index] == 1:
                    row_offsets.append((int(offset[0]), int(offset[1])))
                else:
                    row_offsets.append(None)
            all_offsets.append(row_offsets)
        return all_offsets


class QAEvaluationDataModule:
    """Data module for extractive QA fine-tuning/evaluation."""

    def __init__(self, cfg: Optional[QADataConfig] = None):
        self.cfg = cfg or QADataConfig()
        self.downloader = SquadDataDownloader(self.cfg)
        self.train_collator = SquadQACollator(
            tokenizer_name=self.cfg.tokenizer_name,
            max_length=self.cfg.max_length,
            include_metadata=False,
        )
        self.test_collator = SquadQACollator(
            tokenizer_name=self.cfg.tokenizer_name,
            max_length=self.cfg.max_length,
            include_metadata=True,
        )

    def datasets(self) -> DatasetDict:
        return self.downloader.load_or_download()

    def train_dataloader(self) -> DataLoader:
        dataset = self.datasets()["train"]
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.train_collator,
            generator=generator,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self.datasets()["test"]
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.test_collator,
        )


@dataclass
class RetrievalDataConfig:
    dataset_name: str = "mteb/fever"
    qrels_config: str = "default"
    corpus_config: str = "corpus"
    queries_config: str = "queries"
    train_split: str = "train"
    test_split: str = "test"
    corpus_split: str = "corpus"
    queries_split: str = "queries"
    cache_dir: str = os.path.join(PROJECT_ROOT, "downstream_cache", "mteb_fever")
    tokenizer_name: str = "bert-base-uncased"
    query_max_length: int = 128
    document_max_length: int = 384
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42


class FeverRetrievalDataDownloader:
    """Download and cache MTEB FEVER qrels, corpus, and queries."""

    qrels_columns = {"query-id", "corpus-id", "score"}
    corpus_columns = {"_id", "title", "text"}
    queries_columns = {"_id", "text"}

    def __init__(self, cfg: RetrievalDataConfig):
        self.cfg = cfg

    def download(self) -> DatasetDict:
        train_qrels = load_dataset(
            self.cfg.dataset_name,
            self.cfg.qrels_config,
            split=self.cfg.train_split,
        )
        test_qrels = load_dataset(
            self.cfg.dataset_name,
            self.cfg.qrels_config,
            split=self.cfg.test_split,
        )
        corpus = load_dataset(
            self.cfg.dataset_name,
            self.cfg.corpus_config,
            split=self.cfg.corpus_split,
        )
        queries = load_dataset(
            self.cfg.dataset_name,
            self.cfg.queries_config,
            split=self.cfg.queries_split,
        )
        dataset = DatasetDict(
            {
                "train": train_qrels,
                "test": test_qrels,
                "corpus": corpus,
                "queries": queries,
            }
        )
        self.validate(dataset)
        return dataset

    def validate(self, dataset: DatasetDict) -> None:
        expected = {
            "train": self.qrels_columns,
            "test": self.qrels_columns,
            "corpus": self.corpus_columns,
            "queries": self.queries_columns,
        }
        for split_name, columns in expected.items():
            missing = columns.difference(dataset[split_name].column_names)
            if missing:
                raise ValueError(f"{split_name} split is missing columns: {sorted(missing)}")

    def load_or_download(self) -> DatasetDict:
        if os.path.isdir(self.cfg.cache_dir):
            dataset = load_from_disk(self.cfg.cache_dir)
            self.validate(dataset)
            return dataset

        dataset = self.download()
        os.makedirs(os.path.dirname(self.cfg.cache_dir), exist_ok=True)
        dataset.save_to_disk(self.cfg.cache_dir)
        return dataset


class FeverPositivePairDataset(Dataset):
    """Positive query-document pairs from FEVER qrels.

    This dataset is intended for supervised retrieval fine-tuning.  Full-corpus
    retrieval evaluation should use `RetrievalEvaluationDataModule.datasets()`
    directly so the evaluator can embed all queries and corpus documents.
    """

    def __init__(self, qrels, queries, corpus):
        self.qrels = qrels
        self.queries = queries
        self.corpus = corpus
        self.query_index = self.build_index(queries)
        self.corpus_index = self.build_index(corpus)

    @staticmethod
    def build_index(dataset) -> dict[str, int]:
        return {str(row["_id"]): index for index, row in enumerate(dataset)}

    def __len__(self) -> int:
        return len(self.qrels)

    def __getitem__(self, index: int) -> dict:
        rel = self.qrels[index]
        query_id = str(rel["query-id"])
        corpus_id = str(rel["corpus-id"])
        query = self.queries[self.query_index[query_id]]
        document = self.corpus[self.corpus_index[corpus_id]]
        title = document.get("title", "")
        text = document.get("text", "")
        document_text = f"{title}. {text}" if title else text
        return {
            "query_id": query_id,
            "corpus_id": corpus_id,
            "query": query["text"],
            "document": document_text,
            "score": float(rel["score"]),
        }


class RetrievalPairCollator:
    """Tokenize positive query-document pairs for retrieval training."""

    def __init__(self, tokenizer_name: str, query_max_length: int, document_max_length: int):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length

    def __call__(self, examples: list[dict]) -> dict:
        query_tokens = self.tokenizer(
            [item["query"] for item in examples],
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )
        document_tokens = self.tokenizer(
            [item["document"] for item in examples],
            padding=True,
            truncation=True,
            max_length=self.document_max_length,
            return_tensors="pt",
        )
        return {
            "query_input_ids": query_tokens["input_ids"],
            "query_attention_mask": query_tokens["attention_mask"],
            "query_token_type_ids": query_tokens.get("token_type_ids"),
            "document_input_ids": document_tokens["input_ids"],
            "document_attention_mask": document_tokens["attention_mask"],
            "document_token_type_ids": document_tokens.get("token_type_ids"),
            "scores": torch.tensor([float(item["score"]) for item in examples], dtype=torch.float),
            "query_ids": [item["query_id"] for item in examples],
            "corpus_ids": [item["corpus_id"] for item in examples],
        }


class RetrievalEvaluationDataModule:
    """Data module for FEVER information retrieval."""

    def __init__(self, cfg: Optional[RetrievalDataConfig] = None):
        self.cfg = cfg or RetrievalDataConfig()
        self.downloader = FeverRetrievalDataDownloader(self.cfg)
        self.collator = RetrievalPairCollator(
            tokenizer_name=self.cfg.tokenizer_name,
            query_max_length=self.cfg.query_max_length,
            document_max_length=self.cfg.document_max_length,
        )

    def datasets(self) -> DatasetDict:
        return self.downloader.load_or_download()

    def train_pair_dataloader(self) -> DataLoader:
        dataset = self.datasets()
        pair_dataset = FeverPositivePairDataset(
            qrels=dataset["train"],
            queries=dataset["queries"],
            corpus=dataset["corpus"],
        )
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            pair_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collator,
            generator=generator,
        )

    def test_pair_dataloader(self) -> DataLoader:
        dataset = self.datasets()
        pair_dataset = FeverPositivePairDataset(
            qrels=dataset["test"],
            queries=dataset["queries"],
            corpus=dataset["corpus"],
        )
        return DataLoader(
            pair_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collator,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download downstream evaluation datasets.")
    parser.add_argument(
        "--task",
        choices=["classification", "imdb", "qa", "retrieval", "ir", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.task in {"classification", "imdb", "all"}:
        classification_cfg = ClassificationDataConfig()
        classification_module = ClassificationEvaluationDataModule(classification_cfg)
        classification_dataset = classification_module.datasets()
        print(f"IMDB train rows: {len(classification_dataset['train']):,}")
        print(f"IMDB test rows: {len(classification_dataset['test']):,}")
        print(f"IMDB columns: {classification_dataset['train'].column_names}")
        print(f"IMDB labels: {IMDBDataDownloader.label_names}")
        print(f"IMDB cache: {classification_cfg.cache_dir}")

    if args.task in {"qa", "all"}:
        qa_cfg = QADataConfig()
        qa_module = QAEvaluationDataModule(qa_cfg)
        qa_dataset = qa_module.datasets()
        print(f"SQuAD train rows: {len(qa_dataset['train']):,}")
        print(f"SQuAD test rows: {len(qa_dataset['test']):,}")
        print(f"SQuAD columns: {qa_dataset['train'].column_names}")
        print(f"SQuAD cache: {qa_cfg.cache_dir}")

    if args.task in {"retrieval", "ir", "all"}:
        retrieval_cfg = RetrievalDataConfig()
        retrieval_module = RetrievalEvaluationDataModule(retrieval_cfg)
        retrieval_dataset = retrieval_module.datasets()
        print(f"FEVER train qrels rows: {len(retrieval_dataset['train']):,}")
        print(f"FEVER test qrels rows: {len(retrieval_dataset['test']):,}")
        print(f"FEVER corpus rows: {len(retrieval_dataset['corpus']):,}")
        print(f"FEVER queries rows: {len(retrieval_dataset['queries']):,}")
        print(f"FEVER qrels columns: {retrieval_dataset['train'].column_names}")
        print(f"FEVER corpus columns: {retrieval_dataset['corpus'].column_names}")
        print(f"FEVER queries columns: {retrieval_dataset['queries'].column_names}")
        print(f"FEVER cache: {retrieval_cfg.cache_dir}")


if __name__ == "__main__":
    main()
