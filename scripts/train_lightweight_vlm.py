#!/usr/bin/env python3
"""Train the fast Adaptive NeSy-Gen drafting model.

The default model combines a 5.7M-parameter DeiT-tiny image encoder with an
82M-parameter DistilGPT-2 decoder. The encoder stays frozen; training updates the
decoder, the newly initialized cross-attention, and the encoder-to-decoder projection.
This is intended for a single Colab GPU and resumes automatically.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from adaptive_nesy_gen.backends import lightweight_prompt
from adaptive_nesy_gen.retrieval import load_manifest
from adaptive_nesy_gen.schema import RetrievedStudy


def _training_embeddings(path: Path, expected_rows: int) -> np.ndarray:
    """Load sample-major or legacy-transposed MedSigLIP training embeddings."""
    archive = np.load(path, allow_pickle=False)
    key = next(
        (name for name in ("embeddings", "image_embeddings", "features") if name in archive),
        None,
    )
    if key is None:
        raise ValueError(f"No embeddings found in {path}; keys={archive.files}")
    matrix = np.asarray(archive[key], dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2-D, got {matrix.shape}")
    if matrix.shape[0] != expected_rows and matrix.shape[1] == expected_rows:
        matrix = matrix.T
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            f"MedSigLIP cache/train mismatch: {matrix.shape} for {expected_rows} rows"
        )
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    return matrix


def neighbour_indices(
    embeddings_path: Path,
    studies,
    cache_path: Path,
    max_neighbours: int = 5,
) -> np.ndarray:
    """Build/reuse a fast unique-study FAISS neighbour table for training RAG."""
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)["neighbours"]
        if cached.shape == (len(studies), max_neighbours):
            return cached
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Install adaptive-nesy-gen[lightweight] for FAISS") from exc
    embeddings = _training_embeddings(embeddings_path, len(studies))
    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 64
    index.add(embeddings)
    _, candidates = index.search(embeddings, min(32, len(studies)))
    neighbours = np.full((len(studies), max_neighbours), -1, dtype=np.int64)
    for row_index, row in enumerate(candidates):
        seen = {studies[row_index].study_id}
        keep = []
        for candidate in row:
            if candidate < 0 or studies[candidate].study_id in seen:
                continue
            seen.add(studies[candidate].study_id)
            keep.append(candidate)
            if len(keep) == max_neighbours:
                break
        neighbours[row_index, : len(keep)] = keep
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, neighbours=neighbours)
    temporary.replace(cache_path)
    return neighbours


class LightweightReportDataset:
    """Image/prefix/report examples with prompt loss masked out."""

    def __init__(
        self,
        studies,
        tokenizer,
        processor,
        decoder_start_token_id: int,
        neighbours: np.ndarray | None = None,
        rag_probability: float = 0.7,
        max_prompt_tokens: int = 160,
        max_target_tokens: int = 128,
        training: bool = False,
        seed: int = 13,
    ):
        self.studies = studies
        self.tokenizer = tokenizer
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
        self.neighbours = neighbours
        self.rag_probability = rag_probability
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        self.training = training
        self.seed = seed

    def __len__(self):
        return len(self.studies)

    def _retrieved(self, index: int) -> list[RetrievedStudy]:
        if self.neighbours is None:
            return []
        if random.Random(self.seed + index).random() >= self.rag_probability:
            return []
        return [
            RetrievedStudy(self.studies[item], 1.0)
            for item in self.neighbours[index, :2]
            if item >= 0
        ]

    def __getitem__(self, index):
        study = self.studies[index]
        image = Image.open(study.image_path).convert("RGB")
        if self.training:
            image = self._augment(image)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        prompt_ids = self.tokenizer(
            lightweight_prompt(study.indication, self._retrieved(index)),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).input_ids
        target_ids = self.tokenizer(
            study.report,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens - 1,
        ).input_ids
        target_ids = [*target_ids, self.tokenizer.eos_token_id]
        full = [*prompt_ids, *target_ids]
        decoder_input_ids = [self.decoder_start_token_id, *full[:-1]]
        labels = [-100] * len(prompt_ids) + target_ids
        if len(decoder_input_ids) != len(labels) or not target_ids:
            raise AssertionError(f"Invalid decoder target for {study.study_id}")
        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": [1] * len(decoder_input_ids),
            "labels": labels,
        }

    def _augment(self, image: Image.Image) -> Image.Image:
        # No horizontal flip: laterality must remain valid.
        from torchvision.transforms import ColorJitter, RandomAffine, RandomResizedCrop
        from torchvision.transforms.functional import InterpolationMode

        size = self.processor.size
        height = int(size.get("height", size.get("shortest_edge", 224)))
        width = int(size.get("width", height))
        image = RandomResizedCrop(
            (height, width),
            scale=(0.92, 1.0),
            ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BICUBIC,
        )(image)
        image = RandomAffine(
            degrees=3,
            translate=(0.02, 0.02),
            interpolation=InterpolationMode.BILINEAR,
        )(image)
        return ColorJitter(brightness=0.05, contrast=0.05)(image)


class LightweightCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch

        length = max(len(row["decoder_input_ids"]) for row in rows)
        batch = len(rows)
        decoder_input_ids = torch.full((batch, length), self.pad_token_id, dtype=torch.long)
        decoder_attention_mask = torch.zeros((batch, length), dtype=torch.long)
        labels = torch.full((batch, length), -100, dtype=torch.long)
        for index, row in enumerate(rows):
            size = len(row["decoder_input_ids"])
            decoder_input_ids[index, :size] = torch.tensor(row["decoder_input_ids"])
            decoder_attention_mask[index, :size] = 1
            labels[index, :size] = torch.tensor(row["labels"])
        return {
            "pixel_values": torch.stack([row["pixel_values"] for row in rows]),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }


def _limited(rows, limit: int, seed: int):
    if limit <= 0 or len(rows) <= limit:
        return rows
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[index] for index in sorted(order[:limit])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--medsiglip-cache", type=Path)
    parser.add_argument("--encoder", default="facebook/deit-tiny-patch16-224")
    parser.add_argument("--decoder", default="distilbert/distilgpt2")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rag-probability", type=float, default=0.7)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-val-examples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        VisionEncoderDecoderModel,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("Lightweight training requires a CUDA Colab runtime")
    set_seed(args.seed)
    studies = load_manifest(args.manifest, redact_splits={"test"})
    train_all = [study for study in studies if study.split == "train"]
    validation = _limited(
        [study for study in studies if study.split == "val"],
        args.max_val_examples,
        args.seed,
    )
    train = _limited(train_all, args.max_train_examples, args.seed)
    if not train or not validation:
        raise ValueError("Manifest must contain non-empty train and val splits")
    if any(study.report for study in studies if study.split == "test"):
        raise AssertionError("Test-reference firewall failed")

    processor = AutoImageProcessor.from_pretrained(args.encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder,
        args.decoder,
        decoder_add_cross_attention=True,
    )
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder.use_cache = False
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())

    neighbours = None
    if args.medsiglip_cache:
        if len(train) != len(train_all):
            raise ValueError("Do not combine --max-train-examples with a full MedSigLIP cache")
        neighbours = neighbour_indices(
            args.medsiglip_cache,
            train,
            args.output_dir / "training_neighbours.npz",
        )
    train_dataset = LightweightReportDataset(
        train,
        tokenizer,
        processor,
        model.config.decoder_start_token_id,
        neighbours=neighbours,
        rag_probability=args.rag_probability,
        training=True,
        seed=args.seed,
    )
    val_dataset = LightweightReportDataset(
        validation,
        tokenizer,
        processor,
        model.config.decoder_start_token_id,
        training=False,
        seed=args.seed,
    )
    eval_steps = max(50, min(250, args.max_steps // 5))
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size * 2),
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.05,
        bf16=use_bf16,
        fp16=not use_bf16,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch_fused",
        torch_compile=args.torch_compile,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=LightweightCollator(tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    checkpoint_dir = args.output_dir / "checkpoints"
    resume = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    trainer.train(resume_from_checkpoint=resume)
    final_dir = args.output_dir / "best_model"
    model.config.decoder.use_cache = True
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "encoder": args.encoder,
        "decoder": args.decoder,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "max_steps": args.max_steps,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "test_examples_consumed": 0,
        "horizontal_flip": False,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
