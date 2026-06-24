#!/usr/bin/env python3
"""Train the fast Adaptive NeSy-Gen drafting model.

The default model combines a frozen DeiT-tiny vision transformer with a lightweight
Flan-T5-small text generator. The visual transformer always stays frozen; by default
training updates only the T5 decoder/language head plus a small visual projection.
This is intended for a single Colab GPU and resumes automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

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
    """Image/prompt/report examples for frozen-ViT + T5-style generation."""

    def __init__(
        self,
        studies,
        tokenizer,
        processor,
        neighbours: np.ndarray | None = None,
        rag_probability: float = 0.7,
        max_prompt_tokens: int = 160,
        max_target_tokens: int = 128,
        training: bool = False,
        augment_images: bool = True,
        seed: int = 13,
    ):
        self.studies = studies
        self.tokenizer = tokenizer
        self.processor = processor
        self.neighbours = neighbours
        self.rag_probability = rag_probability
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        self.training = training
        self.augment_images = augment_images
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
        if self.training and self.augment_images:
            image = self._augment(image)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
        prompt_ids = self.tokenizer(
            lightweight_prompt(study.indication, self._retrieved(index)),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).input_ids
        target = self.tokenizer(
            study.report,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_target_tokens,
        )
        labels = list(target.input_ids)
        if not prompt_ids or not labels:
            raise AssertionError(f"Invalid decoder target for {study.study_id}")
        return {
            "pixel_values": pixel_values,
            "input_ids": prompt_ids,
            "attention_mask": [1] * len(prompt_ids),
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

        input_length = max(len(row["input_ids"]) for row in rows)
        label_length = max(len(row["labels"]) for row in rows)
        batch = len(rows)
        input_ids = torch.full((batch, input_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, input_length), dtype=torch.long)
        labels = torch.full((batch, label_length), -100, dtype=torch.long)
        for index, row in enumerate(rows):
            input_size = len(row["input_ids"])
            label_size = len(row["labels"])
            input_ids[index, :input_size] = torch.tensor(row["input_ids"])
            attention_mask[index, :input_size] = 1
            labels[index, :label_size] = torch.tensor(row["labels"])
        return {
            "pixel_values": torch.stack([row["pixel_values"] for row in rows]),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _limited(rows, limit: int, seed: int):
    if limit <= 0 or len(rows) <= limit:
        return rows
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[index] for index in sorted(order[:limit])]


def _cache_path(cache_dir: Path, study) -> Path:
    suffix = Path(study.image_path).suffix.lower() or ".png"
    digest = hashlib.sha1(f"{study.study_id}\0{study.image_path}".encode()).hexdigest()[:16]
    safe_study_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", study.study_id)[:80] or "study"
    return cache_dir / study.split / f"{safe_study_id}_{digest}{suffix}"


def materialize_images(studies, cache_dir: Path | None):
    """Copy Drive-backed images to local SSD once, returning studies with local paths."""
    if cache_dir is None:
        return studies
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = []
    copied = 0
    for study in tqdm(studies, desc=f"Image cache → {cache_dir}", unit="image"):
        source = Path(study.image_path)
        target = _cache_path(cache_dir, study)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size == 0:
            shutil.copy2(source, target)
            copied += 1
        cached.append(replace(study, image_path=str(target)))
    print(
        json.dumps(
            {
                "image_cache_dir": str(cache_dir),
                "images_total": len(cached),
                "images_copied": copied,
                "images_reused": len(cached) - copied,
            }
        )
    )
    return cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--medsiglip-cache", type=Path)
    parser.add_argument("--encoder", default="facebook/deit-tiny-patch16-224")
    parser.add_argument("--decoder", default="google/flan-t5-small")
    parser.add_argument("--visual-tokens", type=int, default=32)
    parser.add_argument(
        "--train-text-encoder",
        action="store_true",
        help="Also update the T5 encoder. Default keeps it frozen for decoder-scoped training.",
    )
    parser.add_argument(
        "--train-embeddings",
        action="store_true",
        help="Also update shared T5 token embeddings. Default keeps them frozen.",
    )
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rag-probability", type=float, default=0.7)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-val-examples", type=int, default=512)
    parser.add_argument("--image-cache-dir", type=Path)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable CPU image augmentation for the fastest one-day deadline run.",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from adaptive_nesy_gen.lightweight_vit_t5 import FrozenViTFlanT5ReportModel

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

    processor = AutoImageProcessor.from_pretrained(args.encoder, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vision_encoder = AutoModel.from_pretrained(args.encoder, add_pooling_layer=False)
    text_model = AutoModelForSeq2SeqLM.from_pretrained(args.decoder)
    text_model.config.use_cache = False
    model = FrozenViTFlanT5ReportModel.create(
        vision_encoder,
        text_model,
        visual_tokens=args.visual_tokens,
        train_text_encoder=args.train_text_encoder,
        train_embeddings=args.train_embeddings,
    )
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
    if args.image_cache_dir:
        train = materialize_images(train, args.image_cache_dir)
        validation = materialize_images(validation, args.image_cache_dir)
    train_dataset = LightweightReportDataset(
        train,
        tokenizer,
        processor,
        neighbours=neighbours,
        rag_probability=args.rag_probability,
        training=True,
        augment_images=not args.no_augmentation,
        seed=args.seed,
    )
    val_dataset = LightweightReportDataset(
        validation,
        tokenizer,
        processor,
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
        # Keep labels in the model call: the custom ViT→T5 wrapper shifts them into
        # decoder_input_ids explicitly. HF label smoothing pops labels before forward.
        label_smoothing_factor=0.0,
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
        disable_tqdm=False,
        dataloader_num_workers=args.dataloader_num_workers,
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
    model.text_model.config.use_cache = True
    model.save_for_generation(
        final_dir,
        encoder_id=args.encoder,
        decoder_id=args.decoder,
        tokenizer=tokenizer,
        processor=processor,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "architecture": "vit-flan-t5",
        "encoder": args.encoder,
        "decoder": args.decoder,
        "visual_tokens": args.visual_tokens,
        "train_text_encoder": args.train_text_encoder,
        "train_embeddings": args.train_embeddings,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "max_steps": args.max_steps,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "test_examples_consumed": 0,
        "horizontal_flip": False,
        "augmentation": not args.no_augmentation,
        "image_cache_dir": str(args.image_cache_dir) if args.image_cache_dir else None,
        "dataloader_num_workers": args.dataloader_num_workers,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
