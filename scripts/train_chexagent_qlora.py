#!/usr/bin/env python3
"""Fast CheXagent QLoRA with a frozen BF16 vision encoder and optional visual RAG.

CheXagent's published remote code pins Transformers 4.40.0. Keep this training
environment separate from the newer Transformers environment used by MedGemma.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from adaptive_nesy_gen.backends import chexagent_prompt
from adaptive_nesy_gen.retrieval import load_manifest
from adaptive_nesy_gen.schema import RetrievedStudy


def _embedding_array(path: Path) -> np.ndarray:
    archive = np.load(path, allow_pickle=False)
    key = next(
        (name for name in ("embeddings", "image_embeddings", "features") if name in archive),
        None,
    )
    if key is None:
        raise ValueError(f"No embeddings found in {path}; keys={archive.files}")
    matrix = archive[key].astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    return matrix


def neighbour_indices(
    embeddings_path: Path,
    studies,
    cache_path: Path,
    max_neighbours: int = 5,
) -> np.ndarray:
    """Build/reuse an HNSW neighbour table; alternate views share a study exclusion."""
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)["neighbours"]
        if len(cached) == len(studies):
            return cached
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Install the chexagent extra, which includes faiss-cpu") from exc
    embeddings = _embedding_array(embeddings_path)
    if len(embeddings) != len(studies):
        raise ValueError(
            f"MedSigLIP cache/manifest misalignment: {len(embeddings)} != {len(studies)}"
        )
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
    np.savez_compressed(cache_path, neighbours=neighbours)
    return neighbours


class ReportDataset:
    def __init__(
        self,
        studies,
        tokenizer,
        max_length: int,
        neighbours: np.ndarray | None = None,
        rag_probability: float = 0.5,
        seed: int = 13,
    ):
        self.studies = studies
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.neighbours = neighbours
        self.rag_probability = rag_probability
        self.seed = seed

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        study = self.studies[index]
        retrieved = []
        if (
            self.neighbours is not None
            and random.Random(self.seed + index).random() < self.rag_probability
        ):
            retrieved = [
                RetrievedStudy(self.studies[item], 1.0)
                for item in self.neighbours[index, :3]
                if item >= 0
            ]
        query = self.tokenizer.from_list_format(
            [{"image": study.image_path}, {"text": chexagent_prompt(study.indication, retrieved)}]
        )
        prompt = [
            {"from": "system", "value": "You are a careful radiology report assistant."},
            {"from": "human", "value": query},
        ]
        full = [*prompt, {"from": "assistant", "value": study.report}]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        )[0]
        input_ids = self.tokenizer.apply_chat_template(
            full, add_generation_prompt=False, return_tensors="pt"
        )[0][: self.max_length]
        labels = input_ids.clone()
        labels[: min(len(prompt_ids), len(labels))] = -100
        return {"input_ids": input_ids, "labels": labels}


class CausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch

        length = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
        for index, row in enumerate(rows):
            size = len(row["input_ids"])
            input_ids[index, :size] = row["input_ids"]
            labels[index, :size] = row["labels"]
            attention_mask[index, :size] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ConditionalImageTransform:
    """Selects augmentation only while the frozen visual module is in training mode."""

    def __init__(self, visual, train_transform, eval_transform):
        self.visual = visual
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def __call__(self, image):
        transform = self.train_transform if self.visual.training else self.eval_transform
        return transform(image)


def configure_visual(model):
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    visual = model.model.visual
    if any("4bit" in type(module).__name__.lower() for module in visual.modules()):
        raise RuntimeError("Vision encoder was quantized; check model.visual skip configuration")
    for parameter in visual.parameters():
        parameter.requires_grad_(False)
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(torch.bfloat16)
    mean, std = visual.mean, visual.std
    deterministic = transforms.Compose(
        [
            transforms.Resize((512, 512), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    augmented = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (512, 512), scale=(0.95, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomRotation(3, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.25),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    visual.image_transform = ConditionalImageTransform(visual, augmented, deterministic)
    return visual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--medsiglip-cache", type=Path)
    parser.add_argument("--model-id", default="StanfordAIMI/CheXagent-2-3b")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--rag-probability", type=float, default=0.5)
    args = parser.parse_args()

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    if transformers.__version__ != "4.40.0":
        raise RuntimeError("CheXagent remote code requires transformers==4.40.0")
    studies = load_manifest(args.manifest)
    if any(study.split not in {"train", "val", "test"} for study in studies):
        raise ValueError("Manifest contains an unknown split")
    train = [study for study in studies if study.split == "train"]
    validation = [study for study in studies if study.split == "val"]
    if not train or not validation:
        raise ValueError("Both train and val rows are required; test rows are never accepted")
    if any(study.split == "test" for study in train + validation):
        raise AssertionError("test leakage into QLoRA")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["model.visual"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
    )
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    except (AttributeError, TypeError):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    visual = configure_visual(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # Regex is deliberately decoder-scoped: CheXagent's visual SigLIP also
            # contains q/k/v projections and must remain fully frozen.
            target_modules=(
                r"model\.layers\.\d+\.(?:self_attn\."
                r"(?:q_proj|k_proj|v_proj|dense)|mlp\.(?:fc1|fc2))"
            ),
        ),
    )
    assert not any(parameter.requires_grad for parameter in visual.parameters())

    neighbours = None
    if args.medsiglip_cache:
        if len(train) != len(_embedding_array(args.medsiglip_cache)):
            raise ValueError("MedSigLIP index must contain exactly the training manifest rows")
        neighbours = neighbour_indices(
            args.medsiglip_cache,
            train,
            args.output_dir / "training_neighbours.npz",
        )
    train_dataset = ReportDataset(
        train, tokenizer, args.max_length, neighbours, args.rag_probability
    )
    val_dataset = ReportDataset(validation, tokenizer, args.max_length, None, 0.0)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=max(25, min(100, args.max_steps // 5)),
        save_steps=max(25, min(100, args.max_steps // 5)),
        logging_steps=5,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        seed=13,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    summary = {
        "train_examples": len(train),
        "val_examples": len(validation),
        "test_examples_consumed": 0,
        "max_steps": args.max_steps,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
