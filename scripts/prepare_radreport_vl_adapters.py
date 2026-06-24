#!/usr/bin/env python3
"""Install manifest and draft-export adapters into a RadReport-VL checkout.

The adapters intentionally do not edit RadReport-VL's vision encoder, cross-attention
bridge, report decoder, or classifier. They only add:

1. A missing ``src.data.mimic_cxr_dataset`` module that can read this project's
   manifest JSONL/CSV format.
2. A manifest draft exporter that loads a trained RadReport-VL checkpoint and writes
   reference-free JSONL drafts for Adaptive NeSy-Gen replay.
3. A tiny training-script compatibility patch that resizes decoder token embeddings
   after RadReport-VL extends its tokenizer, then saves that tokenizer beside the
   checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

DATASET_ADAPTER = r'''
"""Manifest-backed datasets for RadReport-VL.

This file is installed by Adaptive NeSy-Gen's adapter script because the upstream
RadReport-VL repository currently imports ``src.data.mimic_cxr_dataset`` but does
not ship it. It preserves the RadReport-VL model architecture and only supplies
data in the batch contract expected by ``src.training.trainer.RadReportTrainer``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SPLIT_ALIASES = {
    "train": {"train", "training"},
    "validate": {"validate", "validation", "val", "dev"},
    "val": {"validate", "validation", "val", "dev"},
    "test": {"test", "testing"},
}

CHEXPERT_COLUMNS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    report: str
    split: str
    chexpert_labels: list[float] | None = None


def _read_rows(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)


def _candidate_manifest(root_dir: str | Path, split: str) -> Path | None:
    root = Path(root_dir)
    if root.is_file():
        return root
    names = (
        f"{split}.jsonl",
        f"{split}.csv",
        "manifest.jsonl",
        "manifest.csv",
        "metadata.csv",
    )
    return next((root / name for name in names if (root / name).exists()), None)


def _labels(row: dict) -> list[float] | None:
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    source = metadata.get("chexpert_labels") or metadata.get("labels") or row
    values = []
    for name in CHEXPERT_COLUMNS:
        value = source.get(name) if isinstance(source, dict) else None
        if value in {None, ""}:
            return None
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
    return values


def _manifest_rows(root_dir: str | Path, split: str, max_samples: int | None = None):
    manifest = _candidate_manifest(root_dir, split)
    if manifest is None:
        return None
    allowed = SPLIT_ALIASES.get(split, {split})
    rows = []
    for row in _read_rows(manifest):
        row_split = str(row.get("split", split)).lower()
        if row_split not in allowed:
            continue
        image_path = Path(row.get("image_path") or row.get("path") or row.get("jpg_path") or "")
        if not image_path:
            continue
        if not image_path.is_absolute():
            image_path = (manifest.parent / image_path).resolve()
        report = str(row.get("report") or row.get("findings") or row.get("text") or "").strip()
        if not report:
            continue
        rows.append(
            ManifestRow(
                image_path=str(image_path),
                report=report,
                split=row_split,
                chexpert_labels=_labels(row),
            )
        )
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def _safe_name(path: str, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem)[:96] or f"image_{index}"
    suffix = Path(path).suffix.lower() or ".png"
    return f"{index:08d}_{stem}{suffix}"


def _maybe_cache_images(rows: list[ManifestRow], split: str) -> list[ManifestRow]:
    cache_root = os.environ.get("RADREPORT_VL_IMAGE_CACHE_DIR")
    if not cache_root:
        return rows
    target_root = Path(cache_root) / split
    target_root.mkdir(parents=True, exist_ok=True)
    cached = []
    for index, row in enumerate(rows):
        target = target_root / _safe_name(row.image_path, index)
        if not target.exists() or target.stat().st_size == 0:
            shutil.copy2(row.image_path, target)
        cached.append(
            ManifestRow(
                image_path=str(target),
                report=row.report,
                split=row.split,
                chexpert_labels=row.chexpert_labels,
            )
        )
    print(
        json.dumps(
            {
                "radreport_vl_image_cache": str(target_root),
                "split": split,
                "images": len(cached),
            }
        )
    )
    return cached


class ManifestCXRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        tokenizer=None,
        image_size=224,
        max_samples=None,
    ):
        rows = _manifest_rows(root_dir, split, max_samples=max_samples)
        if rows is None:
            print(f"[WARN] No manifest found at {root_dir}; using synthetic {split} data.")
            rows = [
                ManifestRow(
                    image_path="",
                    report=(
                        "[FINDINGS] The lungs are clear. No pleural effusion or "
                        "pneumothorax. [IMPRESSION] No acute cardiopulmonary abnormality."
                    ),
                    split=split,
                    chexpert_labels=[0.0] * 14,
                )
                for _ in range(max_samples or 32)
            ]
        self.rows = _maybe_cache_images(rows, split)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row.image_path:
            image = Image.open(row.image_path).convert("RGB")
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.randn(3, self.image_size, self.image_size).clamp(-2, 2)
        encoded = self.tokenizer(
            row.report,
            max_length=300,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = row.chexpert_labels if row.chexpert_labels is not None else [0.0] * 14
        return {
            "image": image_tensor.float(),
            "input_ids": encoded["input_ids"].squeeze(0).long(),
            "attention_mask": encoded["attention_mask"].squeeze(0).long(),
            "chexpert_labels": torch.tensor(labels, dtype=torch.float32),
            "report_text": row.report,
        }


class IUXRayDataset(ManifestCXRDataset):
    pass


def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    output = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        output[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return output


def build_mimic_cxr_dataloader(
    root_dir,
    split,
    tokenizer,
    batch_size,
    num_workers=2,
    image_size=224,
    use_chexpert=True,
    max_samples=None,
):
    del use_chexpert
    dataset = ManifestCXRDataset(
        root_dir=root_dir,
        split=split,
        tokenizer=tokenizer,
        image_size=image_size,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        collate_fn=collate_fn_skip_none,
        pin_memory=torch.cuda.is_available(),
    )
'''


DRAFT_EXPORTER = r'''
#!/usr/bin/env python3
"""Generate reference-free Adaptive NeSy-Gen drafts from a RadReport-VL checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import build_model, load_config, set_seed
from src.data.report_tokenizer import MedicalReportTokenizer
from src.inference.report_generator import extract_sections_from_generated, preprocess_image


def _rows(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)


def _example_id(study_id: str, image_path: str) -> str:
    digest = hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:12]
    return f"{study_id}::{digest}"


def _manifest(path: Path, split: str):
    allowed = {split}
    if split == "val":
        allowed |= {"validate", "validation", "dev"}
    if split == "validate":
        allowed |= {"val", "validation", "dev"}
    for row in _rows(path):
        row_split = str(row.get("split", "train")).lower()
        if row_split not in allowed:
            continue
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        study_id = str(row["study_id"])
        yield {
            "study_id": study_id,
            "patient_id": str(row.get("patient_id", "")),
            "image_path": str(image_path),
            "indication": str(row.get("indication", "")),
            "split": row_split,
            "example_id": _example_id(study_id, str(image_path)),
        }


def _tokenizer_for_config(cfg: dict):
    backbone = cfg["model"]["decoder"].get("backbone", "biogpt")
    model_id = "microsoft/biogpt" if backbone == "biogpt" else "gpt2"
    return MedicalReportTokenizer.from_pretrained(model_id)


def _load_tokenizer(cfg: dict, checkpoint: Path):
    tokenizer_dir = checkpoint.parent / "tokenizer"
    if tokenizer_dir.exists():
        return MedicalReportTokenizer.load_pretrained(tokenizer_dir)
    return _tokenizer_for_config(cfg)


def _report_from_sections(text: str) -> str:
    sections = extract_sections_from_generated(text)
    report = " ".join(
        part.strip()
        for part in (sections.get("findings", ""), sections.get("impression", ""))
        if part and part.strip()
    )
    return report or text.strip()


@torch.inference_mode()
def generate_one(model, tokenizer, image_path: str, args, device: str) -> tuple[str, str]:
    image_tensor = preprocess_image(image_path, args.image_size, device)
    encoded = model.vision_encoder(image_tensor)
    bridge = model.bridge(encoded["patch_tokens"])
    output = model.decoder.generate(
        image_features=bridge["query_output"],
        decoding=args.decoding,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        return_scores=False,
    )
    token_ids = output["token_ids"][0]
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return full_text, _report_from_sections(full_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--decoding",
        choices=["beam_search", "greedy", "nucleus"],
        default="beam_search",
    )
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    # Reuse RadReport-VL's config/model builder. This preserves its architecture.
    empty_args = argparse.Namespace(
        config=str(args.config),
        resume_from=None,
        phase2_only=False,
        pretrained_checkpoint=None,
        encoder_type=None,
        bridge_type=None,
        decoder_type=None,
        batch_size=None,
        learning_rate=None,
        num_epochs=None,
        lambda_cls=None,
        device=args.device,
        seed=None,
        data_root=None,
        max_samples=None,
    )
    cfg = load_config(str(args.config), empty_args)
    set_seed(cfg.get("hardware", {}).get("seed", 42))
    tokenizer = _load_tokenizer(cfg, args.checkpoint)
    model = build_model(cfg)
    if hasattr(model, "decoder") and hasattr(model.decoder, "backbone"):
        model.decoder.backbone.resize_token_embeddings(tokenizer.vocab_size)
        model.decoder.config.vocab_size = tokenizer.vocab_size
        model.decoder.config.pad_token_id = tokenizer.pad_token_id
        model.decoder.config.bos_token_id = tokenizer.bos_token_id
        model.decoder.config.eos_token_id = tokenizer.eos_token_id
    state = torch.load(args.checkpoint, map_location="cpu")
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(json.dumps({"missing_keys": len(missing), "unexpected_keys": len(unexpected)}))
    model.to(args.device).eval()

    rows = list(_manifest(args.manifest, args.split))
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc="RadReport-VL drafts", unit="study", dynamic_ncols=True):
            full_text, report = generate_one(model, tokenizer, row["image_path"], args, args.device)
            record = {
                "study_id": row["study_id"],
                "example_id": row["example_id"],
                "patient_id": row["patient_id"],
                "image_path": row["image_path"],
                "split": row["split"],
                "raw_report": report,
                "final_report": report,
                "radreport_vl_full_text": full_text,
                "backend": "radreport-vl",
                "test_reference_consumed_during_inference": False,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
    print(json.dumps({"drafts": str(args.output), "count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path, help="Path to a radreport-vl checkout")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / "scripts" / "train.py").exists():
        raise FileNotFoundError(f"{repo} does not look like a radreport-vl checkout")

    dataset_path = repo / "src" / "data" / "mimic_cxr_dataset.py"
    exporter_path = repo / "scripts" / "generate_manifest_drafts.py"
    train_path = repo / "scripts" / "train.py"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    exporter_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(dedent(DATASET_ADAPTER).lstrip(), encoding="utf-8")
    exporter_path.write_text(dedent(DRAFT_EXPORTER).lstrip(), encoding="utf-8")
    exporter_path.chmod(0o755)
    _patch_train_script(train_path)
    print(
        {
            "repo": str(repo),
            "dataset_adapter": str(dataset_path),
            "draft_exporter": str(exporter_path),
            "training_compatibility_patch": str(train_path),
            "architecture_policy": (
                "RadReport-VL vision encoder, bridge, decoder, and classifier are untouched."
            ),
        }
    )


def _patch_train_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "    # Build data loaders\n"
    patch = '''
    # Adapter compatibility: RadReport-VL extends the tokenizer with section and
    # medical tokens. Resize only the LM embedding/output tables so token IDs are
    # in range; this preserves the vision encoder, bridge, decoder stack, and
    # classifier architecture.
    if hasattr(model, "decoder") and hasattr(model.decoder, "backbone"):
        model.decoder.backbone.resize_token_embeddings(tokenizer.vocab_size)
        model.decoder.config.vocab_size = tokenizer.vocab_size
        model.decoder.config.pad_token_id = tokenizer.pad_token_id
        model.decoder.config.bos_token_id = tokenizer.bos_token_id
        model.decoder.config.eos_token_id = tokenizer.eos_token_id
        logger.info(f"Decoder token embeddings resized to {tokenizer.vocab_size}")

    tokenizer_dir = Path(cfg["checkpointing"]["output_dir"]) / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    logger.info(f"Tokenizer saved for inference: {tokenizer_dir}")

'''
    if "Decoder token embeddings resized to" not in text:
        text = text.replace(marker, patch + marker)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
