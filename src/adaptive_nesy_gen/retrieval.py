"""Frozen visual encoders and leakage-safe cached nearest-neighbour retrieval."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .schema import RetrievedStudy, Study


class ImageEncoder(Protocol):
    def encode(self, image_paths: list[str]) -> np.ndarray: ...


class PixelHistogramEncoder:
    """Deterministic CPU encoder for tests and pipeline smoke runs only."""

    def __init__(self, bins: int = 64):
        self.bins = bins

    def encode(self, image_paths: list[str]) -> np.ndarray:
        vectors = []
        for path in image_paths:
            image = Image.open(path).convert("L").resize((64, 64))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            histogram, _ = np.histogram(pixels, bins=self.bins, range=(0.0, 1.0))
            # Add coarse spatial evidence so equal histograms need not be identical.
            quadrants = [block.mean() for block in np.array_split(pixels, 4, axis=0)]
            vector = np.concatenate([histogram.astype(np.float32), quadrants])
            vectors.append(vector)
        matrix = np.stack(vectors)
        return _l2_normalize(matrix)


class MedSigLIPEncoder:
    """Lazy Hugging Face adapter for the frozen google/medsiglip-448 image tower."""

    def __init__(self, model_id: str = "google/medsiglip-448", batch_size: int = 8):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise RuntimeError(
                "Install adaptive-nesy-gen[medgemma] or [chexagent] for MedSigLIP"
            ) from exc
        self.torch = torch
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, image_paths: list[str]) -> np.ndarray:  # pragma: no cover - GPU path
        batches: list[np.ndarray] = []
        for offset in range(0, len(image_paths), self.batch_size):
            images = [
                Image.open(path).convert("RGB")
                for path in image_paths[offset : offset + self.batch_size]
            ]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                if hasattr(self.model, "get_image_features"):
                    output = self.model.get_image_features(**inputs)
                else:
                    output = self.model.vision_model(**inputs).pooler_output
            batches.append(output.float().cpu().numpy())
        return _l2_normalize(np.concatenate(batches))


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    return matrix / denominator


def load_manifest(
    path: str | Path,
    redact_splits: set[str] | frozenset[str] = frozenset(),
) -> list[Study]:
    """Load studies, optionally blanking reports at the ingestion boundary.

    Generation runners use ``redact_splits={"test"}`` so test references cannot
    enter retrieval, prompting, verification, revision, or backend objects.
    """
    path = Path(path)
    redacted = set(redact_splits)
    studies: list[Study] = []
    for row in _manifest_rows(path):
        split = row.get("split", "train")
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw": metadata}
        studies.append(
            Study(
                study_id=str(row["study_id"]),
                patient_id=str(row.get("patient_id") or metadata.get("subject_id", "")),
                image_path=str(image_path),
                report="" if split in redacted else row["report"],
                indication=row.get("indication", ""),
                split=split,
                view=row.get("view", ""),
                metadata=metadata,
            )
        )
    return studies


def _manifest_rows(path: Path):
    """Stream rows so redacted references are never retained in a raw row list."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)


def manifest_example_id(study: Study) -> str:
    """Stable row key that preserves multiple images belonging to one study."""
    image_digest = hashlib.sha256(study.image_path.encode("utf-8")).hexdigest()[:12]
    return f"{study.study_id}::{image_digest}"


class VisualIndex:
    def __init__(self, studies: list[Study], embeddings: np.ndarray, encoder: ImageEncoder):
        if len(studies) != len(embeddings):
            raise ValueError("studies and embeddings have different lengths")
        if any(study.split != "train" for study in studies):
            raise ValueError("VisualIndex may contain training-split studies only")
        self.studies = studies
        self.embeddings = _l2_normalize(embeddings.astype(np.float32))
        self.encoder = encoder

    @classmethod
    def build(cls, studies: list[Study], encoder: ImageEncoder) -> tuple[VisualIndex, float]:
        train = [study for study in studies if study.split == "train"]
        started = time.perf_counter()
        embeddings = encoder.encode([study.image_path for study in train])
        elapsed = (time.perf_counter() - started) * 1000.0
        return cls(train, embeddings, encoder), elapsed

    def save(self, path: str | Path) -> None:
        import json

        metadata = [study.__dict__ for study in self.studies]
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            metadata=np.asarray(json.dumps(metadata)),
            fingerprint=np.asarray(fingerprint),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        encoder: ImageEncoder,
        studies: list[Study] | None = None,
    ) -> VisualIndex:
        archive = np.load(path, allow_pickle=False)
        embedding_key = next(
            (key for key in ("embeddings", "image_embeddings", "features") if key in archive),
            None,
        )
        if embedding_key is None:
            raise ValueError(f"No embedding array found in {path}; keys={archive.files}")
        if "metadata" in archive:
            metadata = json.loads(str(archive["metadata"]))
            indexed = [Study(**row) for row in metadata]
        elif studies is not None:
            indexed = [study for study in studies if study.split == "train"]
        else:
            raise ValueError("A manifest is required for caches without embedded metadata")
        if len(indexed) != len(archive[embedding_key]):
            raise ValueError(
                "Manifest train rows do not align with cache embeddings: "
                f"{len(indexed)} != {len(archive[embedding_key])}"
            )
        return cls(indexed, archive[embedding_key], encoder)

    def query(
        self,
        image_path: str,
        k: int,
        exclude_study_id: str | None = None,
        exclude_patient_id: str | None = None,
    ) -> list[RetrievedStudy]:
        if k <= 0:
            return []
        query = _l2_normalize(self.encoder.encode([image_path]).astype(np.float32))[0]
        scores = self.embeddings @ query
        ranked = np.argsort(-scores, kind="stable")
        results: list[RetrievedStudy] = []
        seen_studies: set[str] = set()
        for index in ranked:
            study = self.studies[int(index)]
            # Study exclusion removes alternate views. Patient exclusion is an optional,
            # stricter integrity analysis rather than the default methodology contract.
            if exclude_study_id and study.study_id == exclude_study_id:
                continue
            if exclude_patient_id and study.patient_id == exclude_patient_id:
                continue
            if study.study_id in seen_studies:
                continue
            results.append(RetrievedStudy(study=study, similarity=float(scores[index])))
            seen_studies.add(study.study_id)
            if len(results) == k:
                break
        return results
