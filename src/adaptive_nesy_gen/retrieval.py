"""Frozen visual encoders and leakage-safe cached nearest-neighbour retrieval."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

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

    def __init__(
        self,
        model_id: str = "google/medsiglip-448",
        batch_size: int = 8,
        progress_callback: Callable[[int, int], None] | None = None,
        show_progress: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise RuntimeError(
                "Install adaptive-nesy-gen[medgemma] or [chexagent] for MedSigLIP"
            ) from exc
        self.torch = torch
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.show_progress = show_progress
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except (OSError, ValueError) as exc:
            raise RuntimeError(_medsiglip_access_message(model_id)) from exc
        if torch.cuda.is_available():
            self.dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            self.dtype = torch.float32
        try:
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype)
        except (OSError, ValueError) as exc:
            raise RuntimeError(_medsiglip_access_message(model_id)) from exc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, image_paths: list[str]) -> np.ndarray:  # pragma: no cover - GPU path
        batches: list[np.ndarray] = []
        progress = tqdm(
            total=len(image_paths),
            desc="Encoding MedSigLIP training images",
            unit="image",
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        offset = 0
        batch_size = self.batch_size
        while offset < len(image_paths):
            batch_paths = image_paths[offset : offset + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = None
            try:
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
                with self.torch.inference_mode():
                    if hasattr(self.model, "get_image_features"):
                        output = self.model.get_image_features(**inputs)
                    else:
                        output = self.model.vision_model(**inputs).pooler_output
            except self.torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    progress.close()
                    raise
                batch_size = max(1, batch_size // 2)
                del inputs
                self.torch.cuda.empty_cache()
                progress.write(f"CUDA OOM; retrying with batch size {batch_size}")
                continue
            batches.append(output.float().cpu().numpy())
            del inputs, output
            offset += len(batch_paths)
            progress.update(len(batch_paths))
            if self.progress_callback is not None:
                self.progress_callback(offset, len(image_paths))
        progress.close()
        return _l2_normalize(np.concatenate(batches))


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    return matrix / denominator


def _medsiglip_access_message(model_id: str) -> str:
    return (
        f"Could not load gated model {model_id}. Accept its Health AI terms at "
        f"https://huggingface.co/{model_id}, authenticate with huggingface_hub.login(), "
        "then rerun the cache build."
    )


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
        self._query_cache: dict[
            tuple[str, int, str | None, str | None], tuple[RetrievedStudy, ...]
        ] = {}

    @classmethod
    def build(cls, studies: list[Study], encoder: ImageEncoder) -> tuple[VisualIndex, float]:
        train = [study for study in studies if study.split == "train"]
        started = time.perf_counter()
        embeddings = encoder.encode([study.image_path for study in train])
        elapsed = (time.perf_counter() - started) * 1000.0
        return cls(train, embeddings, encoder), elapsed

    def save(self, path: str | Path, indexing_ms: float | None = None) -> None:
        import json

        metadata = [study.__dict__ for study in self.studies]
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        payload = {
            "embeddings": self.embeddings,
            "metadata": np.asarray(json.dumps(metadata)),
            "fingerprint": np.asarray(fingerprint),
        }
        if indexing_ms is not None:
            payload["indexing_ms"] = np.asarray(indexing_ms)
        np.savez_compressed(path, **payload)

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
        embeddings = _orient_embedding_matrix(archive[embedding_key], len(indexed))
        if len(indexed) != len(embeddings):
            raise ValueError(
                "Manifest train rows do not align with cache embeddings: "
                f"{len(indexed)} != {len(embeddings)}"
            )
        return cls(indexed, embeddings, encoder)

    def query(
        self,
        image_path: str,
        k: int,
        exclude_study_id: str | None = None,
        exclude_patient_id: str | None = None,
    ) -> list[RetrievedStudy]:
        if k <= 0:
            return []
        cache_key = (image_path, k, exclude_study_id, exclude_patient_id)
        if cache_key in self._query_cache:
            return list(self._query_cache[cache_key])
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
        self._query_cache[cache_key] = tuple(results)
        return results


def _orient_embedding_matrix(matrix: np.ndarray, expected_rows: int) -> np.ndarray:
    """Accept sample-major caches and legacy transposed feature matrices."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] == expected_rows:
        return matrix
    if matrix.shape[1] == expected_rows:
        return matrix.T
    return matrix
