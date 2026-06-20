#!/usr/bin/env python3
"""Build a complete train-only MedSigLIP cache with atomic replacement."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from adaptive_nesy_gen.retrieval import MedSigLIPEncoder, VisualIndex, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", default="google/medsiglip-448")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--progress-file",
        type=Path,
        help="Optional JSON status file for a parent notebook progress bar",
    )
    args = parser.parse_args()

    def report_progress(stage: str, completed: int, total: int) -> None:
        if args.progress_file is None:
            return
        args.progress_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_status = args.progress_file.with_suffix(".tmp")
        temporary_status.write_text(
            json.dumps({"stage": stage, "completed": completed, "total": total}),
            encoding="utf-8",
        )
        temporary_status.replace(args.progress_file)

    report_progress("Reading training manifest", 0, 0)
    studies = load_manifest(args.manifest, redact_splits={"val", "test"})
    train_count = sum(study.split == "train" for study in studies)
    if not train_count:
        raise ValueError("Manifest has no training examples")
    report_progress("Loading MedSigLIP model", 0, train_count)
    encoder = MedSigLIPEncoder(
        args.model_id,
        batch_size=args.batch_size,
        progress_callback=lambda completed, total: report_progress(
            "Encoding training images", completed, total
        ),
        show_progress=args.progress_file is None,
    )
    report_progress("Encoding training images", 0, train_count)
    started = time.perf_counter()
    index, encoding_ms = VisualIndex.build(studies, encoder)
    total_ms = (time.perf_counter() - started) * 1000.0

    report_progress("Saving cache atomically", train_count, train_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.stem}.rebuild.npz")
    index.save(temporary, indexing_ms=total_ms)
    temporary.replace(args.output)
    summary = {
        "manifest": str(args.manifest),
        "output": str(args.output),
        "model_id": args.model_id,
        "training_examples": train_count,
        "embedding_shape": list(index.embeddings.shape),
        "encoding_ms": encoding_ms,
        "indexing_ms": total_ms,
    }
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report_progress("Complete", train_count, train_count)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
