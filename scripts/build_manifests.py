#!/usr/bin/env python3
"""Build leakage-auditable IU-Xray and MIMIC-CXR JSONL manifests."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from pathlib import Path


def clean_report(value: str) -> str:
    value = re.sub(r"\bXXXX\b", " ", value or "", flags=re.IGNORECASE)
    findings = re.search(
        r"FINDINGS\s*:\s*(.*?)(?=\bIMPRESSION\s*:|$)", value, flags=re.I | re.S
    )
    if findings:
        value = findings.group(1)
    return re.sub(r"\s+", " ", value).strip()


def _write(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_iu(annotation: Path, images_root: Path, output: Path) -> None:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    records = []
    for split in ("train", "val", "test"):
        for row in payload.get(split, []):
            study_id = str(row.get("id") or row.get("uid") or "")
            report = clean_report(str(row.get("report") or ""))
            image_values = row.get("image_path") or row.get("images") or []
            if isinstance(image_values, str):
                image_values = [image_values]
            for value in image_values:
                if isinstance(value, dict):
                    value = value.get("path") or value.get("image_path") or value.get("id")
                if not value:
                    continue
                image_path = Path(str(value))
                if not image_path.is_absolute():
                    image_path = images_root / image_path
                if report and image_path.exists():
                    records.append(
                        {
                            "study_id": study_id,
                            "image_path": str(image_path),
                            "indication": re.sub(
                                r"\s+", " ", str(row.get("indication") or "")
                            ).strip(),
                            "report": report,
                            "split": split,
                            "metadata": {"source": "iu_xray"},
                        }
                    )
    _write(records, output)


def _parse_list_value(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return " ".join(str(item) for item in parsed)
    except (ValueError, SyntaxError):
        pass
    return value


def _image_from_row(row: dict, dataset_root: Path) -> str | None:
    for column in ("PA", "AP", "image", "Lateral"):
        value = str(row.get(column) or "").strip()
        if value and value.lower() not in {"nan", "none", "[]"}:
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    value = str(parsed[0]) if parsed else ""
            except (ValueError, SyntaxError):
                pass
            path = Path(value)
            return str(path if path.is_absolute() else dataset_root / path)
    return None


def _mimic_rows(path: Path, dataset_root: Path) -> list[dict]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_path = _image_from_row(row, dataset_root)
            report = clean_report(_parse_list_value(str(row.get("text") or "")))
            if not image_path or not Path(image_path).exists() or not report:
                continue
            match = re.search(r"(?:^|[/_])(s\d+)(?:[/_.]|$)", image_path)
            study_id = match.group(1) if match else Path(image_path).stem
            records.append(
                {
                    "study_id": study_id,
                    "image_path": image_path,
                    "indication": "",
                    "report": report,
                    "split": "train",
                    "metadata": {
                        "source": "mimic_cxr_augmented",
                        "subject_id": str(row.get("subject_id") or ""),
                    },
                }
            )
    return records


def build_mimic(train_csv: Path, validate_csv: Path, dataset_root: Path, output: Path) -> None:
    train = _mimic_rows(train_csv, dataset_root)
    validation = _mimic_rows(validate_csv, dataset_root)
    subjects = sorted({row["metadata"]["subject_id"] for row in validation})
    random.Random(13).shuffle(subjects)
    val_subjects = set(subjects[: len(subjects) // 2])
    for row in validation:
        row["split"] = "val" if row["metadata"]["subject_id"] in val_subjects else "test"
    _write(train + validation, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="dataset", required=True)
    iu = commands.add_parser("iu")
    iu.add_argument("--annotation", required=True, type=Path)
    iu.add_argument("--images-root", required=True, type=Path)
    iu.add_argument("--output", required=True, type=Path)
    mimic = commands.add_parser("mimic")
    mimic.add_argument("--train-csv", required=True, type=Path)
    mimic.add_argument("--validate-csv", required=True, type=Path)
    mimic.add_argument("--dataset-root", required=True, type=Path)
    mimic.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.dataset == "iu":
        build_iu(args.annotation, args.images_root, args.output)
    else:
        build_mimic(args.train_csv, args.validate_csv, args.dataset_root, args.output)


if __name__ == "__main__":
    main()
