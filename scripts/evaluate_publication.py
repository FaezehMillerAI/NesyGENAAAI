#!/usr/bin/env python3
"""Run publication-grade text/clinical metrics and integrity audits after inference."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from adaptive_nesy_gen.evaluation import (
    aggregate_results,
    leakage_audit,
    linker_structural_audit,
    paired_bootstrap_delta,
    prediction_diversity,
    resource_summary,
    retrieval_integrity_audit,
)
from adaptive_nesy_gen.knowledge import KnowledgeGraph
from adaptive_nesy_gen.retrieval import load_manifest, manifest_example_id
from adaptive_nesy_gen.text import DeterministicLinker


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_official_scorers() -> dict[str, Any]:
    """Load heavyweight official evaluators once for reuse across every ablation."""
    try:
        from f1chexbert import F1CheXbert
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from radgraph import F1RadGraph
    except ImportError as exc:  # pragma: no cover - optional publication dependency
        raise RuntimeError("Install adaptive-nesy-gen[eval] for official evaluators") from exc

    progress = tqdm(
        total=3,
        desc="Loading official evaluators",
        unit="model",
        dynamic_ncols=True,
    )
    scorers = {
        "bleu": Bleu(4),
        "rouge_l": Rouge(),
        "meteor": Meteor(),
        "cider": Cider(),
    }
    progress.update()
    scorers["chexbert"] = F1CheXbert()
    progress.update()
    scorers["radgraph"] = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    progress.update()
    progress.close()
    return scorers


def official_coco_metrics(
    hypotheses: list[str], references: list[str], scorers: dict[str, Any] | None = None
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Microsoft COCO implementations of BLEU-1..4, ROUGE-L, METEOR, and CIDEr."""
    ground_truth = {index: [text] for index, text in enumerate(references)}
    generated = {index: [text] for index, text in enumerate(hypotheses)}
    per_study: list[dict[str, float]] = [{} for _ in hypotheses]
    summary: dict[str, float] = {}
    scorers = scorers or load_official_scorers()

    bleu_score, bleu_rows = scorers["bleu"].compute_score(ground_truth, generated)
    for order in range(4):
        name = f"bleu_{order + 1}"
        summary[name] = float(bleu_score[order])
        for row, value in zip(per_study, bleu_rows[order], strict=True):
            row[name] = float(value)

    for name in ("rouge_l", "meteor", "cider"):
        scorer = scorers[name]
        score, values = scorer.compute_score(ground_truth, generated)
        summary[name] = float(score)
        for row, value in zip(per_study, values, strict=True):
            row[name] = float(value)
    return summary, per_study


def official_chexbert_metrics(
    hypotheses: list[str], references: list[str], scorer: Any | None = None
) -> tuple[dict[str, float], list[float]]:
    """Official F1CheXbert package used by Stanford AIMI's RRG24 evaluation."""
    scorer = scorer or load_official_scorers()["chexbert"]
    accuracy, per_study, report_all, report_five = scorer(hyps=hypotheses, refs=references)
    summary = {
        "chexbert_accuracy": _as_float(accuracy),
        "chexbert_all_micro_f1": _classification_f1(report_all, "micro avg"),
        "chexbert_all_macro_f1": _classification_f1(report_all, "macro avg"),
        "chexbert_five_micro_f1": _classification_f1(report_five, "micro avg"),
        "chexbert_five_macro_f1": _classification_f1(report_five, "macro avg"),
    }
    values = np.asarray(per_study, dtype=np.float64).reshape(-1).tolist()
    if len(values) != len(hypotheses):
        raise ValueError("F1CheXbert did not return one accuracy value per study")
    return summary, [float(value) for value in values]


def official_radgraph_metrics(
    hypotheses: list[str], references: list[str], scorer: Any | None = None
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Official F1RadGraph entity, entity-relation, and complete rewards."""
    scorer = scorer or load_official_scorers()["radgraph"]
    means, rewards, _, _ = scorer(
        hyps=hypotheses,
        refs=references,
    )
    names = ("radgraph_entity", "radgraph_entity_relation", "radgraph_complete")
    summary = {name: _as_float(value) for name, value in zip(names, means, strict=True)}
    matrix = np.asarray(rewards, dtype=np.float64)
    if matrix.shape == (len(names), len(hypotheses)):
        matrix = matrix.T
    if matrix.shape != (len(hypotheses), len(names)):
        raise ValueError(f"Unexpected F1RadGraph per-study shape {matrix.shape}")
    rows = [
        {name: float(value) for name, value in zip(names, row, strict=True)} for row in matrix
    ]
    return summary, rows


def evaluate_method(
    name: str,
    records: list[dict[str, Any]],
    references: dict[str, str],
    expected_ids: set[str],
    training_reports: list[str],
    linker: DeterministicLinker,
    scorers: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {str(record["example_id"]): record for record in records}
    integrity = retrieval_integrity_audit(records, expected_ids)
    if not integrity["passed"]:
        raise ValueError(f"Integrity audit failed for {name}: {integrity}")
    ordered_ids = sorted(expected_ids)
    ordered = [by_id[example_id] for example_id in ordered_ids]
    hypotheses = [str(record["final_report"]) for record in ordered]
    refs = [references[example_id] for example_id in ordered_ids]

    coco, per_study = official_coco_metrics(hypotheses, refs, scorers)
    chexbert, chexbert_rows = official_chexbert_metrics(
        hypotheses, refs, scorers["chexbert"]
    )
    radgraph, radgraph_rows = official_radgraph_metrics(
        hypotheses, refs, scorers["radgraph"]
    )
    for row, chexbert_value, radgraph_row in zip(
        per_study, chexbert_rows, radgraph_rows, strict=True
    ):
        row["chexbert_sample_accuracy"] = chexbert_value
        row.update(radgraph_row)
    for row, example_id in zip(per_study, ordered_ids, strict=True):
        row.update({"example_id": example_id, "ablation": name})

    summary: dict[str, Any] = {
        "examples": len(ordered),
        "official_metrics": {**coco, **chexbert, **radgraph},
        "pipeline": aggregate_results(ordered),
        "resources": resource_summary(ordered),
        "integrity": integrity,
        "leakage": leakage_audit(hypotheses, training_reports),
        "diversity": prediction_diversity(hypotheses),
        "linker_predictions": linker_structural_audit(hypotheses, linker),
    }
    return summary, per_study


def paired_intervals(
    per_method: dict[str, list[dict[str, Any]]],
    baseline: str,
    samples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float | int]]]:
    if baseline not in per_method:
        raise ValueError(f"Baseline {baseline!r} is absent; choices={sorted(per_method)}")
    baseline_rows = {row["example_id"]: row for row in per_method[baseline]}
    metrics = (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "rouge_l",
        "meteor",
        "cider",
        "chexbert_sample_accuracy",
        "radgraph_entity",
        "radgraph_entity_relation",
        "radgraph_complete",
    )
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    tests: list[dict[str, Any]] = []
    for method, rows in per_method.items():
        if method == baseline:
            continue
        candidate_rows = {row["example_id"]: row for row in rows}
        ids = sorted(set(baseline_rows) & set(candidate_rows))
        output[method] = {}
        for offset, metric in enumerate(metrics):
            interval = paired_bootstrap_delta(
                [float(candidate_rows[item][metric]) for item in ids],
                [float(baseline_rows[item][metric]) for item in ids],
                samples=samples,
                seed=seed + offset,
            )
            output[method][metric] = interval
            tests.append({"method": method, "metric": metric, "result": interval})
    _apply_holm_correction(tests)
    return output


def write_expert_review_packets(
    records_by_method: dict[str, list[dict[str, Any]]],
    studies_by_id: dict[str, Any],
    linker: DeterministicLinker,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    example_ids = sorted(studies_by_id)
    selected = rng.sample(example_ids, min(sample_size, len(example_ids)))
    records_index = {
        method: {str(record["example_id"]): record for record in rows}
        for method, rows in records_by_method.items()
    }
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, str]] = []
    candidate_number = 0
    for example_id in selected:
        candidates = list(records_by_method)
        rng.shuffle(candidates)
        study = studies_by_id[example_id]
        for method in candidates:
            candidate_number += 1
            blinded_id = f"R{candidate_number:06d}"
            record = records_index[method][example_id]
            blinded_rows.append(
                {
                    "blinded_id": blinded_id,
                    "example_id": example_id,
                    "image_path": study.image_path,
                    "indication": study.indication,
                    "generated_report": record["final_report"],
                    "reference_report": study.report,
                    "hallucination_present_0_or_1": "",
                    "omission_present_0_or_1": "",
                    "clinical_error_severity_0_to_3": "",
                    "overall_quality_1_to_5": "",
                    "reviewer_comments": "",
                }
            )
            key_rows.append({"blinded_id": blinded_id, "ablation": method})
    _write_csv(output_dir / "expert_review_packet.csv", blinded_rows)
    _write_csv(output_dir / "expert_review_blinding_key.csv", key_rows)
    (output_dir / "expert_review_protocol.md").write_text(
        "# Blinded expert review protocol\n\n"
        "1. Use at least two qualified radiology reviewers working independently.\n"
        "2. Review the radiograph, indication, generated report, and reference without "
        "opening `expert_review_blinding_key.csv`.\n"
        "3. Score hallucinations, omissions, clinical severity, and overall quality using "
        "the packet columns; add concise evidence in comments.\n"
        "4. Adjudicate disagreements before unblinding and report inter-rater agreement.\n"
        "5. Freeze the completed packet before opening the key or testing system effects.\n\n"
        "No hallucination-reduction claim is permitted while review status is pending.\n",
        encoding="utf-8",
    )

    linker_rows: list[dict[str, Any]] = []
    for example_id in selected:
        study = studies_by_id[example_id]
        for claim in linker.claims(study.report):
            linker_rows.append(
                {
                    "example_id": example_id,
                    "claim": claim.text,
                    "predicted_mentions": " | ".join(entity.mention for entity in claim.entities),
                    "predicted_primekg_ids": " | ".join(
                        entity.entity_id for entity in claim.entities
                    ),
                    "predicted_negated": " | ".join(
                        str(entity.negated) for entity in claim.entities
                    ),
                    "gold_clinical_claim_0_or_1": "",
                    "gold_primekg_ids": "",
                    "gold_negated": "",
                    "linking_correct_0_or_1": "",
                    "reviewer_comments": "",
                }
            )
    _write_csv(output_dir / "linker_expert_review_packet.csv", linker_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _apply_holm_correction(tests: list[dict[str, Any]]) -> None:
    ordered = sorted(tests, key=lambda item: item["result"]["two_sided_p"])
    adjusted = 0.0
    total = len(ordered)
    for rank, item in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, (total - rank) * item["result"]["two_sided_p"]))
        item["result"]["holm_adjusted_p"] = adjusted


def _as_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _classification_f1(report: dict[str, Any], average: str) -> float:
    return float(report[average]["f1-score"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--primekg-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", default="full_adaptive")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--expert-sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    studies = load_manifest(args.manifest)
    training_reports = [study.report for study in studies if study.split == "train"]
    test = [study for study in studies if study.split == "test"]
    studies_by_id = {manifest_example_id(study): study for study in test}
    if len(studies_by_id) != len(test):
        raise ValueError("Manifest contains duplicate image rows in the official test split")
    references = {example_id: study.report for example_id, study in studies_by_id.items()}
    if not references or not all(references.values()):
        raise ValueError(
            "Official test references are required only in this post-inference process"
        )

    files = sorted(args.predictions_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No ablation JSONL files in {args.predictions_dir}")
    records_by_method = {path.stem: load_jsonl(path) for path in files}
    graph = KnowledgeGraph.from_cache(args.primekg_cache)
    linker = DeterministicLinker.from_knowledge_graph(graph)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorers = load_official_scorers()

    summaries: dict[str, Any] = {}
    per_method: dict[str, list[dict[str, Any]]] = {}
    for method, records in tqdm(
        records_by_method.items(),
        desc="Official metrics and integrity audits",
        unit="ablation",
        dynamic_ncols=True,
    ):
        summaries[method], per_method[method] = evaluate_method(
            method,
            records,
            references,
            set(references),
            training_reports,
            linker,
            scorers,
        )

    intervals = paired_intervals(
        per_method,
        baseline=args.baseline,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_expert_review_packets(
        records_by_method,
        studies_by_id,
        linker,
        args.output_dir,
        args.expert_sample_size,
        args.seed,
    )
    publication = {
        "reference_firewall": (
            "Test references were loaded only by this post-inference evaluator; inference JSONL "
            "files contain no reference field."
        ),
        "ltn_interpretation": (
            "LTN values are implemented constraint-satisfaction scores, not calibrated truth."
        ),
        "trace_interpretation": (
            "Traces are procedural inference records, not complete causal explanations."
        ),
        "bootstrap_estimand": "paired mean per-study metric difference",
        "expert_review_status": "PENDING_HUMAN_REVIEW",
        "linker_accuracy_status": "PENDING_EXPERT_ADJUDICATION",
        "claim_policy": (
            "Do not claim hallucination reduction until the blinded expert packet is completed "
            "and adjudicated."
        ),
        "linker_reference_audit": linker_structural_audit(
            [study.report for study in test], linker
        ),
        "methods": summaries,
        "paired_bootstrap_vs_baseline": intervals,
    }
    (args.output_dir / "publication_metrics.json").write_text(
        json.dumps(publication, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_study_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for rows in per_method.values():
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps({"outputs": str(args.output_dir), "methods": sorted(summaries)}, indent=2))


if __name__ == "__main__":
    main()
