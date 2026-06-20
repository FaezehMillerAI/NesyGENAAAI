"""Command-line entry points for smoke runs, indexing, and trace evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends import StaticBackend
from .evaluation import aggregate_results, bleu_n, entity_f1, leakage_audit, rouge_l_f1
from .knowledge import KnowledgeGraph
from .pipeline import AdaptiveNesyGen
from .retrieval import PixelHistogramEncoder, VisualIndex, load_manifest
from .schema import PipelineConfig
from .text import DeterministicLinker


def _demo_root() -> Path:
    bundled = Path(__file__).resolve().parent / "demo"
    return bundled if bundled.exists() else Path(__file__).resolve().parents[2] / "data" / "demo"


def run_demo(output: str | None = None) -> dict:
    root = _demo_root()
    studies = load_manifest(root / "manifest.csv")
    encoder = PixelHistogramEncoder()
    index, indexing_ms = VisualIndex.build(studies, encoder)
    linker = DeterministicLinker.from_json(root / "lexicon.json")
    graph = KnowledgeGraph.from_csv(root / "graph.csv")
    query = next(study for study in studies if study.split == "test")
    backend = StaticBackend(
        "No pleural effusion. Mild left basilar opacity. No cardiomegaly. "
        "Support devices are unchanged."
    )
    pipeline = AdaptiveNesyGen(index, linker, graph, backend, PipelineConfig(top_k=3))
    result = pipeline.run(query.image_path, "Dyspnea.", study_id=query.study_id)
    payload = result.to_dict()
    payload["indexing_ms"] = indexing_ms
    if output:
        result.save(output)
    return payload


def _evaluate(path: str, manifest: str, lexicon: str) -> dict:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    studies = load_manifest(manifest)
    linker = DeterministicLinker.from_json(lexicon)
    lexical = []
    for record in records:
        if "reference" not in record:
            continue
        prediction, reference = record["final_report"], record["reference"]
        metrics = {f"bleu_{n}": bleu_n(prediction, reference, n) for n in range(1, 5)}
        metrics["rouge_l"] = rouge_l_f1(prediction, reference)
        metrics.update(entity_f1(prediction, reference, linker))
        lexical.append(metrics)
    summary = aggregate_results(records)
    if lexical:
        summary.update(
            {
                key: sum(row[key] for row in lexical) / len(lexical)
                for key in lexical[0]
            }
        )
    summary.update(
        leakage_audit(
            [record["final_report"] for record in records],
            [study.report for study in studies if study.split == "train"],
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-nesy-gen")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic CPU smoke experiment")
    demo.add_argument("--output", help="optional JSON trace path")
    build = subparsers.add_parser("build-index", help="cache train-split image embeddings")
    build.add_argument("--manifest", required=True)
    build.add_argument("--output", required=True)
    evaluate = subparsers.add_parser("evaluate", help="summarize JSONL experiment outputs")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--lexicon", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        payload = run_demo(args.output)
    elif args.command == "build-index":
        studies = load_manifest(args.manifest)
        index, indexing_ms = VisualIndex.build(studies, PixelHistogramEncoder())
        index.save(args.output)
        payload = {"training_studies": len(index.studies), "indexing_ms": indexing_ms}
    else:
        payload = _evaluate(args.predictions, args.manifest, args.lexicon)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
