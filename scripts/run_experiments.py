#!/usr/bin/env python3
"""Run generation once or replay drafts through adaptive verifier ablations."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

from tqdm.auto import tqdm

from adaptive_nesy_gen.backends import (
    CheXagentBackend,
    MedGemmaBackend,
    RetrievalOnlyBackend,
    StaticBackend,
)
from adaptive_nesy_gen.experiments import ablation_configs
from adaptive_nesy_gen.knowledge import KnowledgeGraph
from adaptive_nesy_gen.pipeline import AdaptiveNesyGen
from adaptive_nesy_gen.retrieval import (
    MedSigLIPEncoder,
    VisualIndex,
    load_manifest,
    manifest_example_id,
)
from adaptive_nesy_gen.schema import PipelineConfig
from adaptive_nesy_gen.text import DeterministicLinker


def load_drafts(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    drafts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row.get("example_id") or row["study_id"])
            if key in drafts:
                raise ValueError(f"Duplicate replay key {key} in {path}")
            drafts[key] = row.get("raw_report") or row["final_report"]
    return drafts


def backend_from_args(args):
    if args.backend == "chexagent":
        return CheXagentBackend(
            adapter=args.adapter,
            use_retrieval=args.drafting_mode == "few-shot",
        )
    if args.backend == "medgemma":
        return MedGemmaBackend(use_retrieval=args.drafting_mode == "few-shot")
    if args.backend == "retrieval":
        return RetrievalOnlyBackend()
    return None


def make_record(result, study, args, index_load_ms, ablation, graph_control):
    record = result.to_dict()
    record.update(
        {
            "study_id": study.study_id,
            "example_id": manifest_example_id(study),
            "backend": args.backend,
            "drafting_mode": args.drafting_mode,
            "ablation": ablation,
            "test_reference_consumed_during_inference": False,
            "reference_access_policy": "test reports redacted at manifest ingestion",
            "resources": {
                "index_build_ms": None,
                "indexing_ms": index_load_ms,
                "index_load_ms": index_load_ms,
                "index_size_bytes": args.medsiglip_cache.stat().st_size,
                "peak_gpu_memory_gb": _peak_gpu_memory_gb(),
            },
            "graph_control": graph_control,
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--medsiglip-cache", required=True, type=Path)
    parser.add_argument("--primekg-cache", required=True, type=Path)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--backend",
        choices=["chexagent", "medgemma", "retrieval", "replay"],
        required=True,
    )
    parser.add_argument("--drafting-mode", choices=["zero-shot", "few-shot"], default="few-shot")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--drafts-jsonl", type=Path)
    parser.add_argument("--ablation", default="full_adaptive")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--exclude-same-patient", action="store_true")
    parser.add_argument(
        "--graph-control", choices=["none", "relation-ablated", "shuffled"], default="none"
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="replay every verifier ablation in one model/index load; output is a directory",
    )
    args = parser.parse_args()

    if args.backend == "replay" and not args.drafts_jsonl:
        parser.error("--drafts-jsonl is required for replay")
    if args.suite and args.backend != "replay":
        parser.error("--suite requires --backend replay")
    studies = load_manifest(args.manifest, redact_splits={"test"})
    train = [study for study in studies if study.split == "train"]
    test = [study for study in studies if study.split == "test"]
    if not train or not test:
        raise ValueError("Manifest must contain train and test rows")
    if any(study.report for study in test):
        raise AssertionError("Test-reference firewall failed")
    encoder = MedSigLIPEncoder()
    index_started = time.perf_counter()
    index = VisualIndex.load(args.medsiglip_cache, encoder, studies=studies)
    index_load_ms = (time.perf_counter() - index_started) * 1000.0
    original_graph = KnowledgeGraph.from_cache(args.primekg_cache)
    linker = (
        DeterministicLinker.from_json(args.lexicon)
        if args.lexicon
        else DeterministicLinker.from_knowledge_graph(original_graph)
    )
    graph = original_graph
    if args.graph_control == "relation-ablated":
        graph = graph.relation_ablated()
    elif args.graph_control == "shuffled":
        graph = graph.shuffled(seed=13)
    base = PipelineConfig(top_k=args.top_k)
    configs = ablation_configs(base)
    if args.ablation not in configs:
        raise ValueError(f"Unknown ablation {args.ablation}; choose from {sorted(configs)}")
    config = configs[args.ablation]
    shared_backend = backend_from_args(args)
    drafts = load_drafts(args.drafts_jsonl)
    if args.suite:
        _run_suite(
            args,
            test,
            index,
            linker,
            original_graph,
            configs,
            drafts,
            index_load_ms,
        )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_test = test[: args.limit]
    with args.output.open("w", encoding="utf-8") as handle:
        for study in tqdm(
            selected_test,
            desc="Generating and verifying reports",
            unit="study",
            dynamic_ncols=True,
        ):
            replay_key = manifest_example_id(study)
            if args.backend == "replay":
                if replay_key not in drafts:
                    raise KeyError(f"No replay draft for test example {replay_key}")
                backend = StaticBackend(drafts[replay_key])
            else:
                backend = shared_backend
            pipeline = AdaptiveNesyGen(index, linker, graph, backend, config)
            result = pipeline.run(
                study.image_path,
                study.indication,
                study_id=study.study_id,
                patient_id=study.patient_id if args.exclude_same_patient else None,
            )
            record = make_record(
                result,
                study,
                args,
                index_load_ms,
                args.ablation,
                args.graph_control,
            )
            handle.write(json.dumps(record) + "\n")
            handle.flush()


def _run_suite(args, test, index, linker, graph, configs, drafts, index_load_ms):
    args.output.mkdir(parents=True, exist_ok=True)
    for stale_output in args.output.glob("*.jsonl"):
        stale_output.unlink()
    entries = [(name, config, graph, "none") for name, config in configs.items()]
    entries.extend(
        [
            (
                "full_adaptive_relation-ablated",
                configs["full_adaptive"],
                graph.relation_ablated(),
                "relation-ablated",
            ),
            (
                "full_adaptive_shuffled",
                configs["full_adaptive"],
                graph.shuffled(seed=13),
                "shuffled",
            ),
        ]
    )
    selected_test = test[: args.limit]
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context((args.output / f"{name}.jsonl").open("w", encoding="utf-8"))
            for name, _, _, _ in entries
        }
        study_progress = tqdm(
            selected_test,
            desc="Replaying studies",
            unit="study",
            dynamic_ncols=True,
        )
        ablation_progress = tqdm(
            total=len(selected_test) * len(entries),
            desc="Verifier ablations",
            unit="run",
            dynamic_ncols=True,
        )
        for study in study_progress:
            replay_key = manifest_example_id(study)
            if replay_key not in drafts:
                raise KeyError(f"No replay draft for test example {replay_key}")
            for name, config, controlled_graph, control in entries:
                pipeline = AdaptiveNesyGen(
                    index,
                    linker,
                    controlled_graph,
                    StaticBackend(drafts[replay_key]),
                    config,
                )
                result = pipeline.run(
                    study.image_path,
                    study.indication,
                    study_id=study.study_id,
                    patient_id=study.patient_id if args.exclude_same_patient else None,
                )
                record = make_record(
                    result, study, args, index_load_ms, name, control
                )
                handles[name].write(json.dumps(record) + "\n")
                handles[name].flush()
                ablation_progress.update()
        ablation_progress.close()


def _peak_gpu_memory_gb() -> float:
    try:
        import torch

        return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    except ImportError:
        return 0.0


if __name__ == "__main__":
    main()
