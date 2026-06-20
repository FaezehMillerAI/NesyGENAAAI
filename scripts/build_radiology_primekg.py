#!/usr/bin/env python3
"""Build a training-seeded, radiology-focused PrimeKG cache by streaming raw CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from adaptive_nesy_gen.primekg_io import (
    find_primekg_layout,
    iter_primekg_edges,
    load_primekg_nodes,
)
from adaptive_nesy_gen.retrieval import load_manifest

_TOKEN = re.compile(r"[a-z0-9]+")
_GENERIC_SINGLE_TOKENS = {
    "disease",
    "finding",
    "normal",
    "patient",
    "study",
    "treatment",
}


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.lower()))


class NodeNameMatcher:
    """Exact normalized n-gram matcher suitable for a large PrimeKG node table."""

    def __init__(self, nodes: dict[str, dict], max_tokens: int = 8):
        self.names: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for node_id, node in nodes.items():
            for name in {node["node_name"], node.get("alias", node["node_name"])}:
                tokens = _normalized_tokens(name)
                if not tokens or len(tokens) > max_tokens:
                    continue
                if len(tokens) == 1 and (
                    len(tokens[0]) < 4 or tokens[0] in _GENERIC_SINGLE_TOKENS
                ):
                    continue
                self.names[tokens].add(node_id)
        self.lengths = sorted({len(tokens) for tokens in self.names})

    def match(self, text: str) -> set[str]:
        tokens = _normalized_tokens(text)
        matched: set[str] = set()
        for width in self.lengths:
            for start in range(len(tokens) - width + 1):
                matched.update(self.names.get(tokens[start : start + width], ()))
        return matched


def _node_table(primekg_dir: Path) -> dict[str, dict]:
    layout = find_primekg_layout(primekg_dir)
    if layout.nodes_path:
        return load_primekg_nodes(layout.nodes_path)[1]
    nodes: dict[str, dict] = {}
    for edge in tqdm(
        iter_primekg_edges(primekg_dir),
        desc="Reading PrimeKG nodes",
        unit="edge",
        dynamic_ncols=True,
    ):
        nodes[edge.source_id] = {
            "node_id": edge.source_id,
            "node_name": edge.source_name,
            "node_type": edge.source_type,
            "node_source": "",
        }
        nodes[edge.target_id] = {
            "node_id": edge.target_id,
            "node_name": edge.target_name,
            "node_type": edge.target_type,
            "node_source": "",
        }
    return nodes


def build_cache(
    primekg_dir: Path,
    manifest: Path,
    output_dir: Path,
    hops: int = 1,
    seed_split: str = "train",
) -> dict:
    if hops < 0:
        raise ValueError("hops must be non-negative")
    layout = find_primekg_layout(primekg_dir)
    nodes = _node_table(primekg_dir)
    matcher = NodeNameMatcher(nodes)
    redacted_splits = {"train", "val", "test"} - {seed_split}
    studies = [
        study
        for study in load_manifest(manifest, redact_splits=redacted_splits)
        if study.split == seed_split
    ]
    if not studies:
        raise ValueError(f"Manifest has no {seed_split!r} examples")
    seeds: set[str] = set()
    for study in studies:
        seeds.update(matcher.match(f"{study.indication} {study.report}"))
    if not seeds:
        raise ValueError("No PrimeKG seed nodes matched training indication/report text")

    selected = set(seeds)
    frontier = set(seeds)
    for hop in range(hops):
        expanded: set[str] = set()
        for edge in tqdm(
            iter_primekg_edges(primekg_dir),
            desc=f"Expanding PrimeKG hop {hop + 1}/{hops}",
            unit="edge",
            dynamic_ncols=True,
        ):
            if edge.source_id in frontier or edge.target_id in frontier:
                expanded.update((edge.source_id, edge.target_id))
        frontier = expanded - selected
        selected.update(expanded)
        if not frontier:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    kg_path = output_dir / "kg.csv"
    node_path = output_dir / "nodes.csv"
    edge_count = 0
    with kg_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "x_id",
            "x_name",
            "x_type",
            "y_id",
            "y_name",
            "y_type",
            "relation",
            "display_relation",
            "confidence",
            "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for edge in tqdm(
            iter_primekg_edges(primekg_dir),
            desc="Writing radiology edges",
            unit="edge",
            dynamic_ncols=True,
        ):
            if edge.source_id not in selected or edge.target_id not in selected:
                continue
            writer.writerow(
                {
                    "x_id": edge.source_id,
                    "x_name": edge.source_name,
                    "x_type": edge.source_type,
                    "y_id": edge.target_id,
                    "y_name": edge.target_name,
                    "y_type": edge.target_type,
                    "relation": edge.relation,
                    "display_relation": edge.display_relation,
                    "confidence": edge.confidence,
                    "source": edge.edge_source,
                }
            )
            edge_count += 1
    with node_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["node_id", "node_name", "node_type", "alias", "node_source"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for node_id in sorted(selected):
            node = nodes.get(node_id)
            if node is None:
                continue
            writer.writerow(
                {
                    **node,
                    "alias": node["node_name"],
                }
            )
    summary = {
        "source_primekg_dir": str(primekg_dir),
        "source_kg_csv": str(layout.edge_path),
        "manifest": str(manifest),
        "seed_split": seed_split,
        "seed_policy": "exact_normalized_node_name_ngram",
        "hops": hops,
        "manifest_examples_scanned": len(studies),
        "seed_nodes": len(seeds),
        "subgraph_nodes": len(selected),
        "subgraph_edges": edge_count,
        "kg_csv": str(kg_path),
        "nodes_csv": str(node_path),
    }
    (output_dir / "radiology_primekg_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primekg-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--seed-split", default="train")
    args = parser.parse_args()
    summary = build_cache(
        args.primekg_dir,
        args.manifest,
        args.output_dir,
        args.hops,
        args.seed_split,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
