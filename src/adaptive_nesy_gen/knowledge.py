"""Compact, provenance-preserving PrimeKG cache and claim subgraphs."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from .schema import Claim


@dataclass(frozen=True)
class Edge:
    source_id: str
    target_id: str
    relation: str
    source_type: str
    target_type: str
    source_name: str = ""
    target_name: str = ""
    provenance: str = "PrimeKG"


@dataclass(frozen=True)
class ClaimSubgraph:
    nodes: frozenset[str]
    edges: tuple[Edge, ...]
    paths: tuple[tuple[str, ...], ...]


class KnowledgeGraph:
    def __init__(self, edges: list[Edge]):
        self.edges = edges
        self.node_types: dict[str, str] = {}
        self.node_names: dict[str, str] = {}
        self.adjacency: dict[str, list[tuple[str, Edge]]] = defaultdict(list)
        for edge in edges:
            self.node_types[edge.source_id] = edge.source_type
            self.node_types[edge.target_id] = edge.target_type
            self.node_names[edge.source_id] = edge.source_name or edge.source_id
            self.node_names[edge.target_id] = edge.target_name or edge.target_id
            self.adjacency[edge.source_id].append((edge.target_id, edge))
            self.adjacency[edge.target_id].append((edge.source_id, edge))

    @classmethod
    def from_csv(cls, path: str | Path) -> KnowledgeGraph:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        def value(row, *names, default=""):
            return next(
                (str(row[name]) for name in names if row.get(name) not in {None, ""}),
                default,
            )

        return cls(
            [
                Edge(
                    source_id=value(row, "source_id", "x_id", "x_index"),
                    target_id=value(row, "target_id", "y_id", "y_index"),
                    relation=value(row, "relation", "display_relation"),
                    source_type=value(row, "source_type", "x_type", default="unknown"),
                    target_type=value(row, "target_type", "y_type", default="unknown"),
                    source_name=value(row, "source_name", "x_name"),
                    target_name=value(row, "target_name", "y_name"),
                    provenance=value(row, "provenance", "source", default="PrimeKG"),
                )
                for row in rows
            ]
        )

    @classmethod
    def from_cache(cls, path: str | Path) -> KnowledgeGraph:
        """Load a compact cache from an edge CSV/JSONL or a directory containing one."""
        path = Path(path)
        if path.is_dir():
            preferred = [
                path / name
                for name in ("edges.csv", "graph.csv", "radiology_edges.csv", "edges.jsonl")
            ]
            candidates = [candidate for candidate in preferred if candidate.exists()]
            if not candidates:
                candidates = sorted(path.glob("*.csv")) + sorted(path.glob("*.jsonl"))
            if not candidates:
                raise FileNotFoundError(f"No edge CSV/JSONL found in PrimeKG cache: {path}")
            path = candidates[0]
        if path.suffix == ".csv":
            return cls.from_csv(path)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            return cls([Edge(**row) for row in rows])
        raise ValueError(f"Unsupported PrimeKG cache format: {path}")

    def radiology_cache(self, training_seed_ids: set[str]) -> KnowledgeGraph:
        """Return edges incident to training-only seeds and their one-hop neighbours."""
        neighbours = set(training_seed_ids)
        for seed in training_seed_ids:
            neighbours.update(node for node, _ in self.adjacency.get(seed, []))
        return KnowledgeGraph(
            [
                edge
                for edge in self.edges
                if edge.source_id in neighbours and edge.target_id in neighbours
            ]
        )

    def relation_ablated(self) -> KnowledgeGraph:
        return KnowledgeGraph(
            [
                Edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relation="ablated_relation",
                    source_type=edge.source_type,
                    target_type=edge.target_type,
                    source_name=edge.source_name,
                    target_name=edge.target_name,
                    provenance=f"relation-ablated:{edge.provenance}",
                )
                for edge in self.edges
            ]
        )

    def shuffled(self, seed: int = 13) -> KnowledgeGraph:
        targets = [
            (edge.target_id, edge.target_type, edge.target_name) for edge in self.edges
        ]
        random.Random(seed).shuffle(targets)
        return KnowledgeGraph(
            [
                Edge(
                    source_id=edge.source_id,
                    target_id=target[0],
                    relation=edge.relation,
                    source_type=edge.source_type,
                    target_type=target[1],
                    source_name=edge.source_name,
                    target_name=target[2],
                    provenance=f"shuffled:{edge.provenance}",
                )
                for edge, target in zip(self.edges, targets, strict=True)
            ]
        )

    def has_node(self, node_id: str) -> bool:
        return node_id in self.node_types

    def shortest_path(self, source: str, targets: set[str], max_hops: int = 3) -> tuple[str, ...]:
        if source in targets:
            return (source,)
        queue = deque([(source, (source,))])
        visited = {source}
        while queue:
            node, path = queue.popleft()
            if (len(path) - 1) // 2 >= max_hops:
                continue
            for neighbour, edge in self.adjacency.get(node, []):
                if neighbour in visited:
                    continue
                next_path = (*path, edge.relation, neighbour)
                if neighbour in targets:
                    return next_path
                visited.add(neighbour)
                queue.append((neighbour, next_path))
        return ()

    def subgraph(self, claim: Claim, context: Claim | None = None) -> ClaimSubgraph:
        seeds = {entity.entity_id for entity in claim.entities}
        context_ids = {entity.entity_id for entity in context.entities} if context else set()
        nodes = set(seeds | context_ids)
        selected: dict[tuple[str, str, str], Edge] = {}
        for seed in seeds | context_ids:
            for neighbour, edge in self.adjacency.get(seed, []):
                nodes.add(neighbour)
                selected[(edge.source_id, edge.relation, edge.target_id)] = edge
        paths = [
            (edge.source_id, edge.relation, edge.target_id)
            for edge in selected.values()
            if edge.source_id in seeds or edge.target_id in seeds
        ]
        if context_ids:
            for seed in seeds:
                path = self.shortest_path(seed, context_ids)
                if path:
                    paths.append(path)
        return ClaimSubgraph(
            nodes=frozenset(nodes),
            edges=tuple(selected.values()),
            paths=tuple(paths),
        )
