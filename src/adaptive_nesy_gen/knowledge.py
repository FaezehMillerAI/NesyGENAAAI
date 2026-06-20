"""Compact, provenance-preserving PrimeKG cache and claim subgraphs."""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from .primekg_io import COMPLETE_EDGE_FILES, iter_primekg_edges, load_primekg_nodes
from .schema import Claim


@dataclass(frozen=True)
class Edge:
    source_id: str
    target_id: str
    relation: str
    display_relation: str
    source_type: str
    target_type: str
    source_name: str = ""
    target_name: str = ""
    confidence: float = 1.0
    edge_source: str = "primekg"


@dataclass(frozen=True)
class ClaimSubgraph:
    nodes: frozenset[str]
    edges: tuple[Edge, ...]
    paths: tuple[tuple[str, ...], ...]


class KnowledgeGraph:
    def __init__(self, edges: list[Edge]):
        self.edges = edges
        self.metadata: dict = {}
        self.node_types: dict[str, str] = {}
        self.node_names: dict[str, str] = {}
        self.node_aliases: dict[str, str] = {}
        self.node_sources: dict[str, str] = {}
        self.adjacency: dict[str, list[tuple[str, Edge]]] = defaultdict(list)
        for edge in edges:
            self.node_types[edge.source_id] = edge.source_type
            self.node_types[edge.target_id] = edge.target_type
            self.node_names[edge.source_id] = edge.source_name or edge.source_id
            self.node_names[edge.target_id] = edge.target_name or edge.target_id
            self.node_aliases[edge.source_id] = self.node_names[edge.source_id]
            self.node_aliases[edge.target_id] = self.node_names[edge.target_id]
            self.adjacency[edge.source_id].append((edge.target_id, edge))
            self.adjacency[edge.target_id].append((edge.source_id, edge))

    @classmethod
    def from_csv(cls, path: str | Path) -> KnowledgeGraph:
        return cls([Edge(**row.__dict__) for row in iter_primekg_edges(path)])

    @classmethod
    def from_cache(cls, path: str | Path) -> KnowledgeGraph:
        """Load a compact cache from an edge CSV/JSONL or a directory containing one."""
        path = Path(path)
        cache_dir = path if path.is_dir() else path.parent
        if path.is_dir():
            for name in (*COMPLETE_EDGE_FILES, "graph.csv", "radiology_edges.csv", "edges.jsonl"):
                candidate = path / name
                if candidate.exists():
                    path = candidate
                    break
            else:
                edges_path = path / "edges.csv"
                if edges_path.exists():
                    path = edges_path
        if path.suffix == ".csv":
            graph = cls.from_csv(path)
            nodes_path = path.parent / "nodes.csv"
            if nodes_path.exists():
                _, nodes = load_primekg_nodes(nodes_path)
                for node_id, node in nodes.items():
                    if node_id in graph.node_types:
                        graph.node_types[node_id] = node["node_type"]
                        graph.node_names[node_id] = node["node_name"]
                        graph.node_aliases[node_id] = node["alias"]
                        graph.node_sources[node_id] = node["node_source"]
            summary_path = cache_dir / "radiology_primekg_summary.json"
            if summary_path.exists():
                graph.metadata = json.loads(summary_path.read_text(encoding="utf-8"))
            return graph
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            for row in rows:
                row.setdefault("display_relation", row.get("relation", "related_to"))
                row.setdefault("confidence", 1.0)
                row.setdefault("edge_source", row.pop("provenance", "primekg"))
            return cls([Edge(**row) for row in rows])
        raise ValueError(f"Unsupported PrimeKG cache format: {path}")

    def radiology_cache(self, training_seed_ids: set[str]) -> KnowledgeGraph:
        """Return edges incident to training-only seeds and their one-hop neighbours."""
        neighbours = set(training_seed_ids)
        for seed in training_seed_ids:
            neighbours.update(node for node, _ in self.adjacency.get(seed, []))
        graph = KnowledgeGraph(
            [
                edge
                for edge in self.edges
                if edge.source_id in neighbours and edge.target_id in neighbours
            ]
        )
        graph.metadata = {**self.metadata, "derived_training_seed_nodes": len(training_seed_ids)}
        return graph

    def relation_ablated(self) -> KnowledgeGraph:
        graph = KnowledgeGraph(
            [
                Edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relation="ablated_relation",
                    display_relation="ablated relation",
                    source_type=edge.source_type,
                    target_type=edge.target_type,
                    source_name=edge.source_name,
                    target_name=edge.target_name,
                    confidence=edge.confidence,
                    edge_source=f"relation-ablated:{edge.edge_source}",
                )
                for edge in self.edges
            ]
        )
        graph.metadata = {**self.metadata, "graph_control": "relation-ablated"}
        return graph

    def shuffled(self, seed: int = 13) -> KnowledgeGraph:
        targets = [
            (edge.target_id, edge.target_type, edge.target_name) for edge in self.edges
        ]
        random.Random(seed).shuffle(targets)
        graph = KnowledgeGraph(
            [
                Edge(
                    source_id=edge.source_id,
                    target_id=target[0],
                    relation=edge.relation,
                    display_relation=edge.display_relation,
                    source_type=edge.source_type,
                    target_type=target[1],
                    source_name=edge.source_name,
                    target_name=target[2],
                    confidence=edge.confidence,
                    edge_source=f"shuffled:{edge.edge_source}",
                )
                for edge, target in zip(self.edges, targets, strict=True)
            ]
        )
        graph.metadata = {**self.metadata, "graph_control": "shuffled", "shuffle_seed": seed}
        return graph

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
