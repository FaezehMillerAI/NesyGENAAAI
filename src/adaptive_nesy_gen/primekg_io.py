"""Streaming readers for complete and node-indexed PrimeKG CSV layouts."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

COMPLETE_EDGE_FILES = ("kg.csv", "kg_giant.csv", "kg_raw.csv", "kg_grouped.csv")


@dataclass(frozen=True)
class PrimeKGLayout:
    edge_path: Path
    nodes_path: Path | None
    indexed_edges: bool


@dataclass(frozen=True)
class PrimeKGEdgeRow:
    source_id: str
    target_id: str
    relation: str
    display_relation: str
    source_type: str
    target_type: str
    source_name: str
    target_name: str
    confidence: float
    edge_source: str


def _columns(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return set(next(csv.reader(handle), []))


def find_primekg_layout(path: str | Path) -> PrimeKGLayout:
    path = Path(path)
    if path.is_file():
        columns = _columns(path)
        indexed = {"x_index", "y_index"} <= columns and not (
            {"x_id", "y_id"} <= columns
        )
        nodes = path.parent / "nodes.csv"
        if indexed and not nodes.exists():
            raise FileNotFoundError(f"Indexed PrimeKG edges require {nodes}")
        return PrimeKGLayout(path, nodes if nodes.exists() else None, indexed)
    if not path.is_dir():
        raise FileNotFoundError(path)
    for name in COMPLETE_EDGE_FILES:
        candidate = path / name
        if candidate.exists() and (
            {"x_id", "y_id"} <= _columns(candidate)
            or {"source_id", "target_id"} <= _columns(candidate)
        ):
            nodes = path / "nodes.csv"
            return PrimeKGLayout(candidate, nodes if nodes.exists() else None, False)
    edges = path / "edges.csv"
    nodes = path / "nodes.csv"
    if edges.exists() and nodes.exists() and {"x_index", "y_index"} <= _columns(edges):
        return PrimeKGLayout(edges, nodes, True)
    searched = ", ".join((*COMPLETE_EDGE_FILES, "nodes.csv + edges.csv"))
    raise FileNotFoundError(f"No supported PrimeKG layout in {path}; searched {searched}")


def load_primekg_nodes(path: str | Path) -> tuple[dict[str, dict], dict[str, dict]]:
    by_index: dict[str, dict] = {}
    by_id: dict[str, dict] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            node_id = str(row.get("node_id") or row.get("id") or "")
            if not node_id:
                continue
            normalized = {
                "node_id": node_id,
                "node_name": str(row.get("node_name") or row.get("name") or node_id),
                "node_type": str(row.get("node_type") or row.get("type") or "unknown"),
                "node_source": str(row.get("node_source") or row.get("source") or ""),
            }
            normalized["alias"] = str(row.get("alias") or normalized["node_name"])
            by_id[node_id] = normalized
            index = row.get("node_index")
            if index not in {None, ""}:
                by_index[str(index)] = normalized
    return by_index, by_id


def _value(row: dict, *names: str, default: str = "") -> str:
    return next(
        (str(row[name]) for name in names if row.get(name) not in {None, ""}),
        default,
    )


def _confidence(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def iter_primekg_edges(path: str | Path) -> Iterator[PrimeKGEdgeRow]:
    layout = find_primekg_layout(path)
    by_index, by_id = ({}, {})
    if layout.nodes_path:
        by_index, by_id = load_primekg_nodes(layout.nodes_path)
    with layout.edge_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if layout.indexed_edges:
                source = by_index.get(_value(row, "x_index"))
                target = by_index.get(_value(row, "y_index"))
                if source is None or target is None:
                    continue
                source_id, target_id = source["node_id"], target["node_id"]
            else:
                source_id = _value(row, "source_id", "x_id", "x_index")
                target_id = _value(row, "target_id", "y_id", "y_index")
                source = by_id.get(source_id, {})
                target = by_id.get(target_id, {})
            relation = _value(row, "relation", "display_relation", default="related_to")
            display_relation = _value(row, "display_relation", "relation", default=relation)
            yield PrimeKGEdgeRow(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                display_relation=display_relation,
                source_type=_value(
                    row, "source_type", "x_type", default=source.get("node_type", "unknown")
                ),
                target_type=_value(
                    row, "target_type", "y_type", default=target.get("node_type", "unknown")
                ),
                source_name=_value(
                    row, "source_name", "x_name", default=source.get("node_name", source_id)
                ),
                target_name=_value(
                    row, "target_name", "y_name", default=target.get("node_name", target_id)
                ),
                confidence=_confidence(_value(row, "confidence", default="1.0")),
                edge_source=_value(row, "source", "edge_source", "provenance", default="primekg"),
            )
