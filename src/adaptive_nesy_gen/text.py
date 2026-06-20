"""Deterministic clinical claim segmentation and entity normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .schema import Claim, LinkedEntity

_CLAIM_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"\b[\w'-]+\b")
_NEGATORS = {"no", "not", "without", "absent", "negative", "neither", "nor"}


@dataclass(frozen=True)
class LexiconEntry:
    entity_id: str
    canonical_name: str
    entity_type: str
    synonyms: tuple[str, ...]


class DeterministicLinker:
    """Longest-match linker with fixed-window assertion/negation detection.

    This is intentionally reproducible and auditable. It is not a replacement for
    a validated clinical NER/linking system; its quality must be evaluated separately.
    """

    def __init__(self, entries: list[LexiconEntry], negation_window: int = 5):
        self.entries = entries
        self.negation_window = negation_window
        lookup: list[tuple[str, LexiconEntry]] = []
        for entry in entries:
            terms = set(entry.synonyms) | {entry.canonical_name}
            lookup.extend((term.lower(), entry) for term in terms)
        self._lookup = sorted(lookup, key=lambda item: (-len(item[0]), item[0]))

    @classmethod
    def from_json(cls, path: str | Path) -> DeterministicLinker:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [
                LexiconEntry(
                    entity_id=row["entity_id"],
                    canonical_name=row["canonical_name"],
                    entity_type=row["entity_type"],
                    synonyms=tuple(row.get("synonyms", [])),
                )
                for row in records
            ]
        )

    @classmethod
    def from_knowledge_graph(cls, graph) -> DeterministicLinker:
        """Build a reproducible exact-name lexicon from a compact radiology cache."""
        entries = []
        for node_id, name in sorted(graph.node_names.items()):
            name = name.strip()
            if len(name) < 3 or name == node_id:
                continue
            entries.append(
                LexiconEntry(
                    entity_id=node_id,
                    canonical_name=name,
                    entity_type=graph.node_types.get(node_id, "unknown"),
                    synonyms=(),
                )
            )
        if not entries:
            raise ValueError("PrimeKG cache contains no named nodes for deterministic linking")
        return cls(entries)

    def _is_negated(self, text: str, start: int) -> bool:
        prefix = text[:start].lower()
        tokens = _TOKEN.findall(prefix)[-self.negation_window :]
        # A contrast marker starts a fresh assertion scope.
        for marker in ("but", "however", "although"):
            if marker in tokens:
                tokens = tokens[tokens.index(marker) + 1 :]
        return any(token in _NEGATORS for token in tokens)

    def link(self, text: str) -> tuple[LinkedEntity, ...]:
        lowered = text.lower()
        occupied: list[tuple[int, int]] = []
        linked: list[tuple[int, LinkedEntity]] = []
        for term, entry in self._lookup:
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
            for match in pattern.finditer(lowered):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in occupied):
                    continue
                occupied.append(span)
                linked.append(
                    (
                        span[0],
                        LinkedEntity(
                            mention=text[span[0] : span[1]],
                            entity_id=entry.entity_id,
                            canonical_name=entry.canonical_name,
                            entity_type=entry.entity_type,
                            negated=self._is_negated(text, span[0]),
                            confidence=1.0,
                        ),
                    )
                )
        return tuple(entity for _, entity in sorted(linked, key=lambda item: item[0]))

    def claims(self, report: str) -> list[Claim]:
        pieces = [part.strip() for part in _CLAIM_BOUNDARY.split(report.strip()) if part.strip()]
        return [
            Claim(claim_id=f"c{index:03d}", text=text, entities=self.link(text))
            for index, text in enumerate(pieces, start=1)
        ]


def same_entity_polarity(left: Claim, right: Claim) -> bool:
    def signature(claim: Claim) -> set[tuple[str, bool]]:
        return {(entity.entity_id, entity.negated) for entity in claim.entities}

    return bool(left.entities) and signature(left) == signature(right)
