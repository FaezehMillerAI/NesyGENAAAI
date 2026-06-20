"""Typed contracts shared by the Adaptive NeSy-Gen stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    FLAG = "FLAG"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Study:
    study_id: str
    image_path: str
    report: str
    indication: str = ""
    split: str = "train"
    patient_id: str = ""
    view: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedStudy:
    study: Study
    similarity: float


@dataclass(frozen=True)
class LinkedEntity:
    mention: str
    entity_id: str
    canonical_name: str
    entity_type: str
    negated: bool
    confidence: float


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    entities: tuple[LinkedEntity, ...] = ()

    @property
    def linked(self) -> bool:
        return bool(self.entities)


@dataclass(frozen=True)
class ClauseScores:
    biological: float
    diagnostic: float
    location: float

    @property
    def aggregate(self) -> float:
        return (self.biological + self.diagnostic + self.location) / 3.0


@dataclass(frozen=True)
class EvidenceRecord:
    s_visual: float = 0.0
    s_retrieval: float = 0.0
    n_support: int = 0
    s_kg: float = 0.0
    s_ltn: float = 0.0
    s_gate: float = 0.0

    @property
    def s_ground(self) -> float:
        return max(self.s_visual, self.s_retrieval)


@dataclass(frozen=True)
class Replacement:
    text: str
    study_id: str
    similarity: float


@dataclass
class ClaimTrace:
    claim_id: str
    original_claim: str
    final_claim: str
    entities: list[dict[str, Any]]
    evidence: dict[str, Any]
    graph_triggered: bool
    clause_scores: dict[str, float] | None
    primekg_node_status: dict[str, bool]
    graph_paths: list[list[str]]
    graph_provenance: list[dict[str, str]]
    gate_decision: str
    gate_reason: str
    replacement_provenance: dict[str, Any] | None
    latency_ms: float


@dataclass
class ReportResult:
    raw_report: str
    final_report: str
    traces: list[ClaimTrace]
    retrieved_studies: list[dict[str, Any]]
    timings_ms: dict[str, float]
    graph_calls: int

    @property
    def claims(self) -> int:
        return len(self.traces)

    @property
    def linked_claims(self) -> int:
        return sum(bool(t.entities) for t in self.traces)

    @property
    def escalation_rate_all(self) -> float:
        return self.graph_calls / self.claims if self.claims else 0.0

    @property
    def escalation_rate_linked(self) -> float:
        return self.graph_calls / self.linked_claims if self.linked_claims else 0.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "escalation_rate_all": self.escalation_rate_all,
                "escalation_rate_linked": self.escalation_rate_linked,
            }
        )
        return value

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int = 3
    tau_fast: float = 0.85
    min_support: int = 2
    tau_revise: float = 0.75
    tau_accept_ltn: float = 0.66
    tau_accept_ground: float = 0.50
    enable_graph: bool = True
    enable_ltn: bool = True
    enable_gate: bool = True
    enable_revision: bool = True
    always_verify: bool = False
    claim_level: bool = True
    random_seed: int = 7
    metadata: dict[str, Any] = field(default_factory=dict)
