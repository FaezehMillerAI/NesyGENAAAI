"""Transparent consistency gate and evidence-bounded replacement selection."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Claim, ClauseScores, Decision, EvidenceRecord, Replacement, RetrievedStudy
from .text import DeterministicLinker, same_entity_polarity


@dataclass(frozen=True)
class GateOutcome:
    decision: Decision
    confidence: float
    reason: str


def eligible_replacement(
    claim: Claim,
    retrieved: list[RetrievedStudy],
    linker: DeterministicLinker,
    threshold: float,
) -> Replacement | None:
    """Select a sentence with exactly the same IDs and polarity and no new entity."""
    candidates: list[Replacement] = []
    for neighbour in retrieved:
        if neighbour.similarity < threshold:
            continue
        for candidate in linker.claims(neighbour.study.report):
            if same_entity_polarity(claim, candidate) and candidate.text != claim.text:
                candidates.append(
                    Replacement(
                        text=candidate.text,
                        study_id=neighbour.study.study_id,
                        similarity=neighbour.similarity,
                    )
                )
    return max(candidates, key=lambda item: item.similarity, default=None)


class ConsistencyGate:
    def __init__(self, ltn_threshold: float = 0.66, grounding_threshold: float = 0.50):
        self.ltn_threshold = ltn_threshold
        self.grounding_threshold = grounding_threshold

    def decide(
        self,
        claim: Claim,
        evidence: EvidenceRecord,
        clauses: ClauseScores | None,
        replacement: Replacement | None,
        graph_triggered: bool,
    ) -> GateOutcome:
        if not claim.linked:
            return GateOutcome(Decision.ABSTAIN, 0.0, "no reliable clinical entity linked")
        if not graph_triggered:
            confidence = _clip(
                (evidence.s_ground + min(evidence.n_support / 2, 1.0)) / 2
            )
            return GateOutcome(
                Decision.ACCEPT,
                confidence,
                "fast path: retrieval consensus exceeded configured thresholds",
            )
        ltn_ok = clauses is not None and clauses.aggregate >= self.ltn_threshold
        kg_ok = evidence.s_kg >= 1.0
        retrieval_ok = evidence.s_retrieval >= self.grounding_threshold
        if kg_ok and ltn_ok and retrieval_ok:
            confidence = _clip(
                (evidence.s_retrieval + evidence.s_kg + evidence.s_ltn) / 3.0
            )
            return GateOutcome(
                Decision.ACCEPT,
                confidence,
                "linked entity, retrieval grounding, KG status, and constraints agree",
            )
        if replacement is not None:
            confidence = _clip(
                (replacement.similarity + evidence.s_kg + evidence.s_ltn) / 3.0
            )
            return GateOutcome(
                Decision.REVISE,
                confidence,
                "uncertain claim has an evidence-bounded retrieved replacement",
            )
        return GateOutcome(
            Decision.FLAG,
            _clip((evidence.s_kg + evidence.s_ltn + evidence.s_retrieval) / 3.0),
            "linked claim remains disputed and no eligible replacement exists",
        )


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
