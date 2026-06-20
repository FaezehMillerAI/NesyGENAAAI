"""Claim grounding and LTN-style fuzzy constraint satisfaction."""

from __future__ import annotations

from dataclasses import dataclass

from .knowledge import ClaimSubgraph, KnowledgeGraph
from .schema import Claim, ClauseScores, RetrievedStudy
from .text import DeterministicLinker

_BIO_RELATIONS = {
    "associated_with",
    "phenotype_present",
    "phenotype_absent",
    "causes",
    "contraindication",
}
_DIAG_RELATIONS = {"associated_with", "diagnoses", "indication", "phenotype_present"}
_LOCATION_RELATIONS = {"located_in", "anatomical_site", "part_of"}


@dataclass(frozen=True)
class Grounding:
    retrieval_score: float
    support_count: int


def entity_specific_grounding(
    claim: Claim,
    retrieved: list[RetrievedStudy],
    linker: DeterministicLinker,
) -> Grounding:
    """Count only neighbours matching every linked entity and its assertion polarity."""
    target = {(entity.entity_id, entity.negated) for entity in claim.entities}
    if not target:
        return Grounding(0.0, 0)
    supporting: set[str] = set()
    best = 0.0
    for neighbour in retrieved:
        for candidate in linker.claims(neighbour.study.report):
            signature = {(entity.entity_id, entity.negated) for entity in candidate.entities}
            if target.issubset(signature):
                supporting.add(neighbour.study.study_id)
                best = max(best, neighbour.similarity)
                break
    return Grounding(best, len(supporting))


class LTNVerifier:
    """Deterministic fuzzy evaluator over the implemented graph constraints.

    Scores are clause-satisfaction values, not probabilities of clinical truth.
    A value of 0.5 denotes a clause that is not applicable or has no graph evidence.
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    @staticmethod
    def _relation_score(subgraph: ClaimSubgraph, relations: set[str]) -> float:
        applicable = [edge for edge in subgraph.edges if edge.relation in relations]
        if not applicable:
            return 0.5
        # Smooth existential (probabilistic sum) over deterministic edge truths.
        return 1.0 - (0.2 ** len(applicable))

    def verify(self, claim: Claim, subgraph: ClaimSubgraph) -> ClauseScores:
        node_coverage = sum(self.graph.has_node(entity.entity_id) for entity in claim.entities)
        biological = node_coverage / len(claim.entities) if claim.entities else 0.0
        biological = (biological + self._relation_score(subgraph, _BIO_RELATIONS)) / 2.0
        diagnostic = self._relation_score(subgraph, _DIAG_RELATIONS)

        anatomy_present = any(entity.entity_type == "anatomy" for entity in claim.entities)
        finding_present = any(
            entity.entity_type in {"finding", "disease"} for entity in claim.entities
        )
        location = (
            self._relation_score(subgraph, _LOCATION_RELATIONS)
            if anatomy_present or finding_present
            else 0.5
        )
        return ClauseScores(biological=biological, diagnostic=diagnostic, location=location)
