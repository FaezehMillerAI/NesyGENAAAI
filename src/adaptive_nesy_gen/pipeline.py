"""End-to-end adaptive claim audit with inference-faithful traces."""

from __future__ import annotations

import time
from dataclasses import asdict, replace

from .backends import DraftingBackend
from .gate import ConsistencyGate, GateOutcome, eligible_replacement
from .knowledge import KnowledgeGraph
from .retrieval import VisualIndex
from .schema import (
    Claim,
    ClaimTrace,
    ClauseScores,
    Decision,
    EvidenceRecord,
    PipelineConfig,
    ReportResult,
)
from .text import DeterministicLinker
from .verification import LTNVerifier, entity_specific_grounding


class AdaptiveNesyGen:
    def __init__(
        self,
        visual_index: VisualIndex,
        linker: DeterministicLinker,
        graph: KnowledgeGraph,
        backend: DraftingBackend,
        config: PipelineConfig | None = None,
    ):
        self.index = visual_index
        self.linker = linker
        self.graph = graph
        self.backend = backend
        self.config = config or PipelineConfig()
        self.verifier = LTNVerifier(graph)
        self.gate = ConsistencyGate(
            ltn_threshold=self.config.tau_accept_ltn,
            grounding_threshold=self.config.tau_accept_ground,
        )

    def run(
        self,
        image_path: str,
        indication: str = "",
        study_id: str | None = None,
        patient_id: str | None = None,
    ) -> ReportResult:
        total_started = time.perf_counter()
        started = time.perf_counter()
        retrieved = self.index.query(
            image_path,
            self.config.top_k,
            exclude_study_id=study_id,
            exclude_patient_id=patient_id,
        )
        retrieval_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        raw_report = self.backend.generate(image_path, indication, retrieved).strip()
        generation_ms = (time.perf_counter() - started) * 1000.0
        claims = (
            self.linker.claims(raw_report)
            if self.config.claim_level
            else [Claim("c001", raw_report, self.linker.link(raw_report))]
        )
        indication_claims = self.linker.claims(indication)
        indication_context = indication_claims[0] if indication_claims else None
        max_visual = max((item.similarity for item in retrieved), default=0.0)

        traces: list[ClaimTrace] = []
        final_claims: list[str] = []
        graph_calls = 0
        verification_started = time.perf_counter()
        for claim in claims:
            claim_started = time.perf_counter()
            grounding = entity_specific_grounding(claim, retrieved, self.linker)
            evidence = EvidenceRecord(
                s_visual=max_visual,
                s_retrieval=grounding.retrieval_score,
                n_support=grounding.support_count,
            )
            fast = (
                claim.linked
                and evidence.s_ground >= self.config.tau_fast
                and evidence.n_support >= self.config.min_support
                and not self.config.always_verify
            )
            graph_triggered = claim.linked and self.config.enable_graph and not fast
            clauses: ClauseScores | None = None
            node_status: dict[str, bool] = {}
            graph_paths: list[list[str]] = []
            graph_provenance: list[dict[str, str]] = []
            if graph_triggered:
                graph_calls += 1
                subgraph = self.graph.subgraph(claim, indication_context)
                node_status = {
                    entity.entity_id: self.graph.has_node(entity.entity_id)
                    for entity in claim.entities
                }
                kg_score = sum(node_status.values()) / len(node_status) if node_status else 0.0
                clauses = self.verifier.verify(claim, subgraph) if self.config.enable_ltn else None
                evidence = replace(
                    evidence,
                    s_kg=kg_score,
                    s_ltn=clauses.aggregate if clauses else 0.5,
                )
                graph_paths = [list(path) for path in subgraph.paths]
                graph_provenance = [
                    {
                        "source_id": edge.source_id,
                        "relation": edge.relation,
                        "target_id": edge.target_id,
                        "display_relation": edge.display_relation,
                        "confidence": edge.confidence,
                        "edge_source": edge.edge_source,
                    }
                    for edge in subgraph.edges
                ]

            replacement = (
                eligible_replacement(
                    claim, retrieved, self.linker, threshold=self.config.tau_revise
                )
                if self.config.enable_revision
                else None
            )
            outcome = self._decision(claim, evidence, clauses, replacement, graph_triggered, fast)
            evidence = replace(evidence, s_gate=outcome.confidence)
            final_text = (
                replacement.text
                if outcome.decision is Decision.REVISE and replacement is not None
                else claim.text
            )
            final_claims.append(final_text)
            traces.append(
                ClaimTrace(
                    claim_id=claim.claim_id,
                    original_claim=claim.text,
                    final_claim=final_text,
                    entities=[asdict(entity) for entity in claim.entities],
                    evidence=asdict(evidence),
                    graph_triggered=graph_triggered,
                    clause_scores=(
                        {
                            "biological": clauses.biological,
                            "diagnostic": clauses.diagnostic,
                            "location": clauses.location,
                            "aggregate": clauses.aggregate,
                        }
                        if clauses
                        else None
                    ),
                    primekg_node_status=node_status,
                    graph_paths=graph_paths,
                    graph_provenance=graph_provenance,
                    gate_decision=outcome.decision.value,
                    gate_reason=outcome.reason,
                    replacement_provenance=(
                        asdict(replacement) if final_text != claim.text else None
                    ),
                    latency_ms=(time.perf_counter() - claim_started) * 1000.0,
                )
            )
        verification_ms = (time.perf_counter() - verification_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return ReportResult(
            raw_report=raw_report,
            final_report=" ".join(final_claims),
            traces=traces,
            retrieved_studies=[
                {
                    "study_id": item.study.study_id,
                    "patient_id": item.study.patient_id,
                    "similarity": item.similarity,
                    "split": item.study.split,
                }
                for item in retrieved
            ],
            timings_ms={
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "verification": verification_ms,
                "end_to_end": total_ms,
            },
            graph_calls=graph_calls,
            graph_metadata=self.graph.metadata,
        )

    def _decision(
        self, claim, evidence, clauses, replacement, graph_triggered: bool, fast: bool
    ) -> GateOutcome:
        if not claim.linked:
            return GateOutcome(Decision.ABSTAIN, 0.0, "no reliable clinical entity linked")
        if fast:
            return self.gate.decide(claim, evidence, clauses, replacement, False)
        if not self.config.enable_gate:
            return GateOutcome(Decision.ACCEPT, 1.0, "gate disabled by ablation")
        if not self.config.enable_graph:
            if evidence.s_retrieval >= self.config.tau_accept_ground:
                return GateOutcome(Decision.ACCEPT, evidence.s_retrieval, "retrieval-only evidence")
            return GateOutcome(
                Decision.FLAG, evidence.s_retrieval, "insufficient retrieval evidence"
            )
        return self.gate.decide(claim, evidence, clauses, replacement, graph_triggered)
