"""Dependency-light integrity, clinical-proxy, and explanation diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from difflib import SequenceMatcher
from statistics import mean

from .text import DeterministicLinker


def _tokens(text: str) -> list[str]:
    return [token.strip(".,:;!?()[]").lower() for token in text.split() if token.strip()]


def rouge_l_f1(prediction: str, reference: str) -> float:
    left, right = _tokens(prediction), _tokens(reference)
    if not left or not right:
        return 0.0
    lengths = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, token in enumerate(left, start=1):
        for j, other in enumerate(right, start=1):
            lengths[i][j] = (
                lengths[i - 1][j - 1] + 1
                if token == other
                else max(lengths[i - 1][j], lengths[i][j - 1])
            )
    lcs = lengths[-1][-1]
    precision, recall = lcs / len(left), lcs / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bleu_n(prediction: str, reference: str, n: int = 4) -> float:
    """Single-reference BLEU-N with add-one smoothing for fast experiment checks."""
    predicted, expected = _tokens(prediction), _tokens(reference)
    if not predicted or not expected:
        return 0.0
    precisions = []
    for order in range(1, n + 1):
        pgrams = Counter(tuple(predicted[i : i + order]) for i in range(len(predicted) - order + 1))
        rgrams = Counter(tuple(expected[i : i + order]) for i in range(len(expected) - order + 1))
        overlap = sum(min(count, rgrams[gram]) for gram, count in pgrams.items())
        precisions.append((overlap + 1) / (sum(pgrams.values()) + 1))
    brevity = min(1.0, math.exp(1 - len(expected) / len(predicted)))
    return brevity * math.exp(sum(math.log(value) for value in precisions) / n)


def entity_f1(prediction: str, reference: str, linker: DeterministicLinker) -> dict[str, float]:
    predicted = {
        (entity.entity_id, entity.negated)
        for claim in linker.claims(prediction)
        for entity in claim.entities
    }
    expected = {
        (entity.entity_id, entity.negated)
        for claim in linker.claims(reference)
        for entity in claim.entities
    }
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"entity_precision": precision, "entity_recall": recall, "entity_f1": f1}


def leakage_audit(
    predictions: list[str], training_reports: list[str], high_overlap: float = 0.90
) -> dict[str, float | int]:
    normalized_train = {" ".join(_tokens(report)) for report in training_reports}
    exact = 0
    overlap = 0
    for prediction in predictions:
        normalized = " ".join(_tokens(prediction))
        exact += normalized in normalized_train
        best = max(
            (SequenceMatcher(None, normalized, report).ratio() for report in normalized_train),
            default=0.0,
        )
        overlap += best >= high_overlap
    return {
        "exact_match_count": exact,
        "high_overlap_count": overlap,
        "unique_prediction_ratio": len(set(predictions)) / len(predictions) if predictions else 0.0,
    }


def aggregate_results(records: list[dict]) -> dict[str, float]:
    traces = [trace for record in records for trace in record.get("traces", [])]
    decisions = Counter(trace["gate_decision"] for trace in traces)
    linked = [trace for trace in traces if trace.get("entities")]
    escalated = [trace for trace in traces if trace.get("graph_triggered")]
    grounded = [
        trace
        for trace in linked
        if trace.get("evidence", {}).get("s_retrieval", 0.0) > 0.0
    ]
    path_covered = [trace for trace in escalated if trace.get("graph_paths")]
    required_trace_fields = {
        "original_claim",
        "final_claim",
        "entities",
        "evidence",
        "graph_triggered",
        "gate_decision",
        "gate_reason",
        "latency_ms",
    }
    complete = [trace for trace in traces if required_trace_fields <= trace.keys()]
    result: dict[str, float] = {
        "claims": float(len(traces)),
        "linked_claim_coverage": len(linked) / len(traces) if traces else 0.0,
        "entity_grounding_coverage": len(grounded) / len(linked) if linked else 0.0,
        "escalation_rate_all": len(escalated) / len(traces) if traces else 0.0,
        "escalation_rate_linked": len(escalated) / len(linked) if linked else 0.0,
        "graph_path_coverage": len(path_covered) / len(escalated) if escalated else 0.0,
        "explanation_completeness": len(complete) / len(traces) if traces else 0.0,
        "revision_rate": decisions["REVISE"] / len(traces) if traces else 0.0,
        "mean_claim_latency_ms": mean(trace["latency_ms"] for trace in traces) if traces else 0.0,
        "mean_end_to_end_ms": mean(
            record["timings_ms"]["end_to_end"] for record in records
        )
        if records
        else 0.0,
    }
    for decision in ("ACCEPT", "REVISE", "FLAG", "ABSTAIN"):
        result[f"decision_{decision.lower()}_rate"] = (
            decisions[decision] / len(traces) if traces else 0.0
        )
    return result
