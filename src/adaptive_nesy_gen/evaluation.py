"""Dependency-light integrity, clinical-proxy, and explanation diagnostics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from statistics import mean

import numpy as np

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
) -> dict[str, float | int | str]:
    normalized_train = sorted({" ".join(_tokens(report)) for report in training_reports})
    exact_train = set(normalized_train)
    document_frequency = Counter(
        token for report in normalized_train for token in set(report.split())
    )
    rare_limit = max(256, math.ceil(len(normalized_train) * 0.002))
    postings: dict[str, list[int]] = defaultdict(list)
    length_buckets: dict[int, list[int]] = defaultdict(list)
    for index, report in enumerate(normalized_train):
        length_buckets[len(report) // 40].append(index)
        for token in set(report.split()):
            if document_frequency[token] <= rare_limit:
                postings[token].append(index)
    exact = 0
    overlap = 0
    for prediction in predictions:
        normalized = " ".join(_tokens(prediction))
        exact += normalized in exact_train
        rarest = sorted(
            set(normalized.split()), key=lambda token: document_frequency.get(token, 0)
        )[:5]
        candidates = {
            index for token in rarest for index in postings.get(token, ())
        }
        if not candidates:
            bucket = len(normalized) // 40
            candidates = {
                index
                for nearby in range(max(0, bucket - 1), bucket + 2)
                for index in length_buckets.get(nearby, ())
            }
        minimum_length = len(normalized) * high_overlap / (2.0 - high_overlap)
        maximum_length = len(normalized) * (2.0 - high_overlap) / high_overlap
        best = max(
            (
                SequenceMatcher(None, normalized, normalized_train[index]).ratio()
                for index in candidates
                if minimum_length <= len(normalized_train[index]) <= maximum_length
            ),
            default=0.0,
        )
        overlap += best >= high_overlap
    return {
        "exact_match_count": exact,
        "high_overlap_count": overlap,
        "high_overlap_threshold": high_overlap,
        "high_overlap_candidate_policy": "rare-token index with length-bucket fallback",
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


def resource_summary(records: list[dict]) -> dict[str, float | None]:
    """Aggregate every measured pipeline/resource stage without hiding missing data."""
    result: dict[str, float | None] = {}
    timing_names = ("retrieval", "generation", "verification", "end_to_end")
    for name in timing_names:
        values = [
            float(record["timings_ms"][name])
            for record in records
            if record.get("timings_ms", {}).get(name) is not None
        ]
        result.update(_distribution(values, f"{name}_ms"))
    graph_calls = [float(record.get("graph_calls", 0)) for record in records]
    result.update(_distribution(graph_calls, "graph_calls_per_report"))
    for name in (
        "index_build_ms",
        "indexing_ms",
        "index_load_ms",
        "peak_gpu_memory_gb",
        "index_size_bytes",
    ):
        values = [
            float(record["resources"][name])
            for record in records
            if record.get("resources", {}).get(name) is not None
        ]
        result.update(_distribution(values, name))
    return result


def prediction_diversity(predictions: list[str]) -> dict[str, float | int]:
    normalized = [" ".join(_tokens(text)) for text in predictions]
    counts = Counter(normalized)
    tokenized = [_tokens(text) for text in predictions]
    result: dict[str, float | int] = {
        "prediction_count": len(predictions),
        "unique_prediction_count": len(counts),
        "unique_prediction_ratio": len(counts) / len(predictions) if predictions else 0.0,
        "duplicate_prediction_count": sum(count - 1 for count in counts.values()),
        "most_common_prediction_rate": max(counts.values()) / len(predictions)
        if predictions
        else 0.0,
        "mean_prediction_tokens": mean(map(len, tokenized)) if tokenized else 0.0,
    }
    for order in (1, 2):
        ngrams = [
            tuple(tokens[index : index + order])
            for tokens in tokenized
            for index in range(len(tokens) - order + 1)
        ]
        result[f"distinct_{order}"] = len(set(ngrams)) / len(ngrams) if ngrams else 0.0
    return result


def retrieval_integrity_audit(
    records: list[dict], expected_study_ids: set[str] | None = None
) -> dict[str, bool | int]:
    observed = [str(record.get("example_id") or record["study_id"]) for record in records]
    same_study = 0
    non_training = 0
    for record in records:
        for retrieved in record.get("retrieved_studies", []):
            same_study += str(retrieved.get("study_id")) == str(record["study_id"])
            non_training += retrieved.get("split") != "train"
    duplicates = len(observed) - len(set(observed))
    expected = expected_study_ids if expected_study_ids is not None else set(observed)
    missing = len(expected - set(observed))
    unexpected = len(set(observed) - expected)
    reference_fields = sum("reference" in record for record in records)
    reference_consumed = sum(
        bool(record.get("test_reference_consumed_during_inference")) for record in records
    )
    passed = not any(
        (
            same_study,
            non_training,
            duplicates,
            missing,
            unexpected,
            reference_fields,
            reference_consumed,
        )
    )
    return {
        "passed": passed,
        "same_study_retrievals": same_study,
        "non_training_retrievals": non_training,
        "duplicate_test_records": duplicates,
        "missing_test_records": missing,
        "unexpected_test_records": unexpected,
        "inference_records_containing_references": reference_fields,
        "records_marked_reference_consumed": reference_consumed,
    }


def linker_structural_audit(
    texts: list[str], linker: DeterministicLinker
) -> dict[str, float | int]:
    claims = [claim for text in texts for claim in linker.claims(text)]
    linked = [claim for claim in claims if claim.linked]
    entities = [entity for claim in linked for entity in claim.entities]
    return {
        "reports": len(texts),
        "claims": len(claims),
        "linked_claims": len(linked),
        "claim_coverage": len(linked) / len(claims) if claims else 0.0,
        "linked_mentions": len(entities),
        "unique_primekg_entities": len({entity.entity_id for entity in entities}),
        "negated_mention_rate": sum(entity.negated for entity in entities) / len(entities)
        if entities
        else 0.0,
        "mean_link_confidence": mean(entity.confidence for entity in entities)
        if entities
        else 0.0,
    }


def paired_bootstrap_delta(
    candidate: list[float],
    baseline: list[float],
    samples: int = 2000,
    seed: int = 13,
) -> dict[str, float | int]:
    """Paired percentile CI and two-sided bootstrap p-value for a mean difference."""
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("Paired bootstrap requires equally sized non-empty samples")
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(differences), size=len(differences))
        estimates[sample] = differences[indices].mean()
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    non_positive = (np.count_nonzero(estimates <= 0.0) + 1) / (samples + 1)
    non_negative = (np.count_nonzero(estimates >= 0.0) + 1) / (samples + 1)
    return {
        "pairs": len(differences),
        "bootstrap_samples": samples,
        "mean_delta": float(differences.mean()),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "two_sided_p": float(min(1.0, 2.0 * min(non_positive, non_negative))),
    }


def _distribution(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
    }
