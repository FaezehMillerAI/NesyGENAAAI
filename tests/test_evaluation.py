from adaptive_nesy_gen.evaluation import (
    bleu_n,
    leakage_audit,
    paired_bootstrap_delta,
    prediction_diversity,
    resource_summary,
    retrieval_integrity_audit,
    rouge_l_f1,
)


def test_lexical_smoke_metrics_are_bounded():
    prediction = "No pleural effusion."
    reference = "No pleural effusion is seen."
    assert 0.0 <= bleu_n(prediction, reference, 4) <= 1.0
    assert 0.0 <= rouge_l_f1(prediction, reference) <= 1.0


def test_leakage_audit_counts_exact_matches():
    result = leakage_audit(["No edema.", "No effusion."], ["No edema."])
    assert result["exact_match_count"] == 1
    assert result["unique_prediction_ratio"] == 1.0


def test_publication_integrity_and_resource_summaries():
    record = {
        "study_id": "test-1",
        "retrieved_studies": [{"study_id": "train-1", "split": "train"}],
        "test_reference_consumed_during_inference": False,
        "timings_ms": {
            "retrieval": 1.0,
            "generation": 2.0,
            "verification": 3.0,
            "end_to_end": 6.0,
        },
        "graph_calls": 2,
        "resources": {
            "index_build_ms": None,
            "indexing_ms": 4.0,
            "index_load_ms": 4.0,
            "peak_gpu_memory_gb": 5.0,
            "index_size_bytes": 6.0,
        },
    }
    audit = retrieval_integrity_audit([record], {"test-1"})
    assert audit["passed"] is True
    resources = resource_summary([record])
    assert resources["verification_ms_mean"] == 3.0
    assert resources["graph_calls_per_report_mean"] == 2.0
    assert resources["indexing_ms_mean"] == 4.0
    assert resources["index_build_ms_mean"] is None


def test_prediction_diversity_and_paired_bootstrap():
    diversity = prediction_diversity(["No edema.", "No edema.", "Mild opacity."])
    assert diversity["unique_prediction_count"] == 2
    assert diversity["duplicate_prediction_count"] == 1
    interval = paired_bootstrap_delta([1.0, 2.0], [0.0, 1.0], samples=100)
    assert interval["mean_delta"] == 1.0
    assert interval["ci95_low"] == 1.0
