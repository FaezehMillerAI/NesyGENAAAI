from adaptive_nesy_gen.evaluation import bleu_n, leakage_audit, rouge_l_f1


def test_lexical_smoke_metrics_are_bounded():
    prediction = "No pleural effusion."
    reference = "No pleural effusion is seen."
    assert 0.0 <= bleu_n(prediction, reference, 4) <= 1.0
    assert 0.0 <= rouge_l_f1(prediction, reference) <= 1.0


def test_leakage_audit_counts_exact_matches():
    result = leakage_audit(["No edema.", "No effusion."], ["No edema."])
    assert result["exact_match_count"] == 1
    assert result["unique_prediction_ratio"] == 1.0
