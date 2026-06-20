from __future__ import annotations

from scripts.evaluate_publication import (
    official_chexbert_metrics,
    official_coco_metrics,
    official_radgraph_metrics,
)


class _Bleu:
    def compute_score(self, references, hypotheses):
        size = len(references)
        return [0.1, 0.2, 0.3, 0.4], [[value] * size for value in (0.1, 0.2, 0.3, 0.4)]


class _ScalarMetric:
    def __init__(self, value):
        self.value = value

    def compute_score(self, references, hypotheses):
        return self.value, [self.value] * len(references)


class _Chexbert:
    def __call__(self, hyps, refs):
        report = {
            "micro avg": {"f1-score": 0.8},
            "macro avg": {"f1-score": 0.7},
        }
        return 0.75, [1.0, 0.5], report, report


class _Radgraph:
    def __call__(self, hyps, refs):
        # The official package may return metric-major rewards; the adapter transposes them.
        rewards = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        return (0.15, 0.35, 0.55), rewards, {}, {}


def test_official_metric_adapters_preserve_corpus_and_per_study_values():
    hypotheses = ["No edema.", "Mild opacity."]
    references = ["No edema.", "Opacity."]
    scorers = {
        "bleu": _Bleu(),
        "rouge_l": _ScalarMetric(0.5),
        "meteor": _ScalarMetric(0.6),
        "cider": _ScalarMetric(0.7),
    }
    coco, coco_rows = official_coco_metrics(hypotheses, references, scorers)
    assert coco["bleu_4"] == 0.4
    assert coco_rows[1]["cider"] == 0.7

    chexbert, chexbert_rows = official_chexbert_metrics(
        hypotheses, references, _Chexbert()
    )
    assert chexbert["chexbert_all_micro_f1"] == 0.8
    assert chexbert_rows == [1.0, 0.5]

    radgraph, radgraph_rows = official_radgraph_metrics(
        hypotheses, references, _Radgraph()
    )
    assert radgraph["radgraph_entity_relation"] == 0.35
    assert radgraph_rows[1]["radgraph_complete"] == 0.6
