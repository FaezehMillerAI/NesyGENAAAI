from __future__ import annotations

from pathlib import Path

import numpy as np

from adaptive_nesy_gen.cli import run_demo
from adaptive_nesy_gen.gate import eligible_replacement
from adaptive_nesy_gen.knowledge import KnowledgeGraph
from adaptive_nesy_gen.retrieval import (
    PixelHistogramEncoder,
    VisualIndex,
    load_manifest,
)
from adaptive_nesy_gen.schema import Claim
from adaptive_nesy_gen.text import DeterministicLinker, LexiconEntry
from adaptive_nesy_gen.verification import entity_specific_grounding

ROOT = Path(__file__).parents[1]
DEMO = ROOT / "data" / "demo"


def test_manifest_and_index_are_train_only_and_study_unique():
    studies = load_manifest(DEMO / "manifest.csv")
    index, _ = VisualIndex.build(studies, PixelHistogramEncoder())
    assert len(index.studies) == 3
    assert {study.split for study in index.studies} == {"train"}
    query = next(study for study in studies if study.split == "test")
    results = index.query(query.image_path, 3, exclude_study_id=query.study_id)
    assert len({item.study.study_id for item in results}) == len(results)
    assert index.query(query.image_path, 0) == []


def test_index_round_trip(tmp_path):
    studies = load_manifest(DEMO / "manifest.csv")
    encoder = PixelHistogramEncoder()
    index, _ = VisualIndex.build(studies, encoder)
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = VisualIndex.load(path, encoder)
    np.testing.assert_allclose(index.embeddings, loaded.embeddings)
    assert [row.study_id for row in loaded.studies] == [row.study_id for row in index.studies]


def test_linker_preserves_negation():
    linker = DeterministicLinker.from_json(DEMO / "lexicon.json")
    claims = linker.claims("No pleural effusion. Mild opacity.")
    assert claims[0].entities[0].negated is True
    assert claims[1].entities[0].negated is False


def test_grounding_is_entity_and_polarity_specific():
    studies = load_manifest(DEMO / "manifest.csv")
    index, _ = VisualIndex.build(studies, PixelHistogramEncoder())
    query = next(study for study in studies if study.split == "test")
    retrieved = index.query(query.image_path, 3)
    linker = DeterministicLinker.from_json(DEMO / "lexicon.json")
    claim = linker.claims("No pleural effusion.")[0]
    grounding = entity_specific_grounding(claim, retrieved, linker)
    assert grounding.support_count == 2


def test_revision_requires_exact_entity_and_polarity():
    studies = load_manifest(DEMO / "manifest.csv")
    index, _ = VisualIndex.build(studies, PixelHistogramEncoder())
    query = next(study for study in studies if study.split == "test")
    retrieved = index.query(query.image_path, 3)
    linker = DeterministicLinker.from_json(DEMO / "lexicon.json")
    positive = linker.claims("Right pleural effusion.")[0]
    replacement = eligible_replacement(positive, retrieved, linker, threshold=-1.0)
    assert replacement is not None
    assert "Small right pleural effusion" in replacement.text
    negative = linker.claims("There is no pleural effusion.")[0]
    negative_replacement = eligible_replacement(negative, retrieved, linker, threshold=-1.0)
    assert negative_replacement is not None
    assert negative_replacement.study_id != "s003"


def test_unlinked_claim_cannot_be_treated_as_verified():
    linker = DeterministicLinker(
        [LexiconEntry("RAD:x", "opacity", "finding", tuple())]
    )
    claim = Claim(
        "c1",
        "Support devices are unchanged.",
        linker.link("Support devices are unchanged."),
    )
    assert not claim.linked


def test_demo_emits_complete_procedural_trace():
    payload = run_demo()
    assert payload["graph_calls"] >= 1
    assert payload["escalation_rate_all"] <= 1.0
    assert {trace["gate_decision"] for trace in payload["traces"]} <= {
        "ACCEPT",
        "REVISE",
        "FLAG",
        "ABSTAIN",
    }
    required = {
        "original_claim",
        "final_claim",
        "entities",
        "evidence",
        "graph_triggered",
        "graph_provenance",
        "gate_reason",
        "latency_ms",
    }
    assert all(required <= trace.keys() for trace in payload["traces"])


def test_primekg_raw_column_aliases_are_supported(tmp_path):
    path = tmp_path / "kg.csv"
    path.write_text(
        "x_id,y_id,display_relation,x_type,y_type,x_name,y_name\n"
        "D:1,A:1,located_in,disease,anatomy,Opacity,Lung\n",
        encoding="utf-8",
    )
    graph = KnowledgeGraph.from_csv(path)
    assert graph.has_node("D:1")
    assert graph.edges[0].relation == "located_in"
