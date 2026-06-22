from __future__ import annotations

from pathlib import Path

import numpy as np

from adaptive_nesy_gen.backends import clean_findings_output
from adaptive_nesy_gen.cli import run_demo
from adaptive_nesy_gen.gate import eligible_replacement
from adaptive_nesy_gen.knowledge import KnowledgeGraph
from adaptive_nesy_gen.primekg_io import find_primekg_layout
from adaptive_nesy_gen.retrieval import (
    PixelHistogramEncoder,
    VisualIndex,
    load_manifest,
    manifest_example_id,
)
from adaptive_nesy_gen.schema import Claim
from adaptive_nesy_gen.text import DeterministicLinker, LexiconEntry
from adaptive_nesy_gen.verification import entity_specific_grounding
from scripts.build_radiology_primekg import build_cache

ROOT = Path(__file__).parents[1]
DEMO = ROOT / "data" / "demo"


def test_findings_cleanup_removes_wrappers_but_keeps_content():
    text = "Assistant: FINDINGS:  No focal opacity.\nIMPRESSION: Normal chest."
    assert clean_findings_output(text) == "No focal opacity."


def test_manifest_and_index_are_train_only_and_study_unique():
    studies = load_manifest(DEMO / "manifest.csv")
    index, _ = VisualIndex.build(studies, PixelHistogramEncoder())
    assert len(index.studies) == 3
    assert {study.split for study in index.studies} == {"train"}
    query = next(study for study in studies if study.split == "test")
    results = index.query(query.image_path, 3, exclude_study_id=query.study_id)
    assert len({item.study.study_id for item in results}) == len(results)
    assert index.query(query.image_path, 0) == []


def test_visual_query_is_cached_for_replayed_ablations():
    class CountingEncoder(PixelHistogramEncoder):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def encode(self, image_paths):
            self.calls += 1
            return super().encode(image_paths)

    studies = load_manifest(DEMO / "manifest.csv")
    encoder = CountingEncoder()
    index, _ = VisualIndex.build(studies, encoder)
    query = next(study for study in studies if study.split == "test")
    calls_after_build = encoder.calls
    first = index.query(query.image_path, 3, exclude_study_id=query.study_id)
    second = index.query(query.image_path, 3, exclude_study_id=query.study_id)
    assert encoder.calls == calls_after_build + 1
    assert first == second


def test_manifest_can_redact_test_references_at_ingestion():
    studies = load_manifest(DEMO / "manifest.csv", redact_splits={"test"})
    assert all(study.report for study in studies if study.split == "train")
    assert all(not study.report for study in studies if study.split == "test")


def test_manifest_example_id_keeps_alternate_views_distinct():
    study = load_manifest(DEMO / "manifest.csv")[0]
    alternate = type(study)(**{**study.__dict__, "image_path": study.image_path + ".other"})
    assert manifest_example_id(study) != manifest_example_id(alternate)
    assert manifest_example_id(study).startswith(f"{study.study_id}::")


def test_index_round_trip(tmp_path):
    studies = load_manifest(DEMO / "manifest.csv")
    encoder = PixelHistogramEncoder()
    index, _ = VisualIndex.build(studies, encoder)
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = VisualIndex.load(path, encoder)
    np.testing.assert_allclose(index.embeddings, loaded.embeddings)
    assert [row.study_id for row in loaded.studies] == [row.study_id for row in index.studies]


def test_index_load_accepts_transposed_embedding_cache(tmp_path):
    studies = load_manifest(DEMO / "manifest.csv")
    encoder = PixelHistogramEncoder()
    index, _ = VisualIndex.build(studies, encoder)
    path = tmp_path / "transposed.npz"
    np.savez_compressed(path, embeddings=index.embeddings.T)
    loaded = VisualIndex.load(path, encoder, studies=studies)
    np.testing.assert_allclose(index.embeddings, loaded.embeddings)


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
    assert graph.edges[0].display_relation == "located_in"
    assert graph.edges[0].confidence == 1.0


def test_indexed_primekg_layout_and_training_only_cache_builder(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "nodes.csv").write_text(
        "node_index,node_id,node_type,node_name,node_source\n"
        "1,D:1,phenotype,Opacity,radlex\n"
        "2,A:1,anatomy,Lung,uberon\n"
        "3,RX:1,drug,Aspirin,drugbank\n",
        encoding="utf-8",
    )
    (raw / "edges.csv").write_text(
        "x_index,y_index,display_relation,confidence,source\n"
        "1,2,located in,0.9,primekg-test\n",
        encoding="utf-8",
    )
    image = DEMO / "images" / "query.pgm"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "{\"study_id\":\"s1\",\"image_path\":\""
        + str(image)
        + "\",\"indication\":\"\",\"report\":\"Opacity.\",\"split\":\"train\",\"metadata\":{}}\n"
        "{\"study_id\":\"s2\",\"image_path\":\""
        + str(image)
        + "\",\"indication\":\"\",\"report\":\"Aspirin.\",\"split\":\"test\",\"metadata\":{}}\n",
        encoding="utf-8",
    )
    output = tmp_path / "cache"
    summary = build_cache(raw, manifest, output, hops=1, seed_split="train")
    assert summary["manifest_examples_scanned"] == 1
    assert summary["seed_nodes"] == 1
    assert summary["subgraph_nodes"] == 2
    assert (output / "radiology_primekg_summary.json").exists()
    cached = KnowledgeGraph.from_cache(output)
    assert cached.metadata["seed_split"] == "train"
    assert cached.node_names["D:1"] == "Opacity"
    assert cached.edges[0].relation == "located in"
    assert cached.edges[0].display_relation == "located in"
    assert cached.edges[0].confidence == 0.9
    assert cached.edges[0].edge_source == "primekg-test"
    assert "RX:1" not in cached.node_names


def test_raw_primekg_complete_file_priority(tmp_path):
    header = "x_id,x_name,x_type,y_id,y_name,y_type,relation\n"
    row = "D:1,Opacity,phenotype,A:1,Lung,anatomy,located_in\n"
    (tmp_path / "kg_grouped.csv").write_text(header + row, encoding="utf-8")
    (tmp_path / "kg_raw.csv").write_text(header + row, encoding="utf-8")
    layout = find_primekg_layout(tmp_path)
    assert layout.edge_path.name == "kg_raw.csv"
