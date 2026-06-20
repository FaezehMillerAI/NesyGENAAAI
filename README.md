# Adaptive NeSy-Gen

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FaezehMillerAI/NesyGENAAAI/blob/main/notebooks/Adaptive_NeSy_Gen_Experiments_Colab.ipynb)

Research code for **adaptive, claim-level neuro-symbolic verification with
inference-faithful evidence traces and selective intervention** in chest X-ray
report generation.

The repository implements the proposed method without claiming that it eliminates
hallucinations or establishes state-of-the-art performance. LTN values are
constraint-satisfaction scores, not probabilities that a clinical statement is true.

## What is implemented

1. **Leakage-safe visual retrieval.** Frozen MedSigLIP embeddings are cached once.
   The index accepts only training rows, excludes the query study and alternate views,
   deduplicates studies, and returns top-k training reports as non-authoritative evidence.
2. **Switchable drafting.** Use MedGemma with no task-specific fine-tuning, or a
   frozen-vision CheXagent backend trained with decoder-scoped QLoRA. A static replay
   backend makes verification experiments deterministic.
3. **Claim extraction and linking.** A reproducible longest-match linker records
   mentions, normalized IDs, types, negation, and confidence. It is deliberately
   replaceable by RadGraph-assisted extraction and must be evaluated independently.
4. **Claim-specific grounding.** Retrieval support requires matching entity IDs and
   polarity; a report-level image match alone never labels every claim as grounded.
5. **Adaptive routing.** Claims take the fast path only when `s_ground >= 0.85` and at
   least two distinct studies support the same entity/polarity. Unlinked claims abstain.
6. **PrimeKG/LTN verification.** Uncertain linked claims use a compact, provenance-aware
   graph and fuzzy biological, diagnostic, and location constraints.
7. **Consistency Gate.** Every claim receives `ACCEPT`, `REVISE`, `FLAG`, or `ABSTAIN`,
   with confidence and an explicit decision reason.
8. **Evidence-bounded revision.** A replacement must come from a retrieved training
   study, exceed the threshold, have exactly the same entity IDs and polarity, and add
   no clinical entity. The complete report is never regenerated.
9. **Faithful traces.** JSON outputs record the evidence and decisions actually consumed
   during inference, graph paths, provenance, clause scores, and latency.

## Quick CPU smoke run

```bash
python -m pip install -e '.[dev]'
adaptive-nesy-gen demo --output outputs/demo_trace.json
pytest -q
```

The bundled PGM images and `PrimeKG-demo` graph are synthetic pipeline fixtures only.
They are not clinical data and produce no research result.

## Your existing IU-Xray and MIMIC-CXR artifacts

The Colab notebook is preconfigured for these paths:

| Artifact | IU-Xray | MIMIC-CXR |
|---|---|---|
| Manifest | `/content/drive/MyDrive/aaai_2026_experiments/iuxray_official/aaai_vision_t5_base_convnext_primekg/iuxray_official_manifest.jsonl` | `/content/drive/MyDrive/aaai_2026_experiments/mimic_aug/aaai_vision_t5_base_convnext_primekg/mimic_aug_manifest.jsonl` |
| PrimeKG cache | `/content/drive/MyDrive/primekg_radiology_cache_iuxray_official/` | `/content/drive/MyDrive/primekg_radiology_cache_mimic_aug/` |
| MedSigLIP index | `/content/drive/MyDrive/medsiglip_cache_iuxray_official/train_index.npz` | `/content/drive/MyDrive/medsiglip_cache_mimic_aug/train_index.npz` |

Manifests may be CSV or JSONL and use the common fields `study_id`, `image_path`,
`indication`, `report`, `split`, and `metadata`. Relative paths resolve from the manifest
directory; your absolute Drive image paths work directly. Only `train` rows are admitted
to `VisualIndex`. QLoRA consumes `train` and `val`; the script asserts that zero `test`
examples are consumed.

Rebuild the manifests only if needed:

```bash
python scripts/build_manifests.py iu \
  --annotation /content/drive/MyDrive/iuxray/annotation.json \
  --images-root /content/drive/MyDrive/iuxray/images \
  --output /content/drive/MyDrive/.../iuxray_official_manifest.jsonl

python scripts/build_manifests.py mimic \
  --train-csv /content/drive/MyDrive/mimic_cxr_dataset/versions/2/mimic_cxr_aug_train.csv \
  --validate-csv /content/drive/MyDrive/mimic_cxr_dataset/versions/2/mimic_cxr_aug_validate.csv \
  --dataset-root /content/drive/MyDrive/mimic_cxr_dataset/versions/2 \
  --output /content/drive/MyDrive/.../mimic_aug_manifest.jsonl
```

The IU builder preserves official splits, removes missing images/empty reports and
`XXXX`, extracts Findings when section headings exist, and retains each view as a
separate example sharing the study ID. The MIMIC builder selects `PA → AP → image →
Lateral`, ignores `text_augment`, extracts the study component, and splits validation
subjects into val/test with seed 13.

## Efficient CheXagent QLoRA

CheXagent's released remote code requires `transformers==4.40.0`; use its dedicated
extra. The script quantizes the decoder with NF4, limits LoRA to decoder modules, keeps
the complete visual encoder frozen in BF16, enables gradient checkpointing, uses batch
size one with accumulation, and saves adapter-only checkpoints.

```bash
python -m pip install -e '.[chexagent]'
python scripts/train_chexagent_qlora.py \
  --manifest /content/drive/MyDrive/.../iuxray_official_manifest.jsonl \
  --medsiglip-cache /content/drive/MyDrive/medsiglip_cache_iuxray_official/train_index.npz \
  --output-dir /content/drive/MyDrive/aaai_2026_experiments/adaptive_nesy_gen/iu_chexagent \
  --max-steps 500
```

The cached MedSigLIP embeddings are converted once into a reusable HNSW neighbour table.
Up to five unique training studies are found; the first three reports are inserted into
50% of training prompts. The graph and test reports never enter QLoRA.

CheXagent QLoRA and MedGemma cannot share one Python process because of their incompatible
Transformers requirements. Run the corresponding notebook sections in separate Colab
runtimes; both write the same JSONL experiment contract to Drive.

## MedGemma experiments

```bash
python -m pip install -e '.[medgemma]'
```

Accept the gated model terms and authenticate with Hugging Face before loading
`google/medgemma-4b-it`. For MIMIC-CXR, report this condition as **no task-specific
fine-tuning**, not strict unseen-data zero-shot, because documented pretraining includes
MIMIC-CXR. The backend supports query-only prompts and retrieval-conditioned prompts.

## PrimeKG cache contract

The notebook maps `RUN_DATASET="iuxray_official"` and `RUN_DATASET="mimic_aug"` to:

```text
/content/drive/MyDrive/primekg_radiology_cache_iuxray_official/
/content/drive/MyDrive/primekg_radiology_cache_mimic_aug/
```

Each cache contains `kg.csv`, `nodes.csv`, and `radiology_primekg_summary.json`.
`KnowledgeGraph.from_cache()` preserves the stable IDs, machine and display relations,
node names/types, edge confidence, and source provenance used in explanation traces.

The raw loader searches `/content/drive/MyDrive/dataverse_files/` in this order:
`kg.csv`, `kg_giant.csv`, `kg_raw.csv`, then `kg_grouped.csv`. It accepts complete
`x_id`/`y_id` edge tables or joins an index-based `edges.csv` against `nodes.csv` when a
complete table is unavailable. Missing confidence defaults to `1.0`; missing source
defaults to `primekg`; either relation field is copied to the other when absent.

Build a frozen, training-only one-hop subset with:

```bash
python scripts/build_radiology_primekg.py \
  --primekg-dir /content/drive/MyDrive/dataverse_files \
  --manifest /path/to/iuxray_manifest.jsonl \
  --output-dir /content/drive/MyDrive/primekg_radiology_cache_iuxray_official \
  --hops 1 \
  --seed-split train
```

The builder scans only `indication + report` from the selected split, performs exact
normalized node-name matching, streams the large edge file during hop expansion, writes
`node_name` to an `alias` column, and records source paths, seed policy, split, hop count,
and graph statistics in the summary. PrimeKG never updates CheXagent/LLM parameters.

## Experiment matrix

`adaptive_nesy_gen.experiments.ablation_configs()` provides:

- RAG without PrimeKG/LTN;
- adaptive audit without revision;
- the full adaptive method;
- always-on claim verification;
- adaptive verification without LTN;
- adaptive verification without the Consistency Gate;
- strict and permissive fast-path thresholds.

The notebook also lays out the drafting-only baselines (retrieval-only, raw trained
generator, MedGemma query-only/few-shot), report-level verification, shuffled/relation-
ablated graph controls, and threshold sweeps. Persist the raw generation once, then replay
it across verifier ablations to avoid repeated GPU generation and paired-comparison noise.

## Evaluation

The lightweight evaluator provides BLEU-1–4, ROUGE-L, entity/polarity precision-recall-F1,
explanation coverage, routing/gate distributions, revision rate, latency, exact-match
leakage, high-overlap leakage, and prediction diversity. These are smoke/reproducibility
metrics, not substitutes for the official clinical evaluators.

Publication runs should additionally use the official implementations/checkpoints for:

- METEOR and CIDEr;
- CheXbert/CheXpert label metrics;
- RadGraph F1;
- entity-linking mention precision, linking accuracy, negation accuracy, and coverage;
- paired bootstrap confidence intervals and appropriate multiple-comparison correction;
- expert review before making a hallucination-reduction claim.

Always audit same-study exclusion and confirm that no test reference is accessible during
retrieval, generation, selection, verification, or revision.

## Complexity

With cached normalized embeddings, exact retrieval is `O(Nd)` per query and `O(Nd)`
storage for `N` training images of width `d`; HNSW is used for scalable training-neighbour
precomputation. Deterministic linking is linear in report length times the compact lexicon.
Fast-path claims incur no graph call. An escalated claim explores a bounded one-hop
subgraph plus paths up to three hops, so adaptive verification cost is proportional to the
number of uncertain linked claims rather than all report claims.

## Limitations

- Deterministic linking trades coverage for reproducibility and can miss synonyms,
  coordination, uncertain assertions, and long negation scopes.
- PrimeKG connectivity establishes compatibility with implemented graph constraints, not
  image-level truth, and its coverage/provenance can be incomplete.
- Visual neighbours can share dataset artifacts rather than pathology.
- Procedural traces are inference-faithful records, not complete causal explanations of
  the neural generator.
- Selective revision is intentionally conservative; disputed claims remain flagged when
  an exact evidence-bounded replacement is unavailable.
- This research software is not a medical device and must not guide patient care.

## Layout

```text
src/adaptive_nesy_gen/   method implementation
scripts/                 manifest builders and efficient QLoRA
notebooks/               Colab experiment driver
data/demo/               synthetic smoke fixture
tests/                   integrity and method-contract tests
```

## License

Apache-2.0. Model weights and clinical datasets retain their own licenses and access terms.
