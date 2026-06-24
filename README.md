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

If Colab reports that `adaptive_nesy_gen.retrieval` is missing, rerun the repository
bootstrap cell immediately above the path/configuration cell. It installs with the active
kernel interpreter, places this repository's `src/` first on `sys.path`, and prints the
resolved package file so a stale package collision is visible.

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

Cell 3 accepts both sample-major `(N, D)` and legacy transposed `(D, N)` MedSigLIP
caches. If neither axis equals the manifest's training count, the cache is partial; the
notebook rebuilds all training embeddings with batch progress, adaptive CUDA-OOM batch
reduction, and atomic replacement so a failed rebuild does not destroy the old file. The
model is gated: first accept the Health AI terms at
`https://huggingface.co/google/medsiglip-448` and authenticate with a read token. The same
operation can then be run directly:

The browser approval and the Colab credential are separate. If Hugging Face returns
401/403, the notebook uses a blocking hidden prompt for a fresh read token and explicitly
overrides any stale cached credential. This avoids Colab continuing before Hugging Face's
non-blocking login widget has received a token.

```bash
python -m pip install -e '.[medsiglip]'
python scripts/build_medsiglip_index.py \
  --manifest /path/to/manifest.jsonl \
  --output /content/drive/MyDrive/medsiglip_cache_iuxray_official/train_index.npz
```

The MedGemma generation backend is gated separately. Before Cell 6, accept the terms at
`https://huggingface.co/google/medgemma-4b-it` with the same Hugging Face account. Cell 6
checks access before loading the model and writes a persistent `*.error.log` beside its
JSONL output if the generation subprocess fails.

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
extra, which pins the matching `peft==0.10.0` rather than resolving a newer incompatible
PEFT release. The script quantizes the decoder with NF4, limits LoRA to decoder modules, keeps
the complete visual encoder frozen in BF16, enables gradient checkpointing, uses batch
size one with accumulation, and saves adapter-only checkpoints.

CheXagent expands one image into a fixed 1,026-token span, so training uses its full
2,048-token context rather than truncating at 768. Retrieval neighbours are removed from
the lowest rank only when needed to preserve the complete image span and at least 64
supervised report tokens. Preprocessing asserts balanced image markers and a non-empty
`gpt` target before the first forward pass.

The Colab BLEU-oriented default uses the trained CheXagent adapter, two-beam deterministic
decoding, a 24-token minimum generation budget, retrieval-conditioned prompting, and
section-wrapper cleanup. On GPUs with at least 40 GiB, QLoRA automatically uses batch
size 4 with four-step accumulation (the same effective batch 16) to reduce wall time.
Checkpoints resume automatically, and the saved adapter is selected by lowest official
validation loss rather than final-step position. These choices are leakage-safe optimizations, not a
guarantee of any metric threshold; Cell 8 reports whether official BLEU-1 reaches 0.50.

BF16 is checked before the model download. A T4 is not sufficient for this stated
precision policy; select an A100, L4, or another BF16-capable GPU. Cell 5 records a full
traceback in `training_error.log` beside the adapter if its subprocess fails.

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

## Recommended one-day lightweight run

For a deadline-constrained run, use
[`Lightweight_Adaptive_NeSy_Gen_One_Day_Colab.ipynb`](notebooks/Lightweight_Adaptive_NeSy_Gen_One_Day_Colab.ipynb).
Its default drafter is `facebook/deit-tiny-patch16-224` plus
`google/flan-t5-small`: the DeiT vision transformer is frozen, and the default
training scope updates only the T5 decoder/language head plus a small visual
projection. It trains at 224×224, uses cached MedSigLIP neighbours in 70% of
training prompts, avoids laterality-breaking flips, resumes checkpoints, and
generates the official test split only once. The Colab notebook copies Drive-backed
images to `/content` SSD and disables CPU augmentation by default; this avoids the
30-second dataloader stalls that appear when PIL repeatedly reads images from
Google Drive. All graph/LTN/gate ablations then replay those drafts without
reloading the model.

```bash
python -m pip install -e '.[lightweight]'
python scripts/train_lightweight_vlm.py \
  --manifest /path/to/manifest.jsonl \
  --medsiglip-cache /path/to/train_index.npz \
  --output-dir /content/drive/MyDrive/aaai_2026_experiments/lightweight/iuxray \
  --image-cache-dir /content/adaptive_nesy_gen_image_cache/iuxray \
  --no-augmentation \
  --max-steps 1500

python scripts/run_experiments.py \
  --manifest /path/to/manifest.jsonl \
  --medsiglip-cache /path/to/train_index.npz \
  --primekg-cache /path/to/primekg_cache \
  --backend lightweight --drafting-mode few-shot \
  --model-path /content/drive/MyDrive/aaai_2026_experiments/lightweight/iuxray/best_model \
  --output /content/drive/MyDrive/aaai_2026_experiments/lightweight/iuxray/test.jsonl
```

Start with IU-Xray for rapid iteration, then transfer the chosen configuration to
MIMIC-CXR. BLEU-1 ≥0.50 is recorded as a validation/test target, not guaranteed: the
notebook always reports the measured score and keeps retrieval-only and neural-only
baselines visible to prevent metric cherry-picking.

## RadReport-VL drafter + Adaptive NeSy-Gen verifier

To use [`VivaanGupta17/radreport-vl`](https://github.com/VivaanGupta17/radreport-vl)
as the starting report-generation model without changing its vision encoder,
cross-attention bridge, decoder, or classifier, open
[`RadReport_VL_Adaptive_NeSy_Gen_Colab.ipynb`](notebooks/RadReport_VL_Adaptive_NeSy_Gen_Colab.ipynb).

The notebook installs two adapter files into a cloned RadReport-VL checkout and
applies one tokenizer-compatibility patch:

- a manifest-backed `src.data.mimic_cxr_dataset` module, because the upstream
  repository imports that module but does not currently ship it;
- a draft exporter that loads a RadReport-VL checkpoint and writes reference-free
  test drafts in the Adaptive NeSy-Gen replay contract.
- a training-script patch that resizes only the decoder token embedding/output
  tables after RadReport-VL extends its tokenizer, then saves the tokenizer beside
  checkpoints. The ViT encoder, bridge layers, decoder stack, and classifier are
  otherwise unchanged.

After RadReport-VL drafts are exported, the existing Adaptive NeSy-Gen pipeline
replays them through retrieval grounding, PrimeKG path validation, LTN constraint
scores, gate/revision decisions, graph-control ablations, integrity audits,
official text/clinical metrics, bootstrap intervals, and blinded expert-review
packet generation. This keeps the modeling contribution separable: RadReport-VL is
the drafter, while Adaptive NeSy-Gen supplies the proposal's reasoning and
verification layer.

Do not use the older manual `%%writefile src/data/mimic_cxr_dataset.py` smoke-test
cell for this workflow. That minimal loader only supports simple CSV folders and can
fall back to synthetic random images when given a manifest path. Always run
`scripts/prepare_radreport_vl_adapters.py` from this repository instead so RadReport-VL
trains on the selected AAAI manifest and exports replay-compatible test drafts.

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

The Colab publication stage runs Microsoft COCO BLEU-1–4, ROUGE-L, METEOR and CIDEr,
plus the official F1CheXbert and F1RadGraph packages. It also reports retrieval,
generation, claim-verification and end-to-end timing distributions; cached-index
initialization/load timing and offline-build metadata; graph calls/report; escalation over
all and linked claims; peak GPU memory;
index size; exact/high-overlap leakage; same-study and train-only retrieval integrity;
prediction frequency/diversity; and a separate structural linker audit.

Generation loads the manifest with every test report redacted, and inference JSONL files
contain no `reference` field. A stable per-image `example_id` preserves alternate views
that share a `study_id`. The separate post-inference evaluator then loads and aligns
references, rejects incomplete or duplicate official-test outputs, computes 2,000-sample
paired bootstrap intervals with Holm correction, and writes:

- `publication_metrics.json` and `per_study_metrics.jsonl`;
- a blinded `expert_review_packet.csv`, protocol, and separate blinding key;
- `linker_expert_review_packet.csv` for mention/link/negation adjudication.

Human expert review cannot be executed by software. The evaluator marks the run
`PENDING_HUMAN_REVIEW` and blocks a hallucination-reduction claim until the blinded
reports and linker annotations have been completed and adjudicated.

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
