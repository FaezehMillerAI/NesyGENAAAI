"""Switchable drafting backends for no-training and trained configurations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from PIL import Image

from .schema import RetrievedStudy


class DraftingBackend(Protocol):
    def generate(
        self, image_path: str, indication: str, retrieved: list[RetrievedStudy]
    ) -> str: ...


class StaticBackend:
    """Supplies a fixed draft for deterministic tests, audits, and replay."""

    def __init__(self, report: str):
        self.report = report

    def generate(
        self, image_path: str, indication: str, retrieved: list[RetrievedStudy]
    ) -> str:
        del image_path, indication, retrieved
        return self.report


class RetrievalOnlyBackend:
    """Top-1 visual-neighbour report baseline; never accesses a test reference."""

    def generate(
        self, image_path: str, indication: str, retrieved: list[RetrievedStudy]
    ) -> str:
        del image_path, indication
        return retrieved[0].study.report if retrieved else ""


def clean_findings_output(text: str) -> str:
    """Remove common chat/section wrappers without changing clinical content."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(
        r"^(?:assistant\s*[:\-]?\s*)?(?:findings?\s*:\s*)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.split(r"\bimpression\s*:\s*", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    return cleaned.strip()


def medgemma_prompt(indication: str, retrieved: list[RetrievedStudy]) -> str:
    examples = ""
    if retrieved:
        blocks = [
            f"Training neighbour {index} (non-authoritative): {item.study.report}"
            for index, item in enumerate(retrieved, start=1)
        ]
        examples = "\nVisual retrieval evidence:\n" + "\n".join(blocks)
    return f"""You are drafting the Findings section of a chest radiograph report.
Clinical indication: {indication or 'Not provided.'}{examples}

Write only Findings. Preserve negation and laterality. Retrieved reports are
non-authoritative visual evidence: do not copy any finding that is unsupported by
the current image. Do not add an Impression heading or discuss this instruction."""


class MedGemmaBackend:
    """No task-specific fine-tuning backend using the gated MedGemma weights."""

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        max_new_tokens: int = 180,
        load_in_4bit: bool = True,
        use_retrieval: bool = True,
    ):
        try:
            import torch
            from transformers import BitsAndBytesConfig, pipeline
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise RuntimeError("Install adaptive-nesy-gen[models] for MedGemma") from exc
        model_kwargs: dict = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
        }
        if load_in_4bit and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device_map="auto",
            model_kwargs=model_kwargs,
        )
        self.max_new_tokens = max_new_tokens
        self.use_retrieval = use_retrieval

    def generate(
        self, image_path: str, indication: str, retrieved: list[RetrievedStudy]
    ) -> str:  # pragma: no cover - gated GPU path
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": medgemma_prompt(
                            indication, retrieved if self.use_retrieval else []
                        ),
                    },
                ],
            }
        ]
        output = self.pipe(text=messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            return clean_findings_output(str(generated[-1]["content"]))
        return clean_findings_output(str(generated))


def chexagent_prompt(indication: str, retrieved: list[RetrievedStudy]) -> str:
    evidence = ""
    if retrieved:
        evidence = "\n".join(
            f"Training neighbour {index} (non-authoritative): {item.study.report}"
            for index, item in enumerate(retrieved[:3], start=1)
        )
    return (
        "Write only the Findings section for this chest radiograph. Preserve negation "
        "and laterality. Do not copy findings unsupported by the current image.\n"
        f"Clinical indication: {indication or 'Not provided.'}\n{evidence}"
    ).strip()


def lightweight_prompt(indication: str, retrieved: list[RetrievedStudy]) -> str:
    """Compact decoder prefix used by the small trained vision-language model."""
    compact_indication = re.sub(r"\s+", " ", indication).strip()[:160]
    parts = [f"indication: {compact_indication or 'not provided'}"]
    for index, item in enumerate(retrieved[:2], start=1):
        compact_report = re.sub(r"\s+", " ", item.study.report).strip()[:240]
        parts.append(f"similar {index}: {compact_report}")
    parts.append("findings:")
    return "\n".join(parts)


class LightweightVisionLanguageBackend:
    """A small frozen-ViT/text-decoder model trained by the one-day workflow."""

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 128,
        min_new_tokens: int = 8,
        num_beams: int = 2,
        use_retrieval: bool = True,
    ):
        try:
            import json

            import torch
            from transformers import (
                AutoImageProcessor,
                AutoTokenizer,
                VisionEncoderDecoderModel,
            )
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise RuntimeError("Install adaptive-nesy-gen[lightweight]") from exc
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        config_path = Path(model_path) / "config.json"
        architecture = ""
        if config_path.exists():
            architecture = json.loads(config_path.read_text(encoding="utf-8")).get(
                "architecture", ""
            )
        self.architecture = architecture
        if architecture == "adaptive_nesy_gen_vit_flan_t5":
            from adaptive_nesy_gen.lightweight_vit_t5 import FrozenViTFlanT5ReportModel

            self.model = FrozenViTFlanT5ReportModel.from_pretrained(
                model_path, torch_dtype=dtype
            )
        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.num_beams = num_beams
        self.use_retrieval = use_retrieval

    def generate(
        self, image_path: str, indication: str, retrieved: list[RetrievedStudy]
    ) -> str:  # pragma: no cover - trained GPU path
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(
            self.device, dtype=next(self.model.parameters()).dtype
        )
        evidence = retrieved if self.use_retrieval else []
        prompt = lightweight_prompt(indication, evidence)
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=self.architecture == "adaptive_nesy_gen_vit_flan_t5",
            truncation=True,
            max_length=192,
        )
        input_ids = prompt_ids.input_ids.to(self.device)
        attention_mask = prompt_ids.attention_mask.to(self.device)
        if self.architecture == "adaptive_nesy_gen_vit_flan_t5":
            with self.torch.inference_mode():
                output = self.model.generate_reports(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=self.num_beams,
                    min_new_tokens=self.min_new_tokens,
                    max_new_tokens=self.max_new_tokens,
                    no_repeat_ngram_size=3,
                    length_penalty=0.9,
                    early_stopping=True,
                    use_cache=True,
                )[0]
            return clean_findings_output(
                self.tokenizer.decode(output, skip_special_tokens=True)
            )
        prompt_ids = input_ids
        start_id = int(self.model.config.decoder_start_token_id)
        if prompt_ids.shape[1] == 0 or int(prompt_ids[0, 0]) != start_id:
            start = self.torch.full(
                (prompt_ids.shape[0], 1), start_id, dtype=prompt_ids.dtype, device=self.device
            )
            prompt_ids = self.torch.cat([start, prompt_ids], dim=1)
        with self.torch.inference_mode():
            output = self.model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=prompt_ids,
                do_sample=False,
                num_beams=self.num_beams,
                min_new_tokens=self.min_new_tokens,
                max_new_tokens=self.max_new_tokens,
                no_repeat_ngram_size=3,
                length_penalty=0.9,
                early_stopping=True,
                use_cache=True,
            )[0]
        generated = output[prompt_ids.shape[1] :]
        return clean_findings_output(self.tokenizer.decode(generated, skip_special_tokens=True))


class CheXagentBackend:
    """Frozen-vision CheXagent with an optional locally trained QLoRA adapter."""

    def __init__(
        self,
        adapter: str | Path | None = None,
        model_id: str = "StanfordAIMI/CheXagent-2-3b",
        max_new_tokens: int = 160,
        min_new_tokens: int = 24,
        num_beams: int = 2,
        load_in_4bit: bool = True,
        use_retrieval: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - optional GPU dependency
            raise RuntimeError("Install adaptive-nesy-gen[chexagent] for CheXagent") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        quantization = None
        if load_in_4bit and torch.cuda.is_available():
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["model.visual"],
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load a CheXagent adapter") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.num_beams = num_beams
        self.use_retrieval = use_retrieval

    def generate(
        self, image_path: str, indication: str, retrieved: list[RetrievedStudy]
    ) -> str:  # pragma: no cover - trained GPU path
        evidence = retrieved if self.use_retrieval else []
        input_ids = None
        for keep in range(min(3, len(evidence)), -1, -1):
            query = self.tokenizer.from_list_format(
                [
                    {"image": image_path},
                    {"text": chexagent_prompt(indication, evidence[:keep])},
                ]
            )
            conversation = [
                {"from": "system", "value": "You are a careful radiology report assistant."},
                {"from": "human", "value": query},
            ]
            candidate = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors="pt"
            )
            if candidate.shape[-1] <= 2048 - self.max_new_tokens:
                input_ids = candidate.to(next(self.model.parameters()).device)
                break
        if input_ids is None:
            raise ValueError("CheXagent prompt exceeds its context even without retrieval evidence")
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids,
                do_sample=False,
                num_beams=self.num_beams,
                min_new_tokens=self.min_new_tokens,
                use_cache=True,
                max_new_tokens=self.max_new_tokens,
            )[0]
        decoded = self.tokenizer.decode(
            output[input_ids.size(1) :], skip_special_tokens=True
        )
        return clean_findings_output(decoded)
