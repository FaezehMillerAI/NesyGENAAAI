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
