"""Frozen ViT + lightweight T5-style report generator."""

from __future__ import annotations

import json
from pathlib import Path


class FrozenViTFlanT5ReportModel:
    """Lazy factory namespace for the optional torch/transformers implementation."""

    @staticmethod
    def create(
        vision_encoder,
        text_model,
        visual_tokens: int = 32,
        train_text_encoder: bool = False,
        train_embeddings: bool = False,
    ):
        import torch
        from torch import nn
        from torch.nn import functional as F
        from transformers.modeling_outputs import BaseModelOutput

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = vision_encoder
                self.text_model = text_model
                self.config = text_model.config
                self.generation_config = getattr(text_model, "generation_config", None)
                self.visual_tokens = int(visual_tokens)
                self.train_text_encoder = bool(train_text_encoder)
                self.train_embeddings = bool(train_embeddings)
                vision_hidden = int(vision_encoder.config.hidden_size)
                text_hidden = int(text_model.config.d_model)
                self.visual_projection = nn.Sequential(
                    nn.LayerNorm(vision_hidden),
                    nn.Linear(vision_hidden, text_hidden),
                )
                self._freeze_for_decoder_training()

            def _freeze_for_decoder_training(self) -> None:
                for parameter in self.vision_encoder.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.text_model.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.text_model.decoder.parameters():
                    parameter.requires_grad_(True)
                for parameter in self.text_model.lm_head.parameters():
                    parameter.requires_grad_(True)
                if self.train_text_encoder:
                    for parameter in self.text_model.encoder.parameters():
                        parameter.requires_grad_(True)
                if self.train_embeddings:
                    for parameter in self.text_model.get_input_embeddings().parameters():
                        parameter.requires_grad_(True)

            def _visual_embeds(self, pixel_values):
                self.vision_encoder.eval()
                with torch.no_grad():
                    hidden = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
                    hidden = hidden.last_hidden_state
                if hidden.shape[1] != self.visual_tokens:
                    hidden = F.adaptive_avg_pool1d(
                        hidden.transpose(1, 2), self.visual_tokens
                    ).transpose(1, 2)
                return self.visual_projection(hidden)

            def encode_context(self, pixel_values, input_ids, attention_mask):
                visual_embeds = self._visual_embeds(pixel_values)
                text_embeds = self.text_model.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
                visual_mask = torch.ones(
                    visual_embeds.shape[:2],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
                encoder_outputs = self.text_model.encoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=combined_mask,
                    return_dict=True,
                )
                return encoder_outputs, combined_mask

            def forward(self, pixel_values, input_ids, attention_mask, labels=None, **_):
                if labels is None:
                    raise ValueError("FrozenViTFlanT5ReportModel.forward requires labels")
                encoder_outputs, combined_mask = self.encode_context(
                    pixel_values, input_ids, attention_mask
                )
                decoder_input_ids = self.text_model._shift_right(labels)
                return self.text_model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=combined_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    return_dict=True,
                )

            @torch.inference_mode()
            def generate_reports(self, pixel_values, input_ids, attention_mask, **kwargs):
                encoder_outputs, combined_mask = self.encode_context(
                    pixel_values, input_ids, attention_mask
                )
                return self.text_model.generate(
                    encoder_outputs=BaseModelOutput(
                        last_hidden_state=encoder_outputs.last_hidden_state
                    ),
                    attention_mask=combined_mask,
                    **kwargs,
                )

            def save_for_generation(
                self,
                output_dir: str | Path,
                *,
                encoder_id: str,
                decoder_id: str,
                tokenizer,
                processor,
            ) -> None:
                from safetensors.torch import save_file

                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                self.vision_encoder.save_pretrained(output_path / "vision_encoder")
                self.text_model.save_pretrained(output_path / "text_model")
                tokenizer.save_pretrained(output_path)
                processor.save_pretrained(output_path)
                save_file(
                    self.visual_projection.state_dict(),
                    output_path / "projection.safetensors",
                )
                config = {
                    "architecture": "adaptive_nesy_gen_vit_flan_t5",
                    "encoder": encoder_id,
                    "decoder": decoder_id,
                    "visual_tokens": self.visual_tokens,
                    "train_text_encoder": self.train_text_encoder,
                    "train_embeddings": self.train_embeddings,
                }
                (output_path / "config.json").write_text(
                    json.dumps(config, indent=2), encoding="utf-8"
                )

        return _Model()

    @staticmethod
    def from_pretrained(model_path: str | Path, torch_dtype=None):
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoModelForSeq2SeqLM

        path = Path(model_path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("architecture") != "adaptive_nesy_gen_vit_flan_t5":
            raise ValueError(f"Not a ViT/Flan-T5 lightweight checkpoint: {path}")
        vision_source = path / "vision_encoder"
        text_source = path / "text_model"
        vision_encoder = AutoModel.from_pretrained(
            vision_source if vision_source.exists() else config["encoder"],
            add_pooling_layer=False,
            torch_dtype=torch_dtype,
        )
        text_model = AutoModelForSeq2SeqLM.from_pretrained(
            text_source if text_source.exists() else config["decoder"],
            torch_dtype=torch_dtype,
        )
        model = FrozenViTFlanT5ReportModel.create(
            vision_encoder,
            text_model,
            visual_tokens=int(config.get("visual_tokens", 32)),
            train_text_encoder=bool(config.get("train_text_encoder", False)),
            train_embeddings=bool(config.get("train_embeddings", False)),
        )
        state = load_file(path / "projection.safetensors")
        model.visual_projection.load_state_dict(state)
        return model
