import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from adaptive_nesy_gen.lightweight_vit_t5 import FrozenViTFlanT5ReportModel
from adaptive_nesy_gen.schema import Study
from scripts.train_lightweight_vlm import LightweightReportDataset


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text, **kwargs):
        # Stable token counts are sufficient to verify shifting and loss masking.
        tokens = list(range(1, len(text.split()) + 1))
        if kwargs.get("add_special_tokens"):
            tokens.append(self.eos_token_id)
        return SimpleNamespace(input_ids=tokens)


class FakeProcessor:
    size = {"height": 8, "width": 8}

    def __call__(self, images, return_tensors):
        assert images.mode == "RGB"
        assert return_tensors == "pt"
        return SimpleNamespace(pixel_values=[np.zeros((3, 8, 8), dtype=np.float32)])


class FakeVisionEncoder:
    config = SimpleNamespace(hidden_size=4)

    def parameters(self):
        return []


class FakeTextModel:
    config = SimpleNamespace(d_model=8)
    generation_config = SimpleNamespace()
    decoder = SimpleNamespace(parameters=lambda: [])
    lm_head = SimpleNamespace(parameters=lambda: [])
    encoder = SimpleNamespace(parameters=lambda: [])

    def parameters(self):
        return []

    def get_input_embeddings(self):
        return SimpleNamespace(parameters=lambda: [])


def test_lightweight_dataset_masks_prefix_and_shifts_target(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)
    study = Study("s1", str(image_path), "lungs are clear", indication="cough")
    dataset = LightweightReportDataset(
        [study],
        FakeTokenizer(),
        FakeProcessor(),
        training=False,
    )

    row = dataset[0]

    assert row["input_ids"]
    assert len(row["input_ids"]) == len(row["attention_mask"])
    assert all(token != -100 for token in row["labels"])
    assert row["labels"][-1] == FakeTokenizer.eos_token_id


def test_vit_t5_wrapper_exposes_trainer_config():
    model = FrozenViTFlanT5ReportModel.create(FakeVisionEncoder(), FakeTextModel())

    assert model.config is FakeTextModel.config
    assert model.generation_config is FakeTextModel.generation_config


def test_vit_t5_forward_supplies_decoder_input_ids():
    torch = __import__("torch")
    nn = torch.nn

    class TinyVisionEncoder(nn.Module):
        config = SimpleNamespace(hidden_size=4)

        def forward(self, pixel_values, return_dict):
            assert return_dict
            batch = pixel_values.shape[0]
            hidden = torch.ones(batch, 3, 4, device=pixel_values.device)
            return SimpleNamespace(last_hidden_state=hidden)

    class TinyEncoder(nn.Module):
        def forward(self, inputs_embeds, attention_mask, return_dict):
            assert attention_mask.shape[1] == inputs_embeds.shape[1]
            assert return_dict
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    class TinyTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(d_model=8)
            self.generation_config = SimpleNamespace()
            self.encoder = TinyEncoder()
            self.decoder = nn.Linear(8, 8)
            self.lm_head = nn.Linear(8, 16)
            self.embedding = nn.Embedding(16, 8)
            self.seen_decoder_input_ids = None

        def get_input_embeddings(self):
            return self.embedding

        def _shift_right(self, labels):
            shifted = labels.new_full(labels.shape, 0)
            shifted[:, 1:] = labels[:, :-1].clamp_min(0)
            return shifted

        def forward(
            self,
            encoder_outputs,
            attention_mask,
            decoder_input_ids=None,
            labels=None,
            return_dict=True,
        ):
            del encoder_outputs, attention_mask
            assert return_dict
            assert decoder_input_ids is not None
            assert labels is not None
            self.seen_decoder_input_ids = decoder_input_ids
            return SimpleNamespace(loss=torch.tensor(0.0))

    text_model = TinyTextModel()
    model = FrozenViTFlanT5ReportModel.create(
        TinyVisionEncoder(),
        text_model,
        visual_tokens=2,
    )
    output = model(
        pixel_values=torch.zeros(2, 3, 8, 8),
        input_ids=torch.ones(2, 4, dtype=torch.long),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        labels=torch.tensor([[4, 5, 6], [7, -100, 8]]),
    )

    assert output.loss.item() == 0.0
    assert text_model.seen_decoder_input_ids is not None
    assert text_model.seen_decoder_input_ids.shape == (2, 3)


def test_lightweight_trainer_keeps_labels_for_custom_t5_forward():
    script = Path("scripts/train_lightweight_vlm.py").read_text(encoding="utf-8")
    tree = ast.parse(script)
    smoothing_values = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "label_smoothing_factor"
    ]

    assert smoothing_values
    assert all(isinstance(value, ast.Constant) and value.value == 0.0 for value in smoothing_values)
