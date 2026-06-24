from types import SimpleNamespace

import numpy as np
from PIL import Image

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
