from types import SimpleNamespace

import numpy as np
from PIL import Image

from adaptive_nesy_gen.schema import Study
from scripts.train_lightweight_vlm import LightweightReportDataset


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text, **kwargs):
        del kwargs
        # Stable token counts are sufficient to verify shifting and loss masking.
        return SimpleNamespace(input_ids=list(range(1, len(text.split()) + 1)))


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
        decoder_start_token_id=77,
        training=False,
    )

    row = dataset[0]

    assert row["decoder_input_ids"][0] == 77
    assert len(row["decoder_input_ids"]) == len(row["labels"])
    assert row["labels"][-1] == FakeTokenizer.eos_token_id
    first_target = next(index for index, token in enumerate(row["labels"]) if token != -100)
    assert first_target > 0
    assert row["decoder_input_ids"][first_target + 1] == row["labels"][first_target]
