import numpy as np

from adaptive_nesy_gen.schema import Study
from scripts.train_chexagent_qlora import ReportDataset


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int64)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        value = self.values[key]
        return FakeTensor(value) if isinstance(key, slice) else value

    def __setitem__(self, key, value):
        self.values[key] = value

    def __eq__(self, other):
        return self.values == other

    def __ne__(self, other):
        return self.values != other

    def clone(self):
        return FakeTensor(self.values.copy())


class FakeCheXagentTokenizer:
    img_start_id = 10
    img_end_id = 11

    def __init__(self):
        self.calls = []

    def from_list_format(self, rows):
        return rows

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        assert return_tensors == "pt"
        self.calls.append(messages)
        prompt_text = messages[1]["value"][1]["text"]
        neighbours = prompt_text.count("Training neighbour")
        prompt_length = 12 + 5 * neighbours
        values = [self.img_start_id, *([0] * (prompt_length - 2)), self.img_end_id]
        if not add_generation_prompt:
            assert messages[-1]["from"] == "gpt"
            values.extend([90, 91, 92])
        return [FakeTensor(values)]


def test_report_dataset_preserves_image_span_and_supervised_target():
    studies = [
        Study("current", "/tmp/current.png", "No focal opacity."),
        Study("n1", "/tmp/n1.png", "Neighbour one."),
        Study("n2", "/tmp/n2.png", "Neighbour two."),
        Study("n3", "/tmp/n3.png", "Neighbour three."),
    ]
    tokenizer = FakeCheXagentTokenizer()
    dataset = ReportDataset(
        studies,
        tokenizer,
        max_length=20,
        neighbours=np.asarray([[1, 2, 3], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
        rag_probability=1.0,
        min_target_tokens=3,
    )

    row = dataset[0]

    assert list(row["input_ids"].values).count(tokenizer.img_start_id) == 1
    assert list(row["input_ids"].values).count(tokenizer.img_end_id) == 1
    assert row["labels"].values[-3:].tolist() == [90, 91, 92]
    assert np.all(row["labels"].values[:-3] == -100)
    final_prompt = tokenizer.calls[-1][1]["value"][1]["text"]
    assert "Training neighbour 1" in final_prompt
    assert "Training neighbour 2" not in final_prompt
