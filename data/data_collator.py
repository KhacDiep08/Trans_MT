from torch.nn.utils.rnn import pad_sequence

class DataCollatorMT:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_id
        )
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "labels": labels
        }
