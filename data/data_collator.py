import torch

class DataCollatorMT:
    def __init__(self, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        input_ids = [example["input_ids"] for example in batch]
        attention_mask = [example["attention_mask"] for example in batch]
        labels = [example["labels"] for example in batch]

        # Pad input_ids và attention_mask
        inputs = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )

        # Pad labels
        labels_padded = self.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )["input_ids"]

        # Replace padding token id with -100 for labels (to ignore in loss)
        labels_padded[labels_padded == self.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels_padded
        return inputs
