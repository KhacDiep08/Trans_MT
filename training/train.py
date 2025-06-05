import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from model import Transformer

class ExampleDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src = src_data
        self.tgt = tgt_data

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.src[idx], dtype=torch.long),
            "labels": torch.tensor(self.tgt[idx], dtype=torch.long),
        }

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": labels}

def main():
    src_data = []
    tgt_data = []

    train_dataset = ExampleDataset(src_data, tgt_data)

    model = Transformer(
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        pad_idx=0
    )

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    def compute_loss(model, inputs, return_outputs=False):
        outputs = model(inputs["input_ids"], inputs["labels"][:, :-1])
        logits = outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=None,
        data_collator=collate_fn,
        compute_loss=compute_loss
    )

    trainer.train()

if __name__ == "__main__":
    main()
