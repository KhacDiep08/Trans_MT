import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split

from data.custom_dataset import MachineTranslationDataset
from data.data_collator import DataCollatorMT  # nếu bạn dùng, hoặc bỏ
from mt_model.model import Transformer
from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": labels}

def main():
    # Load tokenizer
    tokenizer_en = SentencePieceTokenizer(model_prefix="tokenizer/vocab/spm_en")
    tokenizer_en.load()
    tokenizer_vi = SentencePieceTokenizer(model_prefix="tokenizer/vocab/spm_vi")
    tokenizer_vi.load()
            
    # Load data
    data = pd.read_csv("data/preprocessed_200k.csv").head(1000)
    
    # Chia train (70%) và temp (30%)
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)

    # Chia temp thành eval (15%) và test (15%)
    eval_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, shuffle=True)
    
    print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}, Test size: {len(test_data)}")

    # Load datasets
    train_dataset = MachineTranslationDataset(train_data, tokenizer_en, tokenizer_vi, src_lang='en', tgt_lang='vi', max_length=128)
    eval_dataset = MachineTranslationDataset(eval_data, tokenizer_en, tokenizer_vi, src_lang='en', tgt_lang='vi', max_length=128)
    test_dataset = MachineTranslationDataset(test_data, tokenizer_en, tokenizer_vi, src_lang='en', tgt_lang='vi', max_length=128)
    
    print(train_dataset[0])  # Kiểm tra dữ liệu đầu vào
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        pad_idx=0
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Train arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",  # eval_strategy deprecated
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

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            loss, outputs = model(input_ids=inputs['input_ids'], labels=inputs['labels'])

            if return_outputs:
                return loss, outputs
            return loss

        
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

if __name__ == "__main__":
    main()
