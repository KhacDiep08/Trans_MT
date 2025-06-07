import torch
from torch.utils.data import Dataset
import pandas as pd

class MachineTranslationDataset(Dataset):
    def __init__(self, df, tokenizer_en, tokenizer_vi, src_lang='en', tgt_lang='vi', max_length=128):
        self.data = df if isinstance(df, pd.DataFrame) else pd.read_csv(df)
        self.tokenizer_en = tokenizer_en
        self.tokenizer_vi = tokenizer_vi
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.pad_token_id = 0  # giả sử token padding là 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data.iloc[idx][self.src_lang]
        tgt_text = self.data.iloc[idx][self.tgt_lang]

        # Mã hóa và padding/truncate
        src_encoded = self.pad_or_truncate(
            self.tokenizer_en.encode(src_text) if self.src_lang == 'en' else self.tokenizer_vi.encode(src_text)
        )
        tgt_encoded = self.pad_or_truncate(
            self.tokenizer_en.encode(tgt_text) if self.tgt_lang == 'en' else self.tokenizer_vi.encode(tgt_text)
        )

        # Tạo attention mask: 1 ở vị trí không phải pad, 0 ở vị trí pad
        src_attention_mask = [1 if token != self.pad_token_id else 0 for token in src_encoded]

        # Chuyển thành tensor
        return {
            "input_ids": torch.tensor(src_encoded, dtype=torch.long),
            "attention_mask": torch.tensor(src_attention_mask, dtype=torch.long),
            "labels": torch.tensor(tgt_encoded, dtype=torch.long),
        }

    def pad_or_truncate(self, tokens):
        if len(tokens) > self.max_length:
            return tokens[:self.max_length]
        return tokens + [self.pad_token_id] * (self.max_length - len(tokens))
