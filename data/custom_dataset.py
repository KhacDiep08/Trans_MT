from torch.utils.data import Dataset
import torch

class MachineTranslationDataset(Dataset):
    def __init__(self, src_encoded, tgt_encoded, bos_id, eos_id):
        self.src_encoded = src_encoded
        self.tgt_input = [[bos_id] + seq for seq in tgt_encoded]
        self.tgt_output = [seq + [eos_id] for seq in tgt_encoded]

    def __len__(self):
        return len(self.src_encoded)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.src_encoded[idx], dtype=torch.long),
            "labels": torch.tensor(self.tgt_output[idx], dtype=torch.long),
        }
