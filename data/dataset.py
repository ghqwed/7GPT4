import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

class TextDataset(Dataset):
    def __init__(self, file_path, config):
        self.config = config
        
        # 验证数据文件
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data file not found at {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read().strip()
            
        if not self.text:
            raise ValueError("Training data file is empty")
            
        self.tokenizer = self._build_tokenizer(file_path)
        tokens = self.tokenizer.encode(self.text).ids
        
        if len(tokens) < self.config.model.max_seq_len + 1:
            raise ValueError(f"Data is too short (len={len(tokens)}), needs at least {self.config.model.max_seq_len + 1} tokens")
            
        self.data = torch.tensor(tokens, dtype=torch.long)
        
    def _build_tokenizer(self, file_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            vocab_size=self.config.model.vocab_size
        )
        
        if not os.path.exists("data/tokenizer.json"):
            tokenizer.train([file_path], trainer)
            tokenizer.save("data/tokenizer.json")
        else:
            tokenizer = Tokenizer.from_file("data/tokenizer.json")
            
        return tokenizer
    
    def __len__(self):
        seq_len = len(self.data) - self.config.model.max_seq_len
        return max(0, seq_len)  # 确保不会返回负值
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.config.model.max_seq_len+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_batch(dataset, batch_size):
    idx = torch.randint(len(dataset), (batch_size,))
    inputs = torch.stack([dataset[i][0] for i in idx])
    targets = torch.stack([dataset[i][1] for i in idx])
    return inputs, targets
