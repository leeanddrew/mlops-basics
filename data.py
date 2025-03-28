import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer

import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"


class DataModule(pl.LightningDataModule):
    def __init__(self,model_name="google/bert_uncased_L-2_H-128_A-2",batch_size=32,max_length = 128):
        super().__init__()
        
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("./hf_cache/models--google--bert_uncased_L-2_H-128_A-2/snapshots/30b0a37ccaaa32f332884b96992754e246e48c5f/",local_files_only=True)
    
    def prepare_data(self):
        cola_dataset = load_dataset("glue","cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset['validation']
    
    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids","attention_mask","label"]
            )
            
            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", columns=["sentence","input_ids", "attention_mask", "label"]
            )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )

if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)