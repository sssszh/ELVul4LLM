
import torch
import io
import numpy as np
import gc
import os
import random
from tqdm import tqdm 
from collections import Counter
import json, pdb 

from multiprocessing import Manager
import transformers
# from parserTool.utils import remove_comments_and_docstrings

class CodeQwenBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problems, model, max_tokens) -> None:
        super().__init__()
        self.dataroot = dataroot
        self.problems = problems 

        self.model = model
        
        self.max_tokens = max_tokens

        self.samples = []           
        self.initialize()
        print("===================================================================================")
        print("load tokenizer:", model)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def initialize(self):

        all_samples = []

        print(f"Loading {len(self.problems)} problems from {self.dataroot}.")

        for idx, line in tqdm(enumerate(self.problems), ncols=0, total=len(self.problems)):
            json_line = json.loads(line)
            # clean_code, code_dict = remove_comments_and_docstrings(json_line['func'], 'c')
            code = ' '.join(json_line['func'].split())
            target = json_line["target"]
            sample = (code, target)
            all_samples.append(sample)

        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        inputs = self.pack_samples(idx)
        return inputs
    
    def pack_samples(self, idx):

        sample_pool = self.samples
        code, target = sample_pool[idx]
        source_ids = self.tokenizer.encode(code, max_length=self.max_tokens, padding='max_length', truncation=True)
        attention_ids = torch.tensor(source_ids)
        attention_mask = attention_ids.ne(self.tokenizer.pad_token_id)

        out_sample = {
            "input_ids": torch.tensor(source_ids),
            "attention_mask": attention_mask,
            "labels": torch.tensor(target)
        }

        return out_sample

