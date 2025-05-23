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

class BERTBaseDataset(torch.utils.data.Dataset):
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

        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(model)

    def initialize(self):

        all_samples = []

        print(f"Loading {len(self.problems)} problems from {self.dataroot}.")

        for idx, line in tqdm(enumerate(self.problems), ncols=0, total=len(self.problems)):
            json_line = json.loads(line)
            # clean_code, code_dict = remove_comments_and_docstrings(json_line['func'], 'c')
            code = ' '.join(json_line['func'].split())
            target = json_line["assigned_label"]
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

        mask_padding_with_zero = True  
        sample_pool = self.samples
        code, target = sample_pool[idx]
        code_tokens = self.tokenizer.tokenize(code)[:self.max_tokens-2]
        source_tokens =[self.tokenizer.cls_token]+code_tokens+[self.tokenizer.sep_token]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(source_ids)
        padding_length = self.max_tokens - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id]*padding_length
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        out_sample = {
            "input_ids": torch.tensor(source_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(target)
        }

        return out_sample

