import io
import logging
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pprint
import sys
import time
import json
import pickle as pkl
import pdb 
import numpy as np
import scipy
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import transformers
from transformers import EvalPrediction, WEIGHTS_NAME
from transformers import Trainer
import torch
import random
from Dataset.CodeLLamaDefectDataset import CodeLLamaBaseDataset
from transformers import CodeLlamaTokenizer, LlamaForSequenceClassification, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, AutoPeftModel
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse

parser = argparse.ArgumentParser(description="Training a model for code generation")
# Dataloading
parser.add_argument('--model_path', default="", type=str, help='path to training data')
parser.add_argument('--lora_path', default="", type=str, help='path to training data')
parser.add_argument('--train_path', default="", type=str, help='path to training data')
parser.add_argument('--val_path', default="", type=str, help='path to training data')
parser.add_argument('--test_path', default="", type=str, help='path to training data')
parser.add_argument('--mode', default="test", type=str, help='path to training data')
parser.add_argument('--num_labels', default=2, type=int, help='path to training data')


args = parser.parse_args()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

q_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

def get_dataset(args): 
    
    if args.mode == "train": 
        dataroot = args.train_path
        with open(args.train_path, 'r') as f:
            problems_1 = f.readlines()
    elif args.mode == "val":
        dataroot = args.val_path
        with open(args.val_path, 'r') as f:
            problems_1 = f.readlines()
    elif args.mode == "test":
        dataroot = args.test_path
        with open(args.test_path, 'r') as f:
            problems_1 = f.readlines()
    
    # problems_2 = problems_2[:100]
    # train in debugging mode with small data split 
    
    train_data = CodeLLamaBaseDataset(
        dataroot=dataroot,
        problems=problems_1,
        model=args.model_path,
        max_tokens=1024,
    )

    return train_data

eval_dataset = get_dataset(args)
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=8, num_workers=8)

tokenizer = transformers.CodeLlamaTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForSequenceClassification.from_pretrained(args.model_path, device_map="auto", quantization_config=q_config, pad_token_id=tokenizer.eos_token_id, num_labels=args.num_labels)
# model = LlamaForSequenceClassification.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16, pad_token_id=tokenizer.eos_token_id)
lora_model = PeftModel.from_pretrained(model, args.lora_path)
lora_model.to(device)

logits_list = []
labels_list = []
for batch in tqdm(eval_dataloader, ncols=0, total=len(eval_dataloader)):
    with torch.no_grad():
        batch_input_ids = batch["input_ids"].to(device)
        batch_attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)
        inputs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
            }
        outputs = lora_model(**inputs)
        logits = outputs.logits
        # logits_list.append(logits.detach().cpu().numpy())
        # labels_list.append(batch_labels.detach().cpu().numpy())

        logits_list.append(logits.detach().to(torch.float).cpu().numpy())
        labels_list.append(batch_labels.detach().to(torch.float).cpu().numpy())
logits_list=np.concatenate(logits_list,0)
labels_list=np.concatenate(labels_list,0)
preds = np.argmax(logits_list, axis=1)
softmax = torch.nn.Softmax(dim=-1)
prob = softmax(torch.tensor(logits_list))[:, 1]

accuracy = accuracy_score(y_true=labels_list, y_pred=preds)
recall = recall_score(y_true=labels_list, y_pred=preds, average='weighted')
precision = precision_score(y_true=labels_list, y_pred=preds, average='weighted')
f1 = f1_score(y_true=labels_list, y_pred=preds, average='weighted')

print(f1)

metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
    }
json.dump(metrics, open("result.json", 'w'))
with open("result.txt", 'w') as f:
    for item in preds:
        f.write(str(item) + '\n')

with open("result_prob.txt", 'w') as f:
    for p in prob:
        f.write(str(np.array(p)) + '\n')

