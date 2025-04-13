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
from Dataset.BERTDefectDataset import BERTBaseDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse

parser = argparse.ArgumentParser(description="Training a model for code generation")
# Dataloading
parser.add_argument('--train_path', default="train.jsonl", type=str, help='path to training data')
parser.add_argument('--val_path', default="test_DGS.jsonl", type=str, help='path to training data')
parser.add_argument('--test_path', default="test_DGS.jsonl", type=str, help='path to training data')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def get_dataset(args, mode="train"): 
    
    if mode == "train": 
        dataroot = args.train_path
        with open(args.train_path, 'r') as f:
            problems_1 = f.readlines()
    elif mode == "val":
        dataroot = args.val_path
        with open(args.val_path, 'r') as f:
            problems_1 = f.readlines()
    elif mode == "test":
        dataroot = args.test_path
        with open(args.test_path, 'r') as f:
            problems_1 = f.readlines()
    
    # problems_2 = problems_2[:100]
    # train in debugging mode with small data split 
    
    train_data = BERTBaseDataset(
        dataroot=dataroot,
        problems=problems_1,
        model="/CodeBERT/model",
        max_tokens=512,
    )

    return train_data

eval_dataset = get_dataset(args, "test")
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=8, num_workers=8)

model = RobertaForSequenceClassification.from_pretrained("", num_labels=5)
model.to(device)

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
        outputs = model(**inputs)
        logits = outputs.logits
        logits_list.append(logits.detach().cpu().numpy())
        labels_list.append(batch_labels.detach().cpu().numpy())
logits_list=np.concatenate(logits_list,0)
labels_list=np.concatenate(labels_list,0)
preds = np.argmax(logits_list, axis=1)
softmax = torch.nn.Softmax(dim=-1)
prob = softmax(torch.tensor(logits_list))[:, 1]


with open("result_0.txt", 'r') as f:
        codellama_7B = f.readlines()
with open("result_1.txt", 'r') as f:
    codellama_13B = f.readlines()
with open("result_2.txt", 'r') as f:
    codeqwen = f.readlines()
with open("result_3.txt", 'r') as f:
    deepseek = f.readlines()
with open("result_4.txt", 'r') as f:
    starcoder = f.readlines()
new_pred = []
for idx, p in enumerate(preds):
    if p == 0:
        new_pred.append(int(codellama_7B[idx].strip()))
    elif p == 1:
        new_pred.append(int(codellama_13B[idx].strip()))
    elif p == 2:
        new_pred.append(int(codeqwen[idx].strip()))
    elif p == 3:
        new_pred.append(int(deepseek[idx].strip()))
    elif p == 4:
        new_pred.append(int(starcoder[idx].strip()))

new_labels = []
with open("test_DGS.jsonl", 'r') as f:
    test_data = f.readlines()
for line in test_data:
    json_line = json.loads(line)
    new_labels.append(json_line["target"])

with open("/result.txt", 'w') as f:
    for prediction in new_pred:
        f.write(f"{prediction}\n")

accuracy = accuracy_score(y_true=new_labels, y_pred=new_pred)
recall = recall_score(y_true=new_labels, y_pred=new_pred, average='weighted')
precision = precision_score(y_true=new_labels, y_pred=new_pred, average='weighted')
f1 = f1_score(y_true=new_labels, y_pred=new_pred, average='weighted')

metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
    }

print(metrics)
