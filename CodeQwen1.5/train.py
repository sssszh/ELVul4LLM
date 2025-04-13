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
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import transformers
from transformers import EvalPrediction, WEIGHTS_NAME
from transformers import Trainer
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from Dataset.CodeQwenDefectDataset import CodeQwenBaseDataset
from transformers import RobertaTokenizer
from accelerate import Accelerator 
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings("ignore")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
)

q_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

def compute_metrics(p: EvalPrediction):
    # grouped_indices is a two-dimensional array where each row represents the indices of various datapoints in p that have 
    # the same prompt 
    # grouped_labels is the actual ternary labels of <prompt, completion> datapoints at the indices provided by grouped_indices

    pred_raw, labels = p    
      
    pred = np.argmax(pred_raw, axis=1)
    # pred_binary = pred == pass_idx
    # labels_binary = labels == pass_idx
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
    }

    return metrics

def run_training(args, train_data, val_data):

    model_path = args.model_path if args.model_path is not None else '{}'.format(args.model)
    print("Loading model from {}...".format(model_path))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_path, quantization_config=q_config, device_map={"": Accelerator().process_index}, pad_token_id=tokenizer.eos_token_id, num_labels=args.num_labels)
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()
    
    print('Finished loading model {}'.format(args.model))

    start_iteration = 0
    train_data.start_iteration = start_iteration
    print(f"Starting main loop")

    training_args = transformers.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True, 
        
        do_train=True,
        do_eval=True,
        do_predict=False,
        save_strategy='steps',
        evaluation_strategy='steps',
        eval_steps=284, 

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=568,
        lr_scheduler_type='linear',

        logging_dir=args.save_dir, 
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=args.save_freq,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=8,

        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing_kwargs={'use_reentrant':False}
        
    )
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=lambda x: compute_metrics(x),
    )
    trainer.train()

    if args.local_rank == 0:
        lora_model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))

def get_dataset(args, mode="train"): 
    
    if mode == "train": 
        dataroot = args.train_path
        with open(args.train_path, 'r') as f:
            problems_1 = f.readlines()
    elif mode == "val":
        dataroot = args.val_path
        with open(args.val_path, 'r') as f:
            problems_1 = f.readlines()
    
    # problems_2 = problems_2[:100]
    # train in debugging mode with small data split 
    if args.db and mode == "train":
        problems_1 = problems_1[:640]
    elif args.db and mode == "val":
        problems_1 = problems_1[:50]
    
    train_data = CodeQwenBaseDataset(
        dataroot=dataroot,
        problems=problems_1,
        model=args.model_path,
        max_tokens=1024,
    )

    return train_data

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset 
    train_data = get_dataset(args, "train")
    val_data = get_dataset(args, "val")

    # Save args to file
    json.dump(argsdict, open(os.path.join(args.save_dir, "args.json"), 'w'))

    # Load and train model; save model checkpoints 
    run_training(args, train_data, val_data)


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description="Training a model for code generation")
    parser.add_argument('--model', default="", type=str, help='type of transformers model as model backbone')
    parser.add_argument('--model_path', default="", type=str, help='path to model backbone pretrained weights') 
    parser.add_argument('--save_dir', default='', type=str, help='path to save trained model checkpoints') 

    # Dataloading
    parser.add_argument('--train_path', default="", type=str, help='path to training data')
    parser.add_argument('--val_path', default="", type=str, help='path to training data')
    parser.add_argument('--test_path', default="", type=str, help='path to training data')

    # Model
    parser.add_argument('--clone_head', default=False, action='store_true', help='Optional: clone a seperate linear layer for RL samples and initialize it from finetuned LM head')
    parser.add_argument('--num_labels', default=2, type=int, help="")
    # Training
    parser.add_argument('--epochs', default=15, type=int, help='total number of training epochs')
    parser.add_argument('--lr', default=2e-5, type=float, help='training learning rate')
    parser.add_argument('--batch-size-per-replica', default=16, type=int, help='batch size per GPU')
    parser.add_argument('--grad-acc-steps', default=2, type=int, help='number of training steps before each gradient update')
    parser.add_argument('--deepspeed', default = None, type=str, help='path to deepspeed configuration file; set None if not using deepspeed')
    parser.add_argument('--fp16', default=False, action='store_true', help='set 16-bit training to reduce memory usage')
    parser.add_argument('--bf16', default=True, action='store_true', help='set 16-bit training to reduce memory usage')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--db', default=False, action='store_true', help='set to turn on debug mode i.e. using dummy small data split and only 1 data worker')
    parser.add_argument('--cnn_size', type=int, default=128, help="For cnn size.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--filter_size', type=int, default=2, help="For cnn filter size.")
    parser.add_argument('--d_size', type=int, default=128, help="For cnn filter size.")
    # Logging
    parser.add_argument('--log-freq', default=10, type=int, help='save training log after this number of training steps')
    parser.add_argument('--save-freq', default=284, type=int, help='save model checkpoints after this number of training steps')
    parser.add_argument('--save_total_limit', default=30, type=int, help='total of number checkpoints to keep; only keep the latest ones') 

    args = parser.parse_args()

    main(args)