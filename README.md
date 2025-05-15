# ELVul4LLM -- Ensembling Large Language Models for Code Vulnerability Detection: An Empirical Evaluation

## 📜 Introduction
This is an empirical study on the impact of ensemble learning on the performance of LLMs in vulnerability detection.
## 📕 Datas

* ✨ **Devign**: [Data Source](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view)

* ✨ **ReVeal**: [Data Source](https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy)

* ✨ **BigVul**: [Data Source](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view)

## 💻 Experiments

### Install Dependencies

```conda env create -f environment.yml```

```sh
cd transformers
pip install -e .
```

### 📖 Baseline LLMs w/o EL

#### Fine-tuning LLMs for Vulnerability Defection with QLora (CodeLlama on Devign Example)

```sh
cd CodeLlama

*****Training*****

python train.py \
    --model ../CodeLlama/model \
    --model_path ../CodeLlama/model \
    --save_dir ../CodeLlama/outputs \
    --train_path ../Datas/Devign/train.jsonl \
    --val_path ../Datas/Devign/valid.jsonl \
    --test_path ../Datas/Devign/test.jsonl \
    --num_labels 2 \
    --epochs 15 \
    --lr 2e-5 \
    --batch-size-per-replica 16 \
    --grad-acc-steps 2 \
    --bf16 True \
    --seed 42 \
    --save_total_limit 5

*****Prediction*****

python prediction.py \
    --model_path ../CodeLlama/model \
    --lora_path ../CodeLlama/outputs \
    --train_path ../Datas/Devign/train.jsonl \
    --val_path ../Datas/Devign/valid.jsonl \
    --test_path ../Datas/Devign/test.jsonl \
    --num_labels 2
```

### 📖 Ensemble Learning

#### ✨ Bagging

* Preparing Data
```sh
cd EL/Bagging
python get_data.py
```
* Training LLMs on Bagging Subset
```sh
See Above
```
* Hard/Soft Voting
```sh
cd EL/Bagging
python vote.py
```

#### ✨ Boosting
* Preparing Data
```sh
cd EL/Boosting
python data_weight.py
```
* Training LLMs with Boosting (CodeLlama Example)
```
python train_boosting.py \
    --model ../CodeLlama/model \
    --model_path ../CodeLlama/model \
    --save_dir ../CodeLlama/outputs \
    --train_path ../Datas/Devign/train_weight.jsonl \
    --val_path ../Datas/Devign/valid_weight.jsonl \
    --test_path ../Datas/Devign/test_weight.jsonl \
    --num_labels 2 \
    --epochs 15 \
    --lr 2e-5 \
    --batch-size-per-replica 16 \
    --grad-acc-steps 2 \
    --bf16 True \
    --seed 42 \
    --save_total_limit 5

*****Prediction*****

python prediction.py \
    --model_path ../CodeLlama/model \
    --lora_path ../CodeLlama/outputs \
    --train_path ../Datas/Devign/train_weight.jsonl \
    --val_path ../Datas/Devign/valid_weight.jsonl \
    --test_path ../Datas/Devign/test_weight.jsonl \
    --num_labels 2
```

#### ✨ Stacking
* Preparing Data with Predictions of LLMs
```sh
cd EL/Stacking
python merge.py
```
* Training Meta-Model
```sh
*****LR*****
python LR.py

*****RF*****
python RF.py

*****KNN*****
python KNN.py

*****SVM*****
python SVM.py
```

#### ✨ DGS
* Preparing Data
cd EL/DGS
```sh
python precess_data.py
```
* Training
```sh
python train_DGS.py \
    --model ../CodeBERT \
    --model_path ../CodeBERT \
    --save_dir ../EL/DGS/outputs \
    --train_path ../Datas/Devign/train_DGS.jsonl \
    --val_path ../Datas/Devign/valid_DGS.jsonl \
    --test_path ../Datas/Devign/test_DGS.jsonl \
    --num_labels 5 \
    --epochs 60 \
    --lr 2e-5 \
    --batch-size-per-replica 16 \
    --grad-acc-steps 2 \
    --bf16 True \
    --seed 42 \
    --save_total_limit 5
```
