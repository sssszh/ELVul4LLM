import json
import random

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def sample_dataset(data, sample_size, random_seed=None):
    if random_seed:
        random.seed(random_seed)
    sampled_data = random.choices(data, k=sample_size)
    return sampled_data

data = load_data("../../Datas/train.jsonl")

for i in range(5):
    sampled_data = sample_dataset(data, sample_size=len(data), random_seed=i+i*1000)
    with open(f"./train_{i}.jsonl", 'w') as f:
        for item in sampled_data:
            json_item = json.dumps(item)
            f.write(json_item + "\n")