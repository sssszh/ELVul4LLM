import json
import numpy as np

def read_jsonl_file(file_path):
 
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_txt_file(file_path):

    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(int(line.strip()))
    return results

def compute_sample_weights(y_true, y_pred, weights, alpha=0.3):

    incorrect = np.array(y_pred) != np.array(y_true)
    factor = np.exp(alpha * incorrect - alpha * (1 - incorrect))
    new_weights = weights * factor
    return new_weights

def update_jsonl_file(data, weights, output_file_path):

    for i, sample in enumerate(data):
        sample['weight'] = weights[i]

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample) + '\n')


jsonl_file_path = 'train_weight.jsonl'
txt_file_path = 'results.txt'
output_jsonl_file_path = 'train_weight_1.jsonl'


data = read_jsonl_file(jsonl_file_path)
y_true = [sample['target'] for sample in data]
y_pred = read_txt_file(txt_file_path)
original_weights = np.array([sample.get('weight', 1.0) for sample in data])


weights = compute_sample_weights(y_true, y_pred, original_weights)


update_jsonl_file(data, weights, output_jsonl_file_path)

print(f'样本权重已更新并保存到 {output_jsonl_file_path}')