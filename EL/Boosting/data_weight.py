import json

def read_jsonl_file(file_path):

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def add_initial_weights(data):

    for sample in data:
        sample['weight'] = 1.0
    return data

def write_jsonl_file(data, output_file_path):

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample) + '\n')


input_jsonl_file_path = 'train.jsonl'
output_jsonl_file_path = 'train_weight.jsonl'


data = read_jsonl_file(input_jsonl_file_path)

data_with_weights = add_initial_weights(data)


write_jsonl_file(data_with_weights, output_jsonl_file_path)


