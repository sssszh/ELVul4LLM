import json

jsonl_file = '/Datas/ReVeal/valid.jsonl' 
pred_files = ['result_0.txt', 
'result_0.txt', 
'result_0.txt', 
'result_0.txt', 
'result_0.txt']  
prob_files = ['result_prob_0.txt', 
'result_prob_0.txt', 
'result_prob_0.txt', 
'result_prob_0.txt', 
'result_prob_0.txt']  
label_output_file = '/train_DGS.jsonl' 


with open(jsonl_file, 'r') as f:
    data = [json.loads(line) for line in f]


pred_results = []
for pred_file in pred_files:
    with open(pred_file, 'r') as f:
        pred_results.append([int(line.strip()) for line in f])


prob_results = []
for prob_file in prob_files:
    with open(prob_file, 'r') as f:
        prob_results.append([float(line.strip()) for line in f])


labeled_data = []
for idx, entry in enumerate(data):
    best_classifier = -1
    best_prob = -1
    correct_label = entry['target']  

    for clf_idx in range(5):
        if pred_results[clf_idx][idx] == correct_label:
            if prob_results[clf_idx][idx] > best_prob:
                best_prob = prob_results[clf_idx][idx]
                best_classifier = clf_idx

    if best_classifier == -1:
      
        min_prob = float('inf')
        for clf_idx in range(5):
            if prob_results[clf_idx][idx] < min_prob:
                min_prob = prob_results[clf_idx][idx]
                best_classifier = clf_idx

    entry['assigned_label'] = best_classifier
    labeled_data.append(entry)

with open(label_output_file, 'w') as f:
    for entry in labeled_data:
        f.write(json.dumps(entry) + '\n')

print(f"Labeled data has been saved to {label_output_file}")
