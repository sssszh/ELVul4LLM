from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

import numpy as np
import json


train_data = json.load(open("/valid.json", 'r'))
test_data = json.load(open("/test.json", 'r'))
train_label = []

with open("/valid.jsonl", 'r') as f:
    data = f.readlines()

for idx, line in enumerate(data):
    json_line = json.loads(line)
    train_label.append(json_line["target"])

with open("/test.jsonl", 'r') as f:
    test_data_ = f.readlines()


meta_learner = LogisticRegression(C=0.5, max_iter=100, penalty='l2', solver='lbfgs')

meta_learner.fit(train_data, train_label)


final_predictions = meta_learner.predict(test_data)





test_label = []
for idx, line in enumerate(test_data_):
    json_line = json.loads(line)
    test_label.append(json_line["target"])

# f1 = f1_score(y_true=test_label, y_pred=final_predictions)
# accuracy = accuracy_score(y_true=test_label, y_pred=final_predictions)
# recall = recall_score(y_true=test_label, y_pred=final_predictions)
# precision = precision_score(y_true=test_label, y_pred=final_predictions)

accuracy = accuracy_score(y_true=test_label, y_pred=final_predictions)
recall = recall_score(y_true=test_label, y_pred=final_predictions, average='weighted')
precision = precision_score(y_true=test_label, y_pred=final_predictions, average='weighted')
f1 = f1_score(y_true=test_label, y_pred=final_predictions, average='weighted')

metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
    }

print(metrics)
