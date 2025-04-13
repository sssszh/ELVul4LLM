def read_pred_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [int(line.strip()) for line in lines]

def read_prob_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [float(line.strip()) for line in lines]

def hard_voting(*args):
    return [1 if sum(votes) > len(votes) / 2 else 0 for votes in zip(*args)]

def soft_voting(*probabilities):
    return [1 if sum(probs) / len(probs) >= 0.5 else 0 for probs in zip(*probabilities)]

def write_results(results, file_path):
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result}\n")

model1_pred_results = read_pred_results('result_0.txt')
model2_pred_results = read_pred_results('result_1.txt')
model3_pred_results = read_pred_results('result_2.txt')
model4_pred_results = read_pred_results('result_3.txt')
model5_pred_results = read_pred_results('result_4.txt')


model1_prob_results = read_prob_results('result_prob_0.txt')
model2_prob_results = read_prob_results('result_prob_1.txt')
model3_prob_results = read_prob_results('result_prob_2.txt')
model4_prob_results = read_prob_results('result_prob_3.txt')
model5_prob_results = read_prob_results('result_prob_4.txt')

hard_votes = hard_voting(model1_pred_results, model2_pred_results, model3_pred_results, model4_pred_results, model5_pred_results)

soft_votes = soft_voting(model1_prob_results, model2_prob_results, model3_prob_results, model4_prob_results, model5_prob_results)

write_results(hard_votes, 'hard.txt')
write_results(soft_votes, 'soft.txt')
