import json
def read_pred_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [int(line.strip()) for line in lines]

def read_prob_results(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data  

def hard_voting(*args):
    
    voted_results = []
    for votes in zip(*args):
        vote_count = {}
        for vote in votes:
            if vote in vote_count:
                vote_count[vote] += 1
            else:
                vote_count[vote] = 1
        
        voted_results.append(max(vote_count, key=vote_count.get))
    return voted_results

def soft_voting(*probabilities):
   
    averaged_results = []
    for probs in zip(*probabilities):
        avg_probs = [sum(category_probs) / len(category_probs) for category_probs in zip(*probs)]
        averaged_results.append(avg_probs.index(max(avg_probs)))
    return averaged_results

def write_results(results, file_path):
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result}\n")


model1_pred_results = read_pred_results('result_0.txt')
model2_pred_results = read_pred_results('result_1.txt')
model3_pred_results = read_pred_results('result_2.txt')
model4_pred_results = read_pred_results('result_3.txt')
model5_pred_results = read_pred_results('result_4.txt')


model1_prob_results = read_prob_results('result_prob_0.json')
model2_prob_results = read_prob_results('result_prob_1.json')
model3_prob_results = read_prob_results('result_prob_2.json')
model4_prob_results = read_prob_results('result_prob_3.json')
model5_prob_results = read_prob_results('result_prob_4.json')


hard_votes = hard_voting(model1_pred_results, model2_pred_results, model3_pred_results, model4_pred_results, model5_pred_results)
soft_votes = soft_voting(model1_prob_results, model2_prob_results, model3_prob_results, model4_prob_results, model5_prob_results)

write_results(hard_votes, 'hard.txt')
write_results(soft_votes, 'soft.txt')


