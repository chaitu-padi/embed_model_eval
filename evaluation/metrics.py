def evaluate(retrieved_ids, relevant_ids, top_k):
    relevant_ids = set(relevant_ids)
    true_positives = len(set(retrieved_ids) & relevant_ids)
    accuracy = true_positives / top_k if top_k else 0
    recall = true_positives / len(relevant_ids) if relevant_ids else 0
    precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }
