def evaluate(retrieved_ids, relevant_ids, top_k):
    """
    Calculate evaluation metrics for retrieved results.
    
    Args:
        retrieved_ids: List of IDs returned by the search
        relevant_ids: List of known relevant IDs (ground truth)
        top_k: Number of top results to consider
    
    Returns:
        Dictionary containing precision, recall, F1 score, and accuracy
    """
    if not retrieved_ids or not relevant_ids:
        print("[Evaluation][Warning] Empty retrieved_ids or relevant_ids")
        return {
            'accuracy': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'f1': 0.0
        }

    print("[Evaluation][Debug] Retrieved IDs:", retrieved_ids)
    print("[Evaluation][Debug] Relevant IDs:", relevant_ids)
    
    # Convert to sets for intersection
    retrieved_set = set(retrieved_ids[:top_k])  # Only consider top-k results
    relevant_set = set(relevant_ids)
    
    # Calculate true positives (correctly retrieved items)
    true_positives = len(retrieved_set & relevant_set)
    
    # Calculate metrics
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    accuracy = true_positives / top_k if top_k > 0 else 0.0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }
