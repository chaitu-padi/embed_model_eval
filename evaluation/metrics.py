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
    
    # Calculate all true/false positives/negatives
    true_positives = len(retrieved_set & relevant_set)
    false_positives = len(retrieved_set - relevant_set)
    false_negatives = len(relevant_set - retrieved_set)
    # For accuracy, we also need true negatives (correctly not retrieved irrelevant items)
    # In our case, true_negatives would be (total_possible_items - tp - fp - fn)
    # But since we don't have total_possible_items, we'll use a modified accuracy
    
    # Calculate metrics
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    
    # Modified accuracy calculation considering both precision and recall
    # This gives us a balanced measure of both correct retrievals and correct exclusions
    accuracy = (true_positives) / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0.0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"[Evaluation][Debug] Metrics calculation details:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }
