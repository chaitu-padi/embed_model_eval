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
    
    # Calculate all true/false positives/negatives
    true_positives = len(retrieved_set & relevant_set)
    false_positives = len(retrieved_set - relevant_set)
    false_negatives = len(relevant_set - retrieved_set)
    # For accuracy, we also need true negatives (correctly not retrieved irrelevant items)
    # For ranking tasks, we can consider true negatives as items that were correctly not retrieved
    # when they were not relevant. This is approximated as the total possible items minus the union
    # of retrieved and relevant sets
    
    # Calculate metrics
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    
    # Calculate accuracy
    # In information retrieval, accuracy measures the proportion of correct decisions
    # out of all decisions made (correct and incorrect retrievals)
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    total_decisions = true_positives + false_positives + false_negatives
    accuracy = 1.0 if total_decisions == 0 else true_positives / total_decisions
    
    # Calculate F1 score using harmonic mean of precision and recall
    if precision + recall > 0:  # Only check if denominator would be non-zero
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0  # F1 score is 0 if both precision and recall are 0
    
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
