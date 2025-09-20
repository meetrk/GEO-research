from typing import List
def choose_document(sources: List[str], scores: List[int]) -> int:
    """
    Chooses the index of one document from the sources based on the lowest non-zero score.
    If multiple documents have the same lowest non-zero score, chooses the one with the highest index.

    Args:
        sources (List[str]): List of document strings.
        scores (List[int]): List of corresponding scores for each document.

    Returns:
        int: The index of the chosen document with the lowest non-zero score. If multiple documents 
             have the same lowest non-zero score, returns the one with the highest index.
             Returns -1 if no valid document is found.
    """

    if not sources or not scores or len(sources) != len(scores):
        return -1

    non_zero_indices = [i for i, score in enumerate(scores) if score != 0]
    if not non_zero_indices:
        return -1

    min_non_zero_score = min(scores[i] for i in non_zero_indices)
    min_score_indices = [i for i in non_zero_indices if scores[i] == min_non_zero_score]
    chosen_index = max(min_score_indices)

    return chosen_index
