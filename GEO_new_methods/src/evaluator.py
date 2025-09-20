import re
from typing import List, Sequence, Mapping, Tuple, Iterable
from collections import defaultdict
from lexrank import LexRank
from lexrank.mappings import STOPWORDS
from nltk.tokenize import sent_tokenize


def evaluate(response: str, sources: List[str]) -> List[Tuple[float, float]]:
    """
    Evaluates response against sources, returning importance and position-weighted word count scores.
    
    Args:
        response (str): The response text with citations.
        sources (List[str]): List of source documents.
    
    Returns:
        List[Tuple[float, float]]: List of (importance, position_weighted_word_count) scores.
    """
    documents = [sent_tokenize(doc) for doc in sources]
    response_sentences = sent_tokenize(response)
    
    lxr = LexRank(documents, stopwords=STOPWORDS['en'])
    lexrank_scores = lxr.rank_sentences(response_sentences, threshold=0.1, fast_power_method=False)
    
    importance_scores, position_weighted_wc = analyze_response(response_sentences, lexrank_scores)
    
    # Normalize by max value
    def normalize_by_max(score_dict):
        if not score_dict:
            return {}
        max_val = max(score_dict.values())
        return {k: v / max_val for k, v in score_dict.items()} if max_val > 0 else {k: 0.0 for k in score_dict}
    
    pos_wc_normalized = normalize_by_max(position_weighted_wc)
    
    # Return scores for each source
    result = []
    for i in range(len(sources)):
        source_index = i + 1  # 1-indexed
        imp_score = importance_scores.get(source_index, 0.0)
        pos_wc_score = pos_wc_normalized.get(source_index, 0.0)
        result.append((imp_score, pos_wc_score))
    
    return result


def evaluate_diff(old_scores, new_scores):
    '''
    Evaluate the difference between old and new scores after the method of edits are performed.
    '''
    diff = []
    for (old_imp, old_pos_wc), (new_imp, new_pos_wc) in zip(old_scores, new_scores):
        imp_change = new_imp - old_imp
        pos_wc_change = new_pos_wc - old_pos_wc
        diff.append((imp_change, pos_wc_change))
    return diff


def _extract_citations(sentence: str) -> Tuple[int, ...]:
    """Extract citation numbers from sentence."""
    pattern = re.compile(r"\[(?P<num>\d+)\]")
    numbers = {int(m.group('num')) for m in pattern.finditer(sentence)}
    return tuple(sorted(numbers))


def compute_normalized_importance(sentences: Sequence[str], scores: Sequence[float]) -> Mapping[int, float]:
    """Compute normalized importance per source based on LexRank scores."""
    if len(sentences) != len(scores):
        raise ValueError("Sentences and scores must have the same length.")

    raw_scores = defaultdict(float)
    total_weight = 0.0

    for sentence, score in zip(sentences, scores):
        citations = _extract_citations(sentence)
        if not citations:
            continue
        for src in citations:
            raw_scores[src] += float(score)
        total_weight += float(score) * len(citations)

    if total_weight <= 0.0:
        return {}

    return {src: weight / total_weight for src, weight in raw_scores.items()}


def compute_position_weighted_word_count(sentences: Sequence[str]) -> Mapping[int, float]:
    """Compute position-weighted word count: word_count / sentence_position."""
    citation_pattern = re.compile(r'\[\d+\]')
    weighted_word_counts = defaultdict(float)

    for idx, sentence in enumerate(sentences, start=1):
        citations = _extract_citations(sentence)
        if not citations:
            continue
            
        # Remove citations and count words
        cleaned_sentence = citation_pattern.sub('', sentence).strip()
        word_count = len(cleaned_sentence.split())
        weight = word_count / idx
        
        for src in citations:
            weighted_word_counts[src] += weight

    return dict(weighted_word_counts)


def analyze_response(sentences: Sequence[str], lexrank_scores: Iterable[float]) -> Tuple[Mapping[int, float], Mapping[int, float]]:
    """Compute importance and position-weighted word count metrics."""
    scores_list = list(lexrank_scores)
    importance = compute_normalized_importance(sentences, scores_list)
    position_weighted_wc = compute_position_weighted_word_count(sentences)
    return importance, position_weighted_wc
