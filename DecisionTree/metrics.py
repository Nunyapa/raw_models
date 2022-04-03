import numpy as np
from typing import Iterable, Mapping, TypeVar, Tuple

T = TypeVar('T', int, float, str)
List = TypeVar('List', list, np.array)


def _get_fractions(values: Iterable[T]) -> Tuple[List, List, int]:
    uniqs, freqs = np.unique(values, return_counts=True)
    total_amount = np.sum(freqs)
    freqs = freqs / total_amount
    return uniqs, freqs, total_amount

def _entropy(probabilities: Iterable[float]) -> float:
    result = [p * np.log(p) for p in probabilities]
    result = -1 * np.sum(result)
    return result

def _information_gain(parent_probs: Iterable[float], left_probs: Iterable[float],  right_probs: Iterable[float]) -> float:
    # print(parent_probs, left_probs, right_probs)
    ig = _entropy(parent_probs)
    for child_prob in (left_probs, right_probs):
        ig -= len(child_prob) * _entropy(child_prob) / len(parent_probs)
    return ig

def _gini_impurity(parent_probs: Iterable[float], left_probs: np.array,  right_probs: np.array):
    left_sample_size = len(left_probs)
    right_sample_size = len(right_probs)
    parent_size = len(parent_probs)

    left_gini = 2 * left_probs.prod() / left_sample_size
    right_gini = 2 * right_probs.prod() / right_sample_size
    gini_impurity = (left_sample_size / parent_size) * left_gini + (right_sample_size / parent_size) * right_gini
    
    return gini_impurity

