import numpy as np
from typing import Iterable, Mapping, TypeVar

T = TypeVar('T')

# TODO : move all metrics to another file
# TODO : make regression metrics
# 


class DecisionTree():

    def __init__(self, 
                depth=None,
                max_leaves=None,
                max_samples_in_leaf=None,
                min_split_samples=None):
        self.depth = depth
        self.max_leaves = max_leaves
        self.max_samples_in_leaf = max_samples_in_leaf
        self.min_split_samples = min_split_samples
    
    @staticmethod
    def _get_column_from_map(mapping, col_idx):
        column = [i[col_idx] for i in mapping]
        return column

    @staticmethod
    def _get_fractions(values):
        uniqs, freqs = np.unique(values, return_counts=True)
        amount = np.sum(freqs)
        freqs = freqs / amount
        return uniqs, freqs, amount 

    @staticmethod
    def _split(sorted_mapping: Mapping[T, T], split_value: T):
        for idx in range(len(sorted_mapping)):
            if split_value < sorted_mapping[idx][0]:
                break
        return sorted_mapping[:idx], sorted_mapping[idx]

    @staticmethod
    def _entropy(probabilities: Iterable[float]) -> float:
        result = [p * np.log(p) for p in probabilities]
        result = -1 * np.sum(result)
        return result

    def _gini_impurity(self, left_probs: Iterable[float],  right_probs: Iterable[float]):
        left_probs_size = len(left_probs)
        right_probs_size = len(right_probs)
        size = left_probs_size + right_probs_size

        left_gini = 2 * left_probs / left_probs_size
        right_gini = 2 * right_probs / right_probs_size
        gini_impurity = (left_probs_size / size) * left_gini + (right_probs_size / size) * right_gini
        
        return gini_impurity
        
    def _information_gain(self, parent_probs: Iterable[float], children_probs: Iterable[Iterable[float], Iterable[float]]) -> float:
        ig = self._entropy(parent_probs)
        for child_prob in children_probs:
            ig -= len(child_prob) * self._entropy(child_prob) / len(parent_probs)
        return ig

    def _find_best_split(self, features_values: Iterable[float], target_values: Iterable[float]):
        assert len(features_values) == len(target_values)
        mapping = list(zip(features_values, target_values))
        mapping = sorted(mapping, key=lambda x: x[0])
        best_ig = -1
        best_split = None
        for f_value, _ in mapping:
            left_sample, right_sample = self._split(mapping, f_value)
            _, _, common_freqs = self._get_fractions(self._get_column_from_map(mapping, col_idx=1))
            _, _, left_freqs = self._get_fractions(self._get_column_from_map(left_sample, col_idx=1))
            _, _, right_freqs = self._get_fractions(self._get_column_from_map(right_sample, col_idx=1))
            ig = self._information_gain(common_freqs, (left_freqs, right_freqs))
            if ig > best_ig:
                best_ig = ig
                best_split = f_value
        
        return best_split, ig

