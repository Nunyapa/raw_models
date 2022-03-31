import numpy as np

def _entropy(probabilities):
    result = [p * np.log(p) for p in probabilities]
    result = -1 * np.sum(result)
    return result

def _information_gain()


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
        amount = p.sum(freqs)
        freqs = freqs / amount
        return uniqs, freqs, amount 

    @staticmethod
    def _split(sorted_mapping, split_value):
        # assert split_value in features_values

        # mapping = list(zip(features_values, target_values))
        # mapping = sorted(mapping, key=lambda x: x[0])
        for idx in range(len(sorted_mapping)):
            if split_value < sorted_mapping[idx][0]:
                break
        return sorted_mapping[:idx], sorted_mapping[idx]

    def _find_best_split(features_values, target_values):
        assert len(features_values) == len(target_values)
        mapping = list(zip(features_values, target_values))
        mapping = sorted(mapping, key=lambda x: x[0])
        for split, _ in mapping:
            left_sample, right_sample = self._split(mapping, split)
            _, left_freqs = self._get_fractions(_get_column_from_map(left_sample, col_idx=1))
            _, right_freqs = self._get_fractions(_get_column_from_map(right_sample, col_idx=1))

            left_entropy = self._entropy(left_freqs)
            right_entropy = self._entropy(right_freqs)


