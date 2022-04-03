from hashlib import new
import numpy as np
from typing import Iterable, Mapping, Tuple, TypeVar
import DecisionTree.metrics as m

T = TypeVar('T', int, float, str)

List = TypeVar('List', list, np.array)

# TODO : move all metrics to another file
# TODO : make regression metrics
#
BIG_CONST = 9999999
SMALL_CONST = -9999999


METRICS = {
    'ig': m._information_gain,
    'gini': m._gini_impurity
}

METRICS_METHOD_OPTIMIZATION = {
    'ig': 'max',
    'gini': 'min'
}

COMPARISON_FUNCTIONS = {
    'max' : lambda old, new: old < new,
    'min' : lambda old, new: old > new
}

class DecisionTree():

    def __init__(self, 
                depth=None,
                max_leaves=None,
                max_samples_in_leaf=None,
                min_split_samples=None,
                split_metric='ig'):
        self.depth = depth
        self.max_leaves = max_leaves
        self.max_samples_in_leaf = max_samples_in_leaf
        self.min_split_samples = min_split_samples
        self.split_metric = split_metric
        self.metric_function = METRICS[self.split_metric] 
        self.metric_method_optimization = METRICS_METHOD_OPTIMIZATION[self.split_metric]
        self.comparions_function = COMPARISON_FUNCTIONS[self.metric_method_optimization]
        self._splits = {}
        self._split_results = {}
        self._best_split_initialization = SMALL_CONST if self.metric_method_optimization == 'max' else BIG_CONST
    
    @staticmethod
    def _get_column_from_map(mapping, col_idx):
        column = [i[col_idx] for i in mapping]
        return column

    @staticmethod
    def _sorted_mapping(feature_values, target_values):
        nrows = feature_values.shape[0]
        mapping = np.zeros(shape=(nrows, 2))
        mapping[:, 0] = feature_values
        mapping[:, 1] = target_values
        mapping = mapping[mapping[:, 0].argsort()]
        return mapping

    @staticmethod
    def _split(sorted_mapping: Mapping[T, T], split_value: T):
        left_sample = sorted_mapping[sorted_mapping[:, 0] <= split_value]
        right_sample = sorted_mapping[sorted_mapping[:, 0] > split_value]
        return left_sample, right_sample

    @staticmethod
    def _split_matrix(sample, targets, split_column, split_value):
        # print('_split_matrix sample ', sample)
        nrows, ncols = sample.shape
        new_sample = np.zeros(shape=(nrows, ncols + 1))
        new_sample[:, :-1] = sample
        new_sample[:, -1] = targets
        left_sample = new_sample[new_sample[:, split_column] <= split_value]
        left_targets = left_sample[:, -1]
        left_sample = left_sample[:, :-1]
        right_sample = new_sample[new_sample[:, split_column] > split_value]
        right_targets = right_sample[:, -1]
        right_sample = right_sample[:, :-1]
        return left_sample, left_targets, right_sample, right_targets

    def _get_best_node(self, sample, targets):
        nrof_columns = sample.shape[1]

        best_metric_value = self._best_split_initialization
        best_col_for_split = None
        col_best_split_value = None

        for col_idx in range(nrof_columns):
            col_split_value, col_metric_value = self._find_best_split(sample[:, col_idx], targets)

            if self.comparions_function(best_metric_value, col_metric_value):
                col_best_split_value = col_split_value
                best_metric_value = col_metric_value
                best_col_for_split = col_idx

        return best_col_for_split, col_best_split_value, best_metric_value

    def _find_best_split(self, feature_values: Iterable[float], target_values: Iterable[float]):
        assert len(feature_values) == len(target_values)

        mapping = self._sorted_mapping(feature_values, target_values)
        # print(mapping)
        best_metric_value = self._best_split_initialization
        best_split_value = None

        # TODO: optimize a split choosing value 
        for f_value, _ in mapping:
            left_sample, right_sample = self._split(mapping, f_value)
            # print(left_sample, right_sample)
            _, parent_freqs, _  = m._get_fractions(mapping[:, 1])
            _, left_freqs, _ = m._get_fractions(left_sample[:, 1])
            _, right_freqs, _  = m._get_fractions(right_sample[:, 1])
            # print('parent_freqs: ', parent_freqs)
            # print('left_freqs: ', left_freqs)
            # print('right_freqs: ', parent_freqs)
            cur_metric_value = self.metric_function(parent_freqs, left_freqs, right_freqs)
            if self.comparions_function(best_metric_value, cur_metric_value):
                best_metric_value = cur_metric_value
                best_split_value = f_value
        
        return best_split_value, best_metric_value

    def build_tree(self, sample, targets, depth, side):
        next_depth = depth + 1
        if depth == self.depth:
            self._splits[f'{depth}_{side}_leaf'] = np.median(targets)
            return 0

        best_col_for_split, col_best_split_value, best_metric_value = self._get_best_node(sample, targets)
        left_sample, left_targets, right_sample, right_targets = self._split_matrix(sample, targets, best_col_for_split, col_best_split_value)
        if left_sample.shape[0] > 1:
            self.build_tree(left_sample, left_targets, next_depth, 'left')
        # else:


        if right_sample.shape[0] > 1:
            self.build_tree(right_sample, right_targets, next_depth, 'right')

        split_name = f'{depth}_{side}_{best_col_for_split}'
        self._splits[split_name] = col_best_split_value
        self._split_results[split_name] = best_metric_value

    def fit(self, X_train, y_train):
        self.build_tree(X_train, y_train, 0, 'root')
        return self._splits, self._split_results