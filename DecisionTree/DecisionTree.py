from hashlib import new
import numpy as np
from typing import Iterable, Mapping, Tuple, TypeVar
import metrics as m

# TODO LIST
# TODO : Prune tree
# TODO : make regression metrics.
# TODO : optimize sample splitting.



T = TypeVar('T', int, float, str)

List = TypeVar('List', list, np.array)

BIG_CONST = 9999999
SMALL_CONST = -9999999


METRICS = {
    'ig': m.information_gain,
    'gini': m.gini_impurity
}

METRICS_METHOD_OPTIMIZATION = {
    'ig': 'max',
    'gini': 'min'
}

COMPARISON_FUNCTIONS = {
    'max': lambda old, new: old < new,
    'min': lambda old, new: old > new
}


class Node:

    def __init__(self,
                 sample=None,
                 targets=None,
                 depth=None
                 ):
        self.sample = sample
        self.targets = targets
        self.depth = depth

        self.left = None
        self.right = None
        self.split_column = None
        self.split_value = None
        self.metric_value = None


class DecisionTree:

    def __init__(self,
                max_depth=None,
                max_leaves=None,
                min_sample_size_in_leaf=None,
                min_split_sample=None,
                split_metric='ig'):
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_sample_size_in_leaf = min_sample_size_in_leaf
        self.min_split_sample = min_split_sample
        self.split_metric = split_metric
        self.metric_function = METRICS[self.split_metric]
        self.metric_method_optimization = METRICS_METHOD_OPTIMIZATION[self.split_metric]
        self.comparison_function = COMPARISON_FUNCTIONS[self.metric_method_optimization]
        self._splits = {}
        self._split_results = {}
        self._best_split_initialization = SMALL_CONST if self.metric_method_optimization == 'max' else BIG_CONST
        self.tree = None

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
    def _split(sorted_mapping, split_value):
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

            if self.comparison_function(best_metric_value, col_metric_value):
                col_best_split_value = col_split_value
                best_metric_value = col_metric_value
                best_col_for_split = col_idx

        return best_col_for_split, col_best_split_value, best_metric_value

    def _find_best_split(self, feature_values, target_values):
        assert len(feature_values) == len(target_values)

        mapping = self._sorted_mapping(feature_values, target_values)
        best_metric_value = self._best_split_initialization
        best_split_value = None

        for f_value, _ in mapping:
            left_sample, right_sample = self._split(mapping, f_value)
            parent_freqs = m.get_fractions(mapping[:, 1])
            left_freqs = m.get_fractions(left_sample[:, 1])
            right_freqs = m.get_fractions(right_sample[:, 1])

            cur_metric_value = self.metric_function(parent_freqs, left_freqs, right_freqs)
            if self.comparison_function(best_metric_value, cur_metric_value):
                best_metric_value = cur_metric_value
                best_split_value = f_value

        return best_split_value, best_metric_value



    def build_tree(self, sample, targets):
        stack = []
        root = Node(sample=sample, targets=targets, depth=1)
        stack.append(root)

        while stack:
            current_node = stack.pop()
            split_params = self._get_best_node(current_node.sample, current_node.targets)
            best_col, split_value, best_metric = split_params
            current_node.split_column = best_col
            current_node.split_value = split_value
            current_node.metric_value = best_metric
            if current_node.sample.shape[0] < self.min_split_sample:
                continue

            left_sample, left_targets, right_sample, right_targets = self._split_matrix(current_node.sample,
                                                                                        current_node.targets,
                                                                                        best_col,
                                                                                        split_value)

            current_node.left = Node(sample=left_sample, targets=left_targets, depth=current_node.depth + 1)
            current_node.right = Node(sample=right_sample, targets=right_targets, depth=current_node.depth + 1)

            if current_node.depth < self.max_depth:
                stack.append(current_node.left)
                stack.append(current_node.right)

        return root

    def prune_tree(self):
        pass

    def fit(self, X_train, y_train):
        self.tree = self.build_tree(X_train, y_train)
