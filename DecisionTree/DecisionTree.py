# from hashlib import new
import numpy as np
# from typing import Iterable, Mapping, Tuple, TypeVar
import metrics as m

# TODO LIST
# TODO : DecisionTreeBase class, DecisionTreeClassifier class, DecisionTreeRegressor class
# TODO : Prune tree
# TODO : optimize sample splitting.


# T = TypeVar('T', int, float, str)
#
# List = TypeVar('List', list, np.array)

BIG_CONST = 9999999
SMALL_CONST = -9999999

METRICS = {
    'ig': m.information_gain,
    'gini': m.gini_impurity,
    'mse': m.mse,
    'mae': m.mae,
    'rmse': m.rmse
}

METRICS_METHOD_OPTIMIZATION = {
    'ig': 'max',
    'gini': 'min',
    'mse': 'min',
    'mae': 'min',
    'rmse': 'min'
}

COMPARISON_FUNCTIONS = {
    'max': lambda old, new: old < new,
    'min': lambda old, new: old > new
}


class Node:

    def __init__(self,
                 sample_indexes=None,
                 targets=None,
                 depth=None):
        self.sample_indexes = sample_indexes
        self.targets = targets
        self.depth = depth

        self.left = None
        self.right = None
        self.split_column = None
        self.split_value = None
        self.metric_value = None

    def is_leaf(self):
        return True if self.left is None and self.right is None else False


class DecisionTree:

    def __init__(self,
                 max_depth=None,
                 max_leaves=None,
                 min_sample_size_in_leaf=None,
                 min_split_sample=None,
                 split_metric='ig'):

        self.min_sample_size_in_leaf = min_sample_size_in_leaf
        self.min_split_sample = min_split_sample
        self.split_metric = split_metric
        self.max_leaves = max_leaves
        self.max_depth = max_depth

        self.metric_function = METRICS[self.split_metric]
        self.metric_method_optimization = METRICS_METHOD_OPTIMIZATION[self.split_metric]
        self.comparison_function = COMPARISON_FUNCTIONS[self.metric_method_optimization]
        self._best_split_initialization = SMALL_CONST if self.metric_method_optimization == 'max' else BIG_CONST
        self.tree = None

    @staticmethod
    def _sort_matrix(matrix, sort_by_col=0):
        sorted_matrix = matrix[matrix[:, sort_by_col].argsort()]
        return sorted_matrix

    @staticmethod
    def _split(feature_values, split_value):
        left_sample_index = np.where(feature_values <= split_value)[0]
        right_sample_index = np.where(feature_values > split_value)[0]
        return left_sample_index, right_sample_index


    def _find_best_split(self, feature_values, targets):
        # TODO : this is not correct implementation of the method we need to pass targets into m.fraction_method
        # to do this we need to concatenate feature_values, targets and indexes then sort them by features values (sorting is needed to
        # use more efficient method of splitting) than pass the sorted matrix into the "drop" method of choosing the best
        # f_value. After that we are ready to split our Nx3 shape matrix into a split function

        sorted_matrix = np.dstack([feature_values, targets])[0]
        sorted_matrix = self._sort_matrix(sorted_matrix)

        best_metric_value = self._best_split_initialization
        best_split_value = None

        # TODO : optimize choosing f_value
        for f_value in feature_values:

            left_sample_index, right_sample_index = self._split(feature_values, f_value)

            if len(left_sample_index) < self.min_sample_size_in_leaf:
                continue
            elif len(right_sample_index) < self.min_sample_size_in_leaf:
                continue

            parent_freqs = m.get_fractions(sorted_matrix[1])
            left_freqs = m.get_fractions(sorted_matrix[left_sample_index, 1])
            right_freqs = m.get_fractions(sorted_matrix[right_sample_index, 1])

            cur_metric_value = self.metric_function(parent_freqs, left_freqs, right_freqs)

            if self.comparison_function(best_metric_value, cur_metric_value):
                best_metric_value = cur_metric_value
                best_split_value = f_value

        return best_split_value, best_metric_value

    def _get_best_node(self, sample, targets, indexes):
        nrof_columns = sample.shape[1]

        best_metric_value = self._best_split_initialization
        best_col_for_split = None
        best_split_value = None

        for col_idx in range(nrof_columns):
            col_split_value, col_metric_value = self._find_best_split(sample[indexes, col_idx], targets[indexes])

            if self.comparison_function(best_metric_value, col_metric_value):
                best_split_value = col_split_value
                best_metric_value = col_metric_value
                best_col_for_split = col_idx

        left_index, right_index = self._split(sample[indexes, best_col_for_split], best_split_value)
        left_index = indexes[left_index]
        right_index = indexes[right_index]

        return left_index, right_index, best_col_for_split, best_split_value, best_metric_value

    def check_sample_suit(self, depth, sample_indexes):
        is_sample_suitable = 1
        sample_size = len(sample_indexes)

        if depth >= self.max_depth:
            is_sample_suitable = 0
        elif sample_size <= self.min_split_sample:
            is_sample_suitable = 0

        return is_sample_suitable

    def display_tree(self):
        stack = [self.tree]
        thresholds = []
        cols = []
        values = []

        while stack:
            current_node = stack.pop()
            thresholds.append(current_node.split_value)
            cols.append(current_node.split_column)
            values.append(np.mean(current_node.targets))
            if not current_node.is_leaf():
                stack.append(current_node.left)
                stack.append(current_node.right)

        return cols, thresholds, values

    def build_tree(self, sample, targets):
        stack = []
        sample_indexes = np.array(range(sample.shape[0]))
        root = Node(sample_indexes=sample_indexes, targets=targets, depth=1)
        stack.append(root)

        while stack:
            current_node = stack.pop()
            split_params = self._get_best_node(sample, targets, current_node.sample_indexes)
            left_index, right_index, best_col, split_value, best_metric = split_params

            current_node.split_column = best_col
            current_node.split_value = split_value
            current_node.metric_value = best_metric

            next_depth = current_node.depth + 1

            current_node.left = Node(sample_indexes=left_index,
                                     targets=targets[left_index],
                                     depth=next_depth)

            current_node.right = Node(sample_indexes=right_index,
                                      targets=targets[right_index],
                                      depth=next_depth)

            if self.check_sample_suit(next_depth, current_node.left.sample_indexes):
                stack.append(current_node.left)

            if self.check_sample_suit(next_depth, current_node.right.sample_indexes):
                stack.append(current_node.right)

        return root

    def prune_tree(self):
        pass

    def fit(self, X_train, y_train):
        self.tree = self.build_tree(X_train, y_train)
