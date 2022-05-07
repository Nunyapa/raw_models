import numpy as np
from .metrics import *

BIG_CONST = 9999999
SMALL_CONST = -9999999
LEAF_FLAG = -1
UNKNOWN_FLAG = None

METRICS = {
    'ig': InformationGainCriterion,
    'gini': GiniCriterion,
    'mse': MseCriterion,
    'mae': MaeCriterion,
    'rmse': RmseCriterion
}

# METRICS_METHOD_OPTIMIZATION = {
#     'ig': 'max',
#     'gini': 'min',
#     'mse': 'min',
#     'mae': 'min',
#     'rmse': 'min'
# }

# COMPARISON_FUNCTIONS = {
#     'max': lambda old, new: old < new,
#     'min': lambda old, new: old > new
# }

class Node:

    def __init__(self,
                 sample_indexes=None,
                 targets=None,
                 depth=None,
                 node_number=None):
        self.sample_indexes = sample_indexes
        self.targets = targets
        self.depth = depth
        self.node_number = node_number

        self.left = LEAF_FLAG
        self.right = LEAF_FLAG
        self.split_column = None
        self.split_value = None
        self.metric_value = None

    def is_leaf(self):
        return True if self.left is LEAF_FLAG and self.right is LEAF_FLAG else False


class DecisionTree:
    def __init__(self,
                 max_depth=None,
                 max_leaves=None,
                 min_sample_size_in_leaf=None,
                 min_split_sample=None,

                 split_metric='ig',
                 split_type='q'):

        self.min_sample_size_in_leaf = min_sample_size_in_leaf
        self.min_split_sample = min_split_sample
        self.split_metric = split_metric
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.classes = None
        self.split_type = split_type

        self.criterion = METRICS[self.split_metric]()

        self._parameters_initialization()


        # self.metric_method_optimization = METRICS_METHOD_OPTIMIZATION[self.split_metric]
        # self.comparison_function = COMPARISON_FUNCTIONS[self.metric_method_optimization]
        self._best_split_initialization = SMALL_CONST if self.criterion.optimization_way == 'max' else BIG_CONST
        self._tree = None
        self._treestack = []

    def _parameters_initialization(self):
        if self.max_depth is None:
            self.max_depth = 10

        if self.max_leaves is None:
            self.max_leaves = 2 ** self.max_depth

        if self.min_split_sample is None:
            self.min_sample_size_in_leaf = 1

        if self.min_split_sample is None:
            self.min_split_sample = 2 * self.min_sample_size_in_leaf

    @staticmethod
    def _sort_matrix(matrix, sort_by_col=0):
        sorted_matrix = matrix[matrix[:, sort_by_col].argsort()]
        return sorted_matrix

    @staticmethod
    def _split(feature_values, split_value):
        # print(feature_values, split_value)
        left_sample_index = np.where(feature_values <= split_value)[0]
        right_sample_index = np.where(feature_values > split_value)[0]
        return left_sample_index, right_sample_index

    @staticmethod
    def _get_tree_stack(in_node):
        stack = [in_node]
        for node in stack:
            if node == LEAF_FLAG:
                stack.append(UNKNOWN_FLAG)
                stack.append(UNKNOWN_FLAG)
            elif node is None:
                continue
            else:
                stack.append(node.left)
                stack.append(node.right)
        return stack

    def print_tree(self):
        stack = self._get_tree_stack(self._tree)
        thresholds = []
        cols = []
        values = []

        for current_node in stack:
            if isinstance(current_node, Node):
                thresholds.append(current_node.split_value)
                cols.append(current_node.split_column)
                values.append(np.mean(current_node.targets))

        return cols, thresholds, values

    def _get_f_values(self, input_vector):
        '''
        takes an input vector of numbers and if amount of unique values are more than 10
        returns 0.1, 0.2...1 quantile of the vector
        '''
        if self.split_type == 'q':
            split_values = np.histogram(input_vector, bins=10)[1]
        elif self.split_type == 'all':
            split_values = input_vector
        else:
            raise ValueError(f'Wrong split type: {self.split_type}. Try "q" or "all"')
        return split_values

    def _get_best_node(self, sample, targets, indexes):
        best_metric_value = self._best_split_initialization
        best_col_for_split = None
        best_split_value = None

        for col_idx in range(sample.shape[1]):
            col_split_value, col_metric_value = self._find_best_split(sample[indexes, col_idx], targets[indexes])

            if self.criterion.comparison_for_optimization(best_metric_value, col_metric_value):
                best_split_value = col_split_value
                best_metric_value = col_metric_value
                best_col_for_split = col_idx

        left_index, right_index = self._split(sample[indexes, best_col_for_split], best_split_value)
        left_index = indexes[left_index]
        right_index = indexes[right_index]

        return left_index, right_index, best_col_for_split, best_split_value, best_metric_value


    def check_sample_suit(self, depth, sample_indexes, leaves_counter):
        is_sample_suitable = True
        sample_size = len(sample_indexes)

        if depth >= self.max_depth:
            is_sample_suitable = False
        elif sample_size <= self.min_split_sample:
            is_sample_suitable = False
        elif (self.max_leaves - leaves_counter) == 0:
            is_sample_suitable = False

        return is_sample_suitable

    def build_tree(self, sample, targets):
        sample_indexes = np.array(range(sample.shape[0]))
        root = Node(sample_indexes=sample_indexes, targets=targets, depth=0, node_number=0)

        self._treestack.append(root)

        leaves_counter = 1

        for current_node in self._treestack:

            split_params = self._get_best_node(sample,
                                               targets,
                                               current_node.sample_indexes)

            left_index, right_index, best_col, split_value, best_metric = split_params

            current_node.split_column = best_col
            current_node.split_value = split_value
            current_node.metric_value = best_metric

            next_depth = current_node.depth + 1
            parent_node_number = current_node.node_number

            current_node.left = Node(sample_indexes=left_index,
                                     targets=targets[left_index],
                                     depth=next_depth,
                                     node_number=parent_node_number + 1)

            current_node.right = Node(sample_indexes=right_index,
                                      targets=targets[right_index],
                                      depth=next_depth,
                                      node_number=parent_node_number + 2)

            leaves_counter += 2 - 1

            if self.check_sample_suit(next_depth, left_index, leaves_counter):
                self._treestack.append(current_node.left)

            if self.check_sample_suit(next_depth, right_index, leaves_counter):
                self._treestack.append(current_node.right)

        return root

    def _find_best_split(self, feature_values, targets):
        pass

    def fit(self, X_train, y_train):
        pass

    def _node_predict(self, node, sample, indexes):
        pass


    def predict(self, data):
        pass

class DecisionTreeClassifier(DecisionTree):

    def _find_best_split(self, feature_values, targets):
        sorted_matrix = np.dstack([feature_values, targets])[0]
        sorted_matrix = self._sort_matrix(sorted_matrix)

        best_metric_value = self._best_split_initialization
        best_split_value = None

        split_values = self._get_f_values(feature_values)

        for f_value in split_values:
            left_sample_index, right_sample_index = self._split(feature_values, f_value)

            if len(left_sample_index) < self.min_sample_size_in_leaf:
                continue
            elif len(right_sample_index) < self.min_sample_size_in_leaf:
                continue

            cur_metric_value = self.criterion.calculate(sorted_matrix[:, 1],
                                                        sorted_matrix[left_sample_index, 1],
                                                        sorted_matrix[right_sample_index, 1])

            if self.criterion.comparison_for_optimization(best_metric_value, cur_metric_value):
                best_metric_value = cur_metric_value
                best_split_value = f_value

        return best_split_value, best_metric_value

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.criterion.classes = self.classes
        self._tree = self.build_tree(X_train, y_train)

    def _node_predict(self, node, sample, indexes):
        if node.is_leaf():
            probs = get_freqs(node.targets, self.classes)
            probs = probs.reshape((len(self.classes), 1))
            result = (np.ones(sample.shape[0]) * probs).T
            return result, indexes

        left_sample_indexes, right_sample_indexes = self._split(sample[:, node.split_column], node.split_value)

        left_result, left_sample_indexes = self._node_predict(node.left,
                                                              sample[left_sample_indexes],
                                                              left_sample_indexes)

        right_result, right_sample_indexes = self._node_predict(node.right,
                                                                sample[right_sample_indexes],
                                                                right_sample_indexes)

        left_sample_indexes = indexes[left_sample_indexes]
        right_sample_indexes = indexes[right_sample_indexes]

        parent_result = np.concatenate([left_result, right_result])
        parent_indexes = np.concatenate([left_sample_indexes, right_sample_indexes])

        return parent_result, parent_indexes

    def predict(self, data):
        indexes = np.array(list(range(data.shape[0])))
        results, parent_indexes = self._node_predict(self._tree, data, indexes)
        sorted_results = np.column_stack([parent_indexes, results])
        sorted_results = self._sort_matrix(sorted_results, sort_by_col=0)
        return sorted_results[:, 1:]


class DecisionTreeRegressor(DecisionTree):

    def _find_best_split(self, feature_values, targets):
        sorted_matrix = np.dstack([feature_values, targets])[0]
        sorted_matrix = self._sort_matrix(sorted_matrix)

        best_metric_value = self._best_split_initialization
        best_split_value = None

        split_values = self._get_f_values(feature_values)

        for f_value in split_values:
            left_sample_index, right_sample_index = self._split(feature_values, f_value)

            if len(left_sample_index) < self.min_sample_size_in_leaf:
                continue
            elif len(right_sample_index) < self.min_sample_size_in_leaf:
                continue

            parent_predict_value = np.array([sorted_matrix[:, 1].mean()])
            left_predict_value = np.array([sorted_matrix[left_sample_index, 1].mean()])
            right_predict_value = np.array([sorted_matrix[right_sample_index, 1].mean()])

            parent_metric = self.criterion.calculate(sorted_matrix[:, 1], parent_predict_value)
            left_metric = self.criterion.calculate(sorted_matrix[left_sample_index, 1], left_predict_value)
            right_metric = self.criterion.calculate(sorted_matrix[right_sample_index, 1], right_predict_value)

            left_metric = left_metric * len(left_sample_index) / len(feature_values)
            right_metric = right_metric * len(left_sample_index) / len(feature_values)

            mean_l_r_metric = left_metric + right_metric

            if self.criterion.comparison_for_optimization(best_metric_value, mean_l_r_metric):
                best_metric_value = mean_l_r_metric
                best_split_value = f_value

        return best_split_value, best_metric_value

    def fit(self, X_train, y_train):
        self._tree = self.build_tree(X_train, y_train)

    def _node_predict(self, node, sample, indexes):
        if node.is_leaf():
            result = np.ones(sample.shape[0]) * node.targets.mean()
            return result, indexes

        left_sample_indexes, right_sample_indexes = self._split(sample[:, node.split_column], node.split_value)

        left_result, left_sample_indexes = self._node_predict(node.left,
                                                              sample[left_sample_indexes],
                                                              left_sample_indexes)

        right_result, right_sample_indexes = self._node_predict(node.right,
                                                                sample[right_sample_indexes],
                                                                right_sample_indexes)

        left_sample_indexes = indexes[left_sample_indexes]
        right_sample_indexes = indexes[right_sample_indexes]

        # print('LR', left_result, right_result)
        parent_result = np.concatenate([left_result, right_result])
        parent_indexes = np.concatenate([left_sample_indexes, right_sample_indexes])

        return parent_result, parent_indexes

    def predict(self, data):
        indexes = np.array(list(range(data.shape[0])))

        results, parent_indexes = self._node_predict(self._tree, data, indexes)
        # print(len(results), len(parent_indexes))
        sorted_results = np.column_stack([parent_indexes, results])
        sorted_results = self._sort_matrix(sorted_results, sort_by_col=0)
        return sorted_results[:, 1:]
