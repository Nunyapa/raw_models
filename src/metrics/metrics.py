import numpy as np
from abc import ABC, abstractmethod


def get_freqs(values, classes):
    uniqs, freqs = np.unique(values, return_counts=True)
    for c in range(len(classes)):
        if classes[c] not in uniqs:
            freqs = np.insert(freqs, c, 0)
    total_amount = np.sum(freqs)
    freqs = freqs / total_amount
    return freqs


class BaseClassificationCriterion:

    def __init__(self):
        self._classes = None

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, classes):
        if len(classes) > 0 and isinstance(classes, (list, np.ndarray)):
            self._classes = classes
        elif isinstance(classes, int):
            self._classes = list(range(classes))
        else:
            print("Please enter a valid number or list of classes")

    def calculate_probabilities(self, sample):
        return get_freqs(sample, self._classes)

    def calculate(self, parent_sample, child_l_sample, child_r_sample):
        pass


class BaseRegressionCriterion:
    pass


# class EntropyCriterion(BaseClassificationCriterion):
#
#     def __init__(self):
#         self.optimization_way = 'min'
#
#     def calculate_probabilities(self, sample, classes):
#         uniqs, freqs = np.unique(sample, return_counts=True)
#         for c in range(len(classes)):
#             if classes[c] not in uniqs:
#                 freqs = np.insert(freqs, c, 0)
#         total_amount = np.sum(freqs)
#         freqs = freqs / total_amount
#         return freqs
#
#     def calculate(self, sample, classes):
#         probabilities = self.calculate_probabilities(sample, classes)
#         result = [p * np.log2(p) for p in probabilities if p != 0]
#         result = -1 * np.sum(result)
#
#         return result
#
#     def optimization_comparison_method(self, old, new):
#         return old > new

class InformationGainCriterion(BaseClassificationCriterion):

    def __init__(self):
        super(InformationGainCriterion, self).__init__()

        self.optimization_way = 'max'

    def _entropy(self, sample):
        probabilities = self.calculate_probabilities(sample)
        result = [p * np.log2(p) for p in probabilities if p != 0]
        result = -1 * np.sum(result)
        return result

    def calculate(self, parent_sample, child_l_sample, child_r_sample):
        ig = self._entropy(parent_sample)
        for child_prob in (child_l_sample, child_r_sample):
            ig -= len(child_prob) * self._entropy(child_prob) / len(parent_sample)
        return ig

    @staticmethod
    def comparison_for_optimization(old, new):
        return old < new


class GiniCriterion(BaseClassificationCriterion):

    def __init__(self):
        super(GiniCriterion, self).__init__()

        self.optimization_way = 'min'

    @staticmethod
    def comparison_for_optimization(old, new):
        return old > new

    def calculate(self, parent_sample, child_l_sample, child_r_sample):
        left_sample_size = len(parent_sample)
        right_sample_size = len(child_l_sample)
        parent_size = len(child_r_sample)

        left_probs = self.calculate_probabilities(child_l_sample)
        right_probs = self.calculate_probabilities(child_r_sample)

        left_gini = 2 * left_probs.prod() / left_sample_size
        right_gini = 2 * right_probs.prod() / right_sample_size
        gini = (left_sample_size / parent_size) * left_gini + (right_sample_size / parent_size) * right_gini

        return gini


class MaeCriterion(BaseRegressionCriterion):

    def __init__(self):
        super(MaeCriterion, self).__init__()

        self.optimization_way = 'min'

    @staticmethod
    def comparison_for_optimization(old, new):
        return old > new

    @staticmethod
    def calculate(y_true, y_pred):
        assert y_pred.shape[0] == 1 or y_pred.shape[0] == y_true.shape[0]

        errors = np.subtract(y_true, y_pred)
        return np.sum(np.abs(errors)) / y_true.shape[0]


class MseCriterion(BaseRegressionCriterion):

    def __init__(self):
        super(MseCriterion, self).__init__()

        self.optimization_way = 'min'

    @staticmethod
    def comparison_for_optimization(old, new):
        return old > new

    @staticmethod
    def calculate(y_true, y_pred):
        assert y_pred.shape[0] == 1 or y_pred.shape[0] == y_true.shape[0]
        # predict_value = np.mean(targets)
        errors = np.subtract(y_true, y_pred)
        return np.sum(np.power(errors, 2)) / y_true.shape[0]


class RmseCriterion(MseCriterion):

    def __init__(self):
        super(MseCriterion, self).__init__()

    @staticmethod
    def calculate(y_true, y_pred):
        result = super().calculate(y_true, y_pred)
        return np.sqrt(result)



#
#
# def entropy(probabilities):
#     result = [p * np.log2(p) for p in probabilities if p != 0]
#     result = -1 * np.sum(result)
#     return result
#
#
# def information_gain(parent_probs, left_probs, right_probs):
#     ig = entropy(parent_probs)
#     for child_prob in (left_probs, right_probs):
#         ig -= len(child_prob) * entropy(child_prob) / len(parent_probs)
#     return ig


# def gini_impurity(parent_probs, left_probs, right_probs):
#     left_sample_size = len(left_probs)
#     right_sample_size = len(right_probs)
#     parent_size = len(parent_probs)
#
#     left_gini = 2 * left_probs.prod() / left_sample_size
#     right_gini = 2 * right_probs.prod() / right_sample_size
#     gini = (left_sample_size / parent_size) * left_gini + (right_sample_size / parent_size) * right_gini
#
#     return gini


# def mape(targets):
#     predict_value = np.mean(targets)
#     errors = np.subtract(targets, predict_value)
#     precentages = np.divide(np.abs(errors), targets) * 100
#     return precentages / targets.shape[0]


# def error_reduction(metric_func, parent_probs, left_probs, right_probs):
#     parent_metric = metric_func(parent_predict_value, sorted_matrix[:, 1])
#     left_metric = metric_func(left_predict_value, sorted_matrix[left_sample_index, 1])
#     right_metric = metric_func(right_predict_value,sorted_matrix[right_sample_index, 1])
#
#     return np.sum(np.abs(errors)) / y_true.shape[0]

# def mae(y_true, y_pred):
#     assert y_pred.shape[0] == 1 or y_pred.shape[0] == y_true.shape[0]
#     # predict_value = np.mean(targets)
#     errors = np.subtract(y_true, y_pred)
#     return np.sum(np.abs(errors)) / y_true.shape[0]
#
#
# def mse(y_true, y_pred):
#     assert y_pred.shape[0] == 1 or y_pred.shape[0] == y_true.shape[0]
#     # predict_value = np.mean(targets)
#     errors = np.subtract(y_true, y_pred)
#     return np.sum(np.power(errors, 2)) / y_true.shape[0]
#
#
# def rmse(y_true, y_pred):
#     return np.sqrt(mse(y_true, y_pred))




