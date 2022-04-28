import numpy as np
# from typing import Iterable, Mapping, TypeVar, Tuple


def get_freqs(values, classes):
    uniqs, freqs = np.unique(values, return_counts=True)
    for c in range(len(classes)):
        if classes[c] not in uniqs:
            freqs = np.insert(freqs, c, 0)
    total_amount = np.sum(freqs)
    freqs = freqs / total_amount
    return freqs


def entropy(probabilities):
    result = [p * np.log(p) for p in probabilities if p != 0]
    result = -1 * np.sum(result)
    return result


def information_gain(parent_probs, left_probs, right_probs):
    ig = entropy(parent_probs)
    for child_prob in (left_probs, right_probs):
        ig -= len(child_prob) * entropy(child_prob) / len(parent_probs)
    return ig


def gini_impurity(parent_probs, left_probs, right_probs):
    left_sample_size = len(left_probs)
    right_sample_size = len(right_probs)
    parent_size = len(parent_probs)

    left_gini = 2 * left_probs.prod() / left_sample_size
    right_gini = 2 * right_probs.prod() / right_sample_size
    gini = (left_sample_size / parent_size) * left_gini + (right_sample_size / parent_size) * right_gini

    return gini


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

def mae(y_true, y_pred):
    assert y_pred.shape[0] == 1 or y_pred.shape[0] == y_true.shape[0]
    # predict_value = np.mean(targets)
    errors = np.subtract(y_true, y_pred)
    return np.sum(np.abs(errors)) / y_true.shape[0]


def mse(y_true, y_pred):
    assert y_pred.shape[0] == 1 or y_pred.shape[0] == y_true.shape[0]
    # predict_value = np.mean(targets)
    errors = np.subtract(y_true, y_pred)
    return np.sum(np.power(errors, 2)) / y_true.shape[0]


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))




