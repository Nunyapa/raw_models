import numpy as np


class BaggingClass:
    '''
    Bagging class.
    Bagging is bootstrap aggregation.

    Builds an ensemble of decision trees which trains on training sets each of size N' and
    all of them are sampling from the main Sample of size N. Each DT is training on a limit amount of features.
    '''

    def __init__(self,
                 tree_params,
                 iterations,
                 bagging_size_ratio,
                 features_size_ratio):

        self.tree_params = tree_params
        self.iterators = iterations
        self.bagging_size_ratio = bagging_size_ratio
        self.features_size_ratio = features_size_ratio

    def sample_with_replacement(self, X, y):
        features_size = round(self.features_size_ratio * X.shape[1])
        features_size = 1 if features_size == 0 else features_size

        features = np.random.choice(np.array(range(X.shape[1])), size=features_size, replace=False)

        parent_sample = np.dstack([X[:, features], y])[0]
        sub_sample_size = round(self.bagging_size_ratio * X.shape[0])

        sub_sample = np.random.choice(parent_sample, size=sub_sample_size, replace=True)
        sub_sample_X = sub_sample[:, :-1]
        sub_sample_y = sub_sample[:, -1]

        return sub_sample_X, sub_sample_y

    def fit(self, X, y, eval_set=(None, None)):
        # TODO: check for eval set
        # TODO: Voting system. (multiple trees will yeild predicts and its needed to weigh them all in order to get one array of predicts)
        pass
