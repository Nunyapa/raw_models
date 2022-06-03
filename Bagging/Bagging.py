import numpy as np
from DecisionTree.DecisionTree import DecisionTreeClassifier


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
                 features_size_ratio,
                 random_state=42):
        # TODO: add early_stopping_rounds and maybe another important params.

        self.tree_params = tree_params
        self.iterators = iterations
        self.bagging_size_ratio = bagging_size_ratio
        self.features_size_ratio = features_size_ratio
        self.random_state = random_state

        self.trees_info = None

    def sample_with_replacement(self, X, y):
        features_size = round(self.features_size_ratio * X.shape[1])
        features_size = 1 if features_size == 0 else features_size

        sub_features = np.random.RandomState(self.random_state).choice(np.array(range(X.shape[1])),
                                                                   size=features_size,
                                                                   replace=False)

        parent_sample = np.dstack([X[:, sub_features], y])[0]
        sub_sample_size = round(self.bagging_size_ratio * X.shape[0])

        sub_sample = np.random.RandomState(self.random_state).choice(parent_sample,
                                                                     size=sub_sample_size,
                                                                     replace=True)

        sub_sample_info = {
            'features': sub_features
        }

        sub_sample_X = sub_sample[:, :-1]
        sub_sample_y = sub_sample[:, -1]

        return sub_sample_X, sub_sample_y, sub_sample_info

    def predict(self, X):
        if self.trees_info is None:
            raise ValueError('self.trees_info is None. Call fit first')

        iterators_list = range(self.iterators)
        results = []
        for tree_idx in iterators_list:
            tree = self.trees_info[tree_idx]['tree']
            sub_sample_info = self.trees_info[tree_idx]['tree_info']
            sub_features = sub_sample_info['features']

            result = tree.predict(X[:, sub_features])

            results.append(result)

        results = np.array([results])
        results = np.median(results, axis=1)

        return results


    def compute_metric(self):
        pass

    def _fit(self, X, y, eval_set=(None, None)):
        self.trees_info = {}
        iterators_list = range(self.iterators)

        for tree_idx in iterators_list:
            current_tree = DecisionTreeClassifier(**self.tree_params)
            sub_sample_X, sub_sample_y, sub_sample_info = self.sample_with_replacement(X, y)

            current_tree.fit(sub_sample_X, sub_sample_y)


            tree_info_dict = {
                'tree': current_tree,
                'tree_info': sub_sample_info
            }

            self.trees_info[tree_idx] = tree_info_dict


    def fit(self, X, y, eval_set=(None, None)):
        # TODO: check for eval set
        # TODO: Want to stop training if self.early_stopping steps was reached without metric improvement.
        pass
