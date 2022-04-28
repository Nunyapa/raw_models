import numpy as np
from DecisionTree import *
from sklearn.metrics import mean_squared_error, roc_auc_score

sample = np.random.randn(100, 15)
# print(sample[:, 0])
targets = np.random.randint(0, 2, size=100)

# tree = DecisionTreeRegressor(max_depth=5,
#                     max_leaves=32,
#                     min_sample_size_in_leaf=1,
#                     min_split_sample=2,
#                     split_metric='mse')
# tree.fit(sample, targets)
# print(tree.display_tree())
# print(tree._get_tree_stack(tree.tree_))
# print(len(tree._get_tree_stack(tree.tree_)))
#
# preds = tree.predict(sample).T
# print(preds[0])
# print(mean_squared_error(targets, preds[0]))

tree = DecisionTreeClassifier(max_depth=3,
                    max_leaves=32,
                    min_sample_size_in_leaf=1,
                    min_split_sample=2,
                    split_metric='ig')

tree.fit(sample, targets)
print(tree.display_tree())
print(tree._get_tree_stack(tree.tree_))
print(len(tree._get_tree_stack(tree.tree_)))

preds = tree.predict(sample)
preds = np.argmax(preds, axis=1)
print(preds)
# print()
print(roc_auc_score(targets, preds))
