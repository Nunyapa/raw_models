import numpy as np
from DecisionTree import *
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.datasets import load_iris, load_diabetes

sample = np.random.randn(100, 15)
# print(sample[:, 0])
targets = np.random.randint(0, 2, size=100)

diabetes_df = load_diabetes()
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

# tree = DecisionTreeClassifier(max_depth=3,
#                     max_leaves=32,
#                     min_sample_size_in_leaf=1,
#                     min_split_sample=2,
#                     split_metric='ig')

# tree.fit(sample, targets)
# print(tree.display_tree())
# print(tree._get_tree_stack(tree._tree))
# print(len(tree._get_tree_stack(tree._tree)))

# preds = tree.predict(sample)
# preds = np.argmax(preds, axis=1)
# print(preds)
# # print()
# print(roc_auc_score(targets, preds))


my_reg_dt = DecisionTreeRegressor(
    max_depth=5,
    # max_leaves=32,
    min_sample_size_in_leaf=30,
    min_split_sample=60,
    split_metric='mse',
    split_type='q'
)

my_reg_dt.fit(diabetes_df['data'], diabetes_df['target'])

print(my_reg_dt.print_tree())
