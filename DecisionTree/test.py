import numpy as np
from DecisionTree import *

sample = np.random.randn(10, 15)
# print(sample[:, 0])
targets = np.random.randint(0, 2, size=10)

tree = DecisionTree(max_depth=5,
                max_leaves=32,
                min_sample_size_in_leaf=1,
                min_split_sample=2,
                split_metric='ig')

tree.fit(sample, targets)

print(tree.tree.right.left.metric_value, tree.tree.left.left.metric_value)
