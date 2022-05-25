import numpy as np


class BaggingClass:
    '''
    Bagging class.
    Bagging is bootstrap aggregation.

    Builds an ensemble of decision trees which trains on training sets each of size N' and
    all of them are sampling from the main Sample of size N. Each DT is training on a limit amount of features.
    '''

    def __init__(self):
        pass