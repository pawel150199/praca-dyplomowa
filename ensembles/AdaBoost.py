import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import base
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone

class AdaBoostClassifier(DecisionTreeClassifier, ClassifierMixin):
    def __init__(self, n_estimators=50, base_estimator=None):
        self.n_estimators = n_estimators
        self.models = [None]*n_estimators
        if base_estimator == None:
            base_estimator = DecisionTreeClassifier(max_depth=1)
        self.base_estimator = base_estimator
        self.estimator_errors_ = []
        
    def __index_to_vector(y,k,labelDict):



