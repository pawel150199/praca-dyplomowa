import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone 
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode 



"""AdaBoost - Przykładowa implementacja Boostingu"""

class AdaBoostClassifier(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=5, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        np.random.seed(self.random_state)
    
    def fit(self, X, y):
        """Trening"""
        self.estimators_ = []

    def _boost(self, iboost, X, y, sample_weights, random_state):
        """Pojedynczy boost"""



    
