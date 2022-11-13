import numpy as np
from scipy.stats import mode
from sklearn.ensemble import BaseEnsemble
from imblearn.over_sampling import SMOTE
from Bagging import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone 
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

"""Oversampled Bagging Classifier"""

class OB(BaggingClassifier):

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=5, random_state=None, hard_voting=True):
        BaggingClassifier.__init__(self, base_estimator=base_estimator, n_estimators=n_estimators, hard_voting=hard_voting, random_state=random_state)
        self.base_estimator = base_estimator
        self.random_state = random_state
        np.random.seed(self.random_state)

    def __oversample(self, X, y):
        """Oversampling"""
        preproc = SMOTE(random_state=1410)
        X_new, y_new = preproc.fit_resample(X,y)
        return X_new, y_new
    
    def fit(self, X, y):
        """Fitting"""
        X, y = self.__oversample(X,y)
        X, y = check_X_y(X,y)
        super(OB, self).fit(X,y)
    
    def predict(self, X):
        """Predict"""
        super(OB, self).predict(X)
    
    def ensemble_support_matrix(self, X):
        """Support matrix"""
        super(OB, self).ensemble_support_matrix(X)
