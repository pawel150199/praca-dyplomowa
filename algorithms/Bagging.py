import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone
from scipy.stats import mode 
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

class BaggingClassifier(BaseEnsemble, ClassifierMixin):
    """Bagging Classifier"""
    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=5, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        np.random.seed(self.random_state)
    
    def fit(self, X, y):
        """Trening"""
        X, y = check_X_y(X,y)
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]
        self.ensemble_ = []
        #Bagging
        for i in range(self.n_estimators):
            bootstrap = np.random.choice(X.shape[0],size=X.shape[0], replace=True)
            self.ensemble_.append(clone(self.base_estimator).fit(X[bootstrap], y[bootstrap]))
        return self
    
    def predict(self, X):
        """Predykcja"""
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("Number of features does not match")

        pred_ = []
        for i, member_clf in enumerate(self.ensemble_):
            pred_.append(member_clf.predict(X))
        pred_ = np.array(pred_)
        prediction = mode(pred_, axis=0)[0].flatten()
        return self.classes_[prediction]