import numpy as np
from sklearn.ensemble import BaseEnsemble
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone 
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode

"""Oversampled Bagging Classifier"""

class OB(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=5, random_state=None, hard_voting=True):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.hard_voting = hard_voting
        self.random_state = random_state
        np.random.seed(self.random_state)
    
    def __oversample(self, X, y):
        """Oversampling"""
        preproc = SMOTE(random_state=1410)
        X_new, y_new = preproc.fit_resample(X,y)
        return X_new, y_new
    
    def fit(self, X, y):
        """Training"""
        X, y = self.__oversample(X,y)
        X, y = check_X_y(X,y)
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]
        
        # Matrix for classifiers
        self.ensemble_ = []

        # Bagging
        for i in range(self.n_estimators):
            self.bootstrap = np.random.choice(len(X),size=len(X), replace=True)
            self.ensemble_.append(clone(self.base_estimator).fit(X[self.bootstrap], y[self.bootstrap]))
        return self
    
    def predict(self, X):
        """Prediction"""
        # Check if models are fitted
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("Number of features does not match")


        if self.hard_voting:
            # Hard voting
            pred_ = []
            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X))
            pred_ = np.array(pred_)
            prediction = mode(pred_, axis=0)[0].flatten()
            return self.classes_[prediction]

        else:
            # Soft voting
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
            return self.classes_[prediction]
                  
    def ensemble_support_matrix(self, X):
        """Support matrix"""
        probas_ = []
        for _, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X))
        return np.array(probas_)
