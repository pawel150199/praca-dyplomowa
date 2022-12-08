import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.svm import LinearSVC
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode

"""Random Sample Partition"""

class RandomSamplePartition(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=LinearSVC(), n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_subspace_features = n_subspace_features
        self.hard_voting = hard_voting
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        """Fitting"""
        X, y = check_X_y(X,y)
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]

        if self.n_subspace_features > self.n_features:
            raise ValueError("Number of features in subspace higher than number of features.")

        self.subspaces =[]
        self.subspaces = np.random.randint(0, self.n_features, (self.n_estimators, self.n_subspace_features))
        
        # Fit new models and save it in ensemble matrix
        # This part work only on 
        #self.ensemble_ = []
        #for i in range(self.n_estimators):
        #   self.ensemble_.append(clone(self.base_estimator).fit(X[:,self.subspaces[i]], y))

        # Fit new models and save it in ensemble matrix
        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.bootstrap = np.random.choice(len(X),size=len(X), replace=True)
            X_par=X[self.bootstrap, :]
            X_par=X_par[:, self.subspaces[i]]
            self.ensemble_.append(clone(self.base_estimator).fit(X_par, y[self.bootstrap]))
        return self


    def predict(self, X):
        """Prediction"""
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("Number of features does not match")


        if self.hard_voting:
            # Hard voting
            pred_ = []
            for i in range(self.n_estimators):
                pred_.append(self.ensemble_[i].predict(X[:,self.subspaces[i]]))
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
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X[:, self.subspaces[i]]))
        return np.array(probas_)
