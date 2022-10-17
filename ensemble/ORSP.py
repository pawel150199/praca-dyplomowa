import numpy as np
from scipy.stats import mode
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaseEnsemble
from imblearn.over_sampling import SMOTE
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

"""Oversampled Random Sample Partition - na cechach"""

class ORSP(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=LinearSVC(), n_estimators=10, n_subspace_choose=0.4, n_subspace_features=2, hard_voting=True, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_subspace_features = n_subspace_features
        self.hard_voting = hard_voting
        self.n_subspace_choose = n_subspace_choose
        self.random_state = random_state
        np.random.seed(self.random_state)

    def oversample(self, X, y):
        """Oversampling"""
        preproc = SMOTE(random_state=1410)
        X_new, y_new = preproc.fit_resample(X,y)
        return X_new, y_new

    def fit(self, X, y):
        """Uczenie"""
        self.n_subspace_choose=1
        X, y = self.oversample(X,y)
        X, y = check_X_y(X,y)
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]

        if self.n_subspace_features > self.n_features:
            raise ValueError("Number of features in subspace higher than number of features.")

        n_subspace = int(X.shape[1]/self.n_subspace_features) 
        self.n_subspace_choose = int(self.n_subspace_choose * n_subspace)
        self.subspaces =[]
        self.subspaces = np.random.choice(X.shape[1], size=(n_subspace, self.n_subspace_features), replace=False) 
        
        x = np.random.choice(n_subspace, size=(self.n_subspace_choose),replace=False)
        self.subspaces = self.subspaces[x,:]

        #Zmiana ilości estymatorów w przypadku przekroczenia dostępnych podprzestrzeni
        if self.n_estimators > self.n_subspace_choose:
            self.n_estimators = self.n_subspace_choose

        #Wyuczenie nowych modeli i stworzenie zespołu
        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.ensemble_.append(clone(self.base_estimator).fit(X[:,self.subspaces[i]], y))
        return self


    def predict(self, X):
        """Predykcja"""
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")


        if self.hard_voting:
            #Głosowanie większościowe
            pred_ = []
            for i in range(self.n_estimators):
                pred_.append(self.ensemble_[i].predict(X[:, self.subspaces[i]]))
            pred_ = np.array(pred_)
            prediction = mode(pred_, axis=0)[0].flatten()
            return self.classes_[prediction]
        else:
            #Głosowanie na podstawie wektorów wsparc
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
            return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        """Wyliczenie macierzy wsparcia"""
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X[:, self.subspaces[i]]))
        return np.array(probas_)
