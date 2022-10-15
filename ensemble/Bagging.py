import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone 
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode 
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold

"""Bagging Classifier"""

class BaggingClassifier(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=5, random_state=None, hard_voting=True):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.hard_voting = hard_voting
        self.random_state = random_state
        np.random.seed(self.random_state)
    
    def fit(self, X, y):
        """Trening"""
        X, y = check_X_y(X,y)
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]
        
        #Macierz na wyniki
        self.ensemble_ = []

        #Bagging
        for i in range(self.n_estimators):
            self.bootstrap = np.random.choice(len(X),size=len(X), replace=True)
            self.ensemble_.append(clone(self.base_estimator).fit(X[self.bootstrap], y[self.bootstrap]))
        return self
    
    def predict(self, X):
        """Predykcja"""
        #Sprawdzenie czy modele są wyuczone
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.n_features:
            raise ValueError("Number of features does not match")


        if self.hard_voting:
            #Głosowanie większościowe
            pred_ = []
            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X))
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
        """Macierz wsparć"""
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X))
        return np.array(probas_)


if __name__ == '__main__':
    datasets = ['bupa']

    clfs = {
        'Bagging NHV 10': BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=False, n_estimators=10, random_state=1234),
        'Bagging NHV 30': BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=False, n_estimators=30, random_state=1234),
        'Bagging HV 10': BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=True, n_estimators=10, random_state=1234),
        'Bagging HV 30': BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=True, n_estimators=30, random_state=1234)
    }   

    n_repeat = 5
    n_split = 2
    scores = np.zeros((len(clfs), 1, n_split*n_repeat))
    rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=1410)

    #for data_id, dataset in enumerate(datasets):
        #dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
        #X = dataset[:, :-1]
        #y = dataset[:, -1].astype(int)
    X, y = make_classification(
            n_samples=100, n_classes=4, n_informative=4, random_state=100)
    for fold_id, (train, test) in enumerate(rskf.split(X,y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, 0, fold_id] = accuracy_score(y[test], y_pred)

    mean = np.mean(scores, axis=2)
    std = np.std(scores, axis=2)
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))