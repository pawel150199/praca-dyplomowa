import numpy as np
import sys

sys.path.append('../algorithms')
from Bagging import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from RandomSamplePartition import RSP
from RandomSubspaceEnsemble import RandomSubspaceEnsemble
from strlearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Base estimators
base_estimators = {
    'GNB': GaussianNB(),
    'SVC': SVC(),
    'kNN': KNeighborsClassifier(),
    'Linear SVC': LinearSVC()
}

# Ensemble
clfs = {
    'Bagging' : BaggingClassifier(base_estimator=SVC(), n_estimators=10),
    'RSP': RSP(base_estimator=SVC(), n_estimators=10),
    'RSE': RandomSubspaceEnsemble(base_estimator=SVC(), n_estimators=10)
}

datasets = ['appendicitis', 'balance', 'banana', 'bupa', 'glass']

if __name__ == '__main__':

    n_repeat = 5
    n_split = 2
    scores = np.zeros((len(clfs), len(datasets), n_split*n_repeat))
    rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=1410)

    for data_id, dataset in enumerate(datasets):
        sys.path.append('../datasets')
        dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X,y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clfs[clf_name]
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, 0, fold_id] = balanced_accuracy_score(y[test], y_pred)

    mean = np.mean(scores, axis=2)
    std = np.std(scores, axis=2)
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))