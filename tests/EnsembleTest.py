import numpy as np
import os, sys
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
sys.path.append('../ensemble')
from Bagging import BaggingClassifier
from RandomSamplePartition import RandomSamplePartition
from RandomSubspaceEnsemble import RandomSubspaceEnsemble
from UB import UB
from URSE import URSE
from URSP import URSP
from OB import OB
from ORSE import ORSE
from ORSP import ORSP
from sklearn.metrics import balanced_accuracy_score

"""Test e2e Ensemble methods"""

# Classifiers
clfs = {
    'Bagging' : BaggingClassifier(base_estimator=GaussianNB(), n_estimators=5),
    'RSP': RandomSamplePartition(base_estimator=GaussianNB(), n_estimators=5),
    'RSE': RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=5),
    'OB' : OB(base_estimator=GaussianNB(), n_estimators=5),
    'ORSP': ORSP(base_estimator=GaussianNB(), n_estimators=5),
    'ORSE': ORSE(base_estimator=GaussianNB(), n_estimators=5),
    'UB' : UB(base_estimator=GaussianNB(), n_estimators=5),
    'URSP': URSP(base_estimator=GaussianNB(), n_estimators=5),
    'URSE': URSE(base_estimator=GaussianNB(), n_estimators=5)
}

# Datasets
datasets = ['bupa']

def main():
    n_splits = 2
    n_repeats = 5
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1234)

    # Tables with outputs
    scores = np.zeros((n_splits*n_repeats, len(clfs)))

    # Experiment
    for data_id, data_name in enumerate(datasets):
        os.chdir('../datasets')
        dataset = np.genfromtxt("%s.csv" % (data_name) , delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                clf = clfs[clf_name]
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                scores[fold_id, clf_id] = balanced_accuracy_score(y[test],y_pred)
    
    # Show outputs
    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))

if __name__ =='__main__':
    main()