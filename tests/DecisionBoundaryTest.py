import os, sys
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

sys.path.append('../ensemble')
from Bagging import BaggingClassifier
from RandomSubspaceEnsemble import RandomSubspaceEnsemble
from RandomSamplePartition import RandomSamplePartition

sys.path.append('../analysis')
from DecisionBoundary2 import DecisionBoundary


#Dane
#X, y = make_classification(
#    n_samples=200,
#    n_features=2,
#    n_redundant=0,
#    n_repeated=0,
#    n_informative=2,
#    flip_y=0.01,
#    random_state=1410,
#    n_classes=2,
#    weights=[0.2, 0.8]
#)
iris = load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

#Zespoły klasyfikatorów
clfs = {
    #'kNN' : KNeighborsClassifier(n_neighbors=5),
    'RSE kNN x5' : RandomSubspaceEnsemble(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=5, random_state=1410, n_subspace_features=2),
    'RSE kNN x10' : RandomSubspaceEnsemble(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=10, random_state=1410, n_subspace_features=2),
    'RSE kNN x15' : RandomSubspaceEnsemble(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=15, random_state=1410, n_subspace_features=2),
}

db = DecisionBoundary(clfs, 'kNN')
db.process(X, y)