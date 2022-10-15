import os, sys
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

sys.path.append('../ensemble')
from Bagging import BaggingClassifier
from RandomSubspaceEnsemble import RandomSubspaceEnsemble
from RandomSamplePartition import RandomSamplePartition

"""Pokaz granic decyzyjnych poszczególnych zespołów klasyfikatorów"""
#Dane
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_repeated=0,
    n_informative=2,
    flip_y=0.01,
    random_state=1410,
    n_classes=2,
    weights=[0.2, 0.8]
)

#Zespoły klasyfikatorów
clfs = {
    #'Linear SVC' : LinearSVC(random_state=1410)
    #'Bagging LSVC x5' : BaggingClassifier(base_estimator=LinearSVC(random_state=1410), n_estimators=5, random_state=1410),
    #'Bagging LSVC x10' : BaggingClassifier(base_estimator=LinearSVC(random_state=1410), n_estimators=10, random_state=1410),
    #'Bagging LSVC x15' : BaggingClassifier(base_estimator=LinearSVC(random_state=1410), n_estimators=15, random_state=1410),
}

#Definicja granic
min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

#Definicja  skali x i y
x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)

#Definicja meshgrid
xx, yy = np.meshgrid(x1grid, x2grid)

#Spłaszczanie gridów do wektora
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

#Zmiana z vektora do kolumny
grid = np.hstack((r1, r2))

#Model
for clf_id, clf_name in enumerate(clfs):
    clf = clone(clfs[clf_name])
    clf.fit(X, y)
    yhat = clf.predict(grid)
    zz = yhat.reshape(xx.shape)

    #Wykres
    plt.contour(xx, yy, zz, cmap='Paired')
    for class_value in range(2):
        row_ix = np.where(y == class_value)
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
        os.chdir('../images')
        plt.savefig(f"boundaryDecision{clf_name}.png")
