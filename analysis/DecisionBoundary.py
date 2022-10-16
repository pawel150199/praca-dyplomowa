import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone

sys.path.append('../ensemble')
from Bagging import BaggingClassifier
from RandomSubspaceEnsemble import RandomSubspaceEnsemble
from RandomSamplePartition import RandomSamplePartition

"""
Moja własna implementacja 
Pokaz granic decyzyjnych poszczególnych zespołów klasyfikatorów
"""

class DecisionBoundary():

    def __init__(self, clfs, name):
        self.clfs = clfs
        self.name = name

    def process(self, X , y):
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
        for clf_id, clf_name in enumerate(self.clfs):
            fig, ax = 0, 0
            clf = clone(self.clfs[clf_name])
            clf.fit(X, y)
            yhat = clf.predict(grid)
            zz = yhat.reshape(xx.shape)
            
            plt.contour(xx, yy, zz, cmap='Paired')
            for class_value in range(2):
                row_ix = np.where(y == class_value)
                plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
                plt.ylim(-4, 4)
                plt.xlim(-4, 4)
                plt.xlabel("Feature 0")
                plt.ylabel("Feature 1")
                plt.title(f"{clf_name} Decision Boundary")
                os.chdir('../images')
                plt.savefig(f"boundaryDecision-{self.name}-{clf_name}.png")
                
            
