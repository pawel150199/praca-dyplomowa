import unittest
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

"""Test e2e Decision Boundary class"""

class Test(unittest.TestCase):
    def prepare(self):
        iris = load_iris()
        self.X = iris.data[:, [0, 2]]
        self.y = iris.target

        # Ensemble classifiers
        self.clfs = {
            #'kNN' : KNeighborsClassifier(n_neighbors=5),
            'RSE kNN x5' : RandomSubspaceEnsemble(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=5, random_state=1410, n_subspace_features=2),
            'RSE kNN x10' : RandomSubspaceEnsemble(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=10, random_state=1410, n_subspace_features=2),
            'RSE kNN x15' : RandomSubspaceEnsemble(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=15, random_state=1410, n_subspace_features=2),
        }
    
    def testDecisionBoundary(self):
        """Decision Boundary test"""
        self.prepare()
        db = DecisionBoundary(self.clfs, 'kNN')
        db.process(self.X, self.y)
        os.chdir('../images')
        for _, clf_name in enumerate(self.clfs):
            self.assertTrue(os.path.exists(f"kNN-BoundaryDecision-{clf_name}.png"))
    

if __name__=='__main__':
    unittest.main()