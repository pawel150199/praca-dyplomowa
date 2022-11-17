import sys
import unittest
import warnings
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
sys.path.append("../ensembles")
from RandomSubspaceMethod import RandomSubspaceMethod
sys.path.append("../analysis")
from DecisionBoundary2 import DecisionBoundary

# Ignore warnings
warnings.filterwarnings("ignore")

"""Test e2e Decision Boundary class"""

class Test(unittest.TestCase):
    def prepare(self):
        iris = load_iris()
        self.X = iris.data[:, [0, 2]]
        self.y = iris.target

        # Ensemble classifiers
        self.clfs = {
            'RSE kNN' : RandomSubspaceEnsemble(base_estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=5, random_state=1410, n_subspace_features=2),
        }
    
    def testDecisionBoundary(self):
        """Decision Boundary test"""
        self.prepare()
        db = DecisionBoundary(self.clfs, 'Test')
        db.process(self.X, self.y)
        os.chdir('../images')
        for _, clf_name in enumerate(self.clfs):
            self.assertTrue(os.path.exists(f"Test-BoundaryDecision-{clf_name}.png"))
    

if __name__=='__main__':
    unittest.main()