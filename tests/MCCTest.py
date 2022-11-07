import unittest
import sys
import warnings
from sklearn.datasets import make_classification
sys.path.append("../preprocessing")
from ModifiedClusterCentroids import ModifiedClusterCentroids

"""Test ModifiedClusterCentroids class"""
# Ignore warnings
X,y = make_classification(
            n_samples=1000,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=1234,
            weights=[0.2, 0.8]
        )

class Test(unittest.TestCase):
    def test_DBSCAN_const(self):
        preprocs = ModifiedClusterCentroids(cluster_algorithm='DBSCAN', CC_strategy='const')
        X_new, y_new = preprocs.fit_resample(X,y)
        self.assertNotEqual(X.shape[0], X_new.shape[0])
        self.assertEqual(X.shape[1], X_new.shape[1])
        self.assertNotEqual(X.shape[0], 0)
        self.assertNotEqual(X.shape[1], 0)
    
    def test_DBSCAN_auto(self):
        preprocs = ModifiedClusterCentroids(cluster_algorithm='DBSCAN', CC_strategy='const')
        X_new, y_new = preprocs.fit_resample(X,y)
        self.assertNotEqual(X.shape[0], X_new.shape[0])
        self.assertEqual(X.shape[1], X_new.shape[1])
        self.assertNotEqual(X.shape[0], 0)
        self.assertNotEqual(X.shape[1], 0)

    def test_OPTICS_const(self):
        preprocs = ModifiedClusterCentroids(cluster_algorithm='OPTICS', CC_strategy='const')
        X_new, y_new = preprocs.fit_resample(X,y)
        self.assertNotEqual(X.shape[0], X_new.shape[0])
        self.assertEqual(X.shape[1], X_new.shape[1])
        self.assertNotEqual(X.shape[0], 0)
        self.assertNotEqual(X.shape[1], 0)
    
    def test_OPTICS_auto(self):
        preprocs = ModifiedClusterCentroids(cluster_algorithm='OPTICS', CC_strategy='auto')
        X_new, y_new = preprocs.fit_resample(X,y)
        self.assertNotEqual(X.shape[0], X_new.shape[0])
        self.assertEqual(X.shape[1], X_new.shape[1])
        self.assertNotEqual(X.shape[0], 0)
        self.assertNotEqual(X.shape[1], 0)
    
    def test_ERROR(self):
        with self.assertRaises(ValueError):
            preprocs = ModifiedClusterCentroids(cluster_algorithm='OPTIC', CC_strategy='auto')
            X_new, y_new = preprocs.fit_resample(X,y)
            
        with self.assertRaises(ValueError):
            preprocs = ModifiedClusterCentroids(cluster_algorithm='OPTICS', CC_strategy='au')
            X_new, y_new = preprocs.fit_resample(X,y)

if __name__=="__main__":
    unittest.main()