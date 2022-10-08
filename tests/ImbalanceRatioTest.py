import os
os.chdir('../algorithms')
import unittest
import numpy as np
from sklearn.datasets import make_classification
from ImbalanceRatioTest import IR

X,y = make_classification(
            n_samples=1000,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=1234,
            weights=[0.2, 0.8]
        )
