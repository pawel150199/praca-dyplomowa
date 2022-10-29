import numpy as np
import os, sys
import time
from sklearn.naive_bayes import GaussianNB
from Boosting import Boosting
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Dane
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    flip_y=0.08,
    weights=[0.8, 0.2],
    random_state=1410
)

# Podział na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Klasyfikator
clf = Boosting(base_estimator=GaussianNB(), M=300)
