import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import DecisionBoundaryDisplay


"""Show decision boundary of choosen classifiers"""


class DecisionBoundary():
    
    def __init__(self, clfs, name):
        self.clfs = clfs
        self.name = name

    def process(self, X , y):
        """Process"""
        for _, clf_name in enumerate(self.clfs):
            clf = clone(self.clfs[clf_name])
            clf.fit(X, y)
            
            # Decision boundary
            _, axarr = plt.subplots(1,1, sharex="col", sharey="row", figsize=(15,7))
            DecisionBoundaryDisplay.from_estimator(clf, X, alpha=0.4, ax=axarr, response_method="predict")
            axarr.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors="k")
            axarr.set_xlabel("Feature 0")
            axarr.set_ylabel("Feature 1")
            axarr.set_title(f"{self.name} - {clf_name} Boundary Decision")
            plt.tight_layout()
            os.chdir('../images')
            plt.savefig(f"{self.name}-BoundaryDecision-{clf_name}.png")            
