from cmath import sqrt
import numpy as np
import sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
sys.path.append("../ensembles")
from Bagging import BaggingClassifier
from RSPmod import RandomSamplePartition as RSP
from RandomSubspaceMethod import RandomSubspaceMethod
from RandomSamplePartition import RandomSamplePartition
from OB import OB
from ORSM import ORSM
from ORSP import ORSP
from UB import UB
from URSM import URSM
from URSP import URSP

"""
This code is used to display comparision between implemented ensemble methods
"""

warnings.filterwarnings('ignore')

names = [
    #"Bagging",
    "UB",
    "OB",
    "RSM",
    "ORSM",
    "URSM",
    "RSP",
    "ORSP",
    "URSP",
]

base_estimator = GaussianNB()
n_estimators = 5
classifiers = [
    #BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators),
    RandomSubspaceMethod(base_estimator=base_estimator, n_estimators=n_estimators, n_subspace_features=2),
    RandomSamplePartition(base_estimator=base_estimator, n_estimators=n_estimators, n_subspace_features=2),
    OB(base_estimator=base_estimator, n_estimators=n_estimators),
    ORSM(base_estimator=base_estimator, n_estimators=n_estimators, n_subspace_features=2),
    ORSP(base_estimator=base_estimator, n_estimators=n_estimators, n_subspace_features=2),
    UB(base_estimator=base_estimator, n_estimators=n_estimators),
    URSM(base_estimator=base_estimator, n_estimators=n_estimators, n_subspace_features=2),
    URSP(base_estimator=base_estimator, n_estimators=n_estimators, n_subspace_features=2)
]

X, y = make_classification(
    n_features=10, n_redundant=0, n_informative=10, random_state=1, n_clusters_per_class=1, n_classes=2
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]

figure = plt.figure(figsize=(27, 9))
i = 1
# Iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # Preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # Iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.savefig("../images/BoundaryDecisionComparision.png")
