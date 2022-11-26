import sys
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
sys.path.append("../ensembles")
from Bagging import BaggingClassifier
from RandomSamplePartition import RandomSamplePartition as RSP
from RandomSubspaceMethod import RandomSubspaceMethod as RSM
from ORSP import ORSP
from ORSM import ORSM
from OB import OB
from URSP import URSP
from URSM import URSM
from UB import UB

warnings.filterwarnings('ignore')

dataset = np.genfromtxt("../datasets/ecoli2.csv", delimiter=',')
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
)

n_estimators = np.linspace(1,50,50)
print(n_estimators)
train_errors = []
test_errors = []

# Bagging
for n in n_estimators:
    print(n)
    clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=int(n))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    train_errors.append(1 - acc_train)

plt.subplots(1,1,figsize=(10,5))
plt.plot(n_estimators, train_errors, label="Bagging")

train_errors = []
test_errors = []

# RSM
for n in n_estimators:
    clf = RSM(base_estimator=KNeighborsClassifier(), n_estimators=int(n))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    train_errors.append(1 - acc_train)

plt.plot(n_estimators, train_errors, label="RSM")

train_errors = []
test_errors = []

# RSP
for n in n_estimators:
    clf = RSP(base_estimator=KNeighborsClassifier(), n_estimators=int(n))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    train_errors.append(1 - acc_train)

plt.plot(n_estimators, train_errors, label="RSP")
plt.legend(loc="upper right")
plt.ylim(0, 0.25)
plt.xlim(1,50)
plt.xlabel("Liczba klasyfikatorów w zespole")
plt.ylabel("Błąd")
plt.title("Wykres zależności błędu klasyfikacji.")
plt.tight_layout()
plt.savefig("../images/EnsembleGeneralization.png")

plt.subplots(1,1,figsize=(10,5))
train_errors = []
test_errors = []

# OB
for n in n_estimators:
    print(n)
    clf = OB(base_estimator=KNeighborsClassifier(), n_estimators=int(n))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    train_errors.append(1 - acc_train)

plt.plot(n_estimators, train_errors, label="OB")

train_errors = []
test_errors = []

# ORSM
for n in n_estimators:
    clf = ORSM(base_estimator=KNeighborsClassifier(), n_estimators=int(n))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    train_errors.append(1 - acc_train)

plt.plot(n_estimators, train_errors, label="ORSM")

train_errors = []
test_errors = []

# ORSP
for n in n_estimators:
    clf = ORSP(base_estimator=KNeighborsClassifier(), n_estimators=int(n))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    train_errors.append(1 - acc_train)

plt.plot(n_estimators, train_errors, label="ORSP")
plt.legend(loc="upper right")
plt.ylim(0, 0.25)
plt.xlim(1,50)
plt.xlabel("Liczba klasyfikatorów w zespole")
plt.ylabel("Błąd")
plt.title("Wykres zależności błędu klasyfikacji.")
plt.tight_layout()
plt.savefig("../images/OversampledEnsembleGeneralization.png")

plt.subplots(1,1,figsize=(10,5))
train_errors = []
test_errors = []

# UB
for n in n_estimators:
    print(n)
    clf = UB(base_estimator=KNeighborsClassifier(), n_estimators=int(n))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    train_errors.append(1 - acc_train)

plt.plot(n_estimators, train_errors, label="UB")

train_errors = []
test_errors = []

# URSM
for n in n_estimators:
    clf = URSM(base_estimator=KNeighborsClassifier(), n_estimators=int(n))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    train_errors.append(1 - acc_train)

plt.plot(n_estimators, train_errors, label="URSM")

train_errors = []
test_errors = []

# RSP
for n in n_estimators:
    clf = URSP(base_estimator=KNeighborsClassifier(), n_estimators=int(n))
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    train_errors.append(1 - acc_train)

plt.plot(n_estimators, train_errors, label="URSP")




plt.legend(loc="upper right")
plt.ylim(0, 0.25)
plt.xlim(1,50)
plt.xlabel("Liczba klasyfikatorów w zespole")
plt.ylabel("Błąd")
plt.title("Wykres zależności błędu klasyfikacji.")
plt.tight_layout()
plt.savefig("../images/UndersampledEnsembleGeneralization.png")
