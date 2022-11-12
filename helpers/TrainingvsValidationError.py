import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

X,y = make_classification(
    n_samples=400,
    n_redundant=0,
    n_repeated=0,
    n_informative=2,
    n_classes=2,
    flip_y=0.05,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
)

n_neighbors = np.linspace(1,10,10)
clf = KNeighborsClassifier()
train_errors = []
test_errors = []

for n in n_neighbors:
    clf.set_params(n_neighbors=int(n))
    clf.fit(X_train, y_train)
    train_errors.append(clf.score(X_train, y_train))
    test_errors.append(clf.score(X_test, y_test))

i_alpha_optim = np.argmax(test_errors)
alpha_optim = n_neighbors[i_alpha_optim]

plt.subplots(1,1,figsize=(10,5))

plt.plot(n_neighbors, train_errors, label="Zbiór treningowy")
plt.plot(n_neighbors, test_errors, label="Zbiór testowy")

plt.legend(loc="lower left")
plt.ylim([0, 1.2])
plt.xlabel("Liczba sąsiadów")
plt.ylabel("Błąd")
plt.tight_layout()
plt.savefig("../images/training_validation_errors.png")
