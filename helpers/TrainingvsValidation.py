import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dataset = np.genfromtxt("../datasets/ecoli2.csv", delimiter=',')
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, shuffle=False
)

n_neighbors = np.linspace(1,15,15).astype(int)
train_errors = []
test_errors = []
clf = KNeighborsClassifier()

for n in n_neighbors:
    print(n)
    clf.set_params(n_neighbors=n)
    clf.fit(X_train, y_train)
    train_errors.append(clf.score(X_train, y_train))
    test_errors.append(clf.score(X_test, y_test))

plt.subplot(111)
plt.plot(n_neighbors, train_errors, label="Zbiór treningowy")
plt.plot(n_neighbors, test_errors, label="Zbiór testowy")
plt.legend(loc="upper right")
plt.ylim([0, 1])
plt.xlim([1,15])
plt.xlabel("Liczba sąsiadów")
plt.ylabel("Błąd")
plt.grid()
plt.tight_layout()
plt.savefig("../images/TrainingValidationErrors.png")
