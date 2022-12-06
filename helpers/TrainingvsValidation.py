import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    flip_y=.05,
    random_state=1410,
    weights=[0.2, 0.8]

)

#dataset = np.genfromtxt("../datasets/glass4.csv", delimiter=',')
#X = dataset[:, :-1]
#y = dataset[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
)

n_neighbors = np.linspace(1,15,15).astype(int)
train_errors = []
test_errors = []
clf = KNeighborsClassifier()

for n in n_neighbors:
    print(n)
    clf.set_params(n_neighbors=n)
    clf.fit(X_train, y_train)
    train_errors.append(1 - clf.score(X_train, y_train))
    test_errors.append(1 - clf.score(X_test, y_test))

dim = np.arange(1,15,1)
plt.subplot(1,1,1)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13) 
plt.plot(n_neighbors, train_errors, label="Zbiór treningowy")
plt.plot(n_neighbors, test_errors, label="Zbiór testowy")
plt.legend(loc="upper right", frameon=False)
plt.ylim([0, 1])
plt.xlim([1,15])
plt.xticks(dim)
plt.xlabel("Liczba sąsiadów")
plt.ylabel("Błąd")
plt.grid()
plt.tight_layout()
plt.savefig("../images/TrainingValidationErrors.png", dpi=400)
