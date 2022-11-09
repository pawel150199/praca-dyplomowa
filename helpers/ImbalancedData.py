import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X,y = make_classification(
    n_samples=50,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    flip_y=0.05,
    weights=[0.9, 0.1],
    scale=1.7
)

plt.scatter(X[:,0],X[:,1], marker="o", c=y, cmap="bwr")
plt.title("Dane niezbalansowane")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.tight_layout()
plt.savefig("../images/ExampleOfImbalancedData.png")
