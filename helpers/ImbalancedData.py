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

dim = np.arange(-4,5,1)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16) 
plt.scatter(X[:,0],X[:,1], marker="o", c=y, cmap="bwr")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xticks(dim)
plt.yticks(dim)
plt.tight_layout()
plt.grid(True)
plt.savefig("../images/ExampleOfImbalancedData.png", dpi=400)
