import sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
sys.path.append("../preprocessing")
from ModifiedClusterCentroids import ModifiedClusterCentroids


X,y = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_informative=2,
    weights=[0.3,0.7],
    random_state=1234
)

fig, ax = plt.subplots(211, figsize=(10,5))
res = ModifiedClusterCentroids()
X_new, y_new = res.fit_resample(X,y)
