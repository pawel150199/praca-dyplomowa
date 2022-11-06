from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt 

# Load dataset
dataset = load_iris()
X, y = dataset.data, dataset.target
class_names = dataset.target_names
feature_names = dataset.feature_names

# Classifier
clf = tree.DecisionTreeClassifier(random_state=1410)
clf.fit(X,y)

# Plot tree
tree.plot_tree(clf, filled=True, rounded=True, class_names=class_names, feature_names=feature_names)
plt.savefig("../images/DecisionTree.png")




