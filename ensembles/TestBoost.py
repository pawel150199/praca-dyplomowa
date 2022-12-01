from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from RSPmod import RandomSamplePartition

# Data
X, y = make_classification(
    n_samples=10000,
    n_classes=2,
    n_features=100,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    flip_y=0.08,
    weights=[0.8, 0.2],
    random_state=1410
)

# Divide to training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
# Classifier
clf = RandomSamplePartition(base_estimator=GaussianNB(), n_estimators=10, n_subspace_choose=0.7, n_subspace_features=3)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(y_pred)
acc = balanced_accuracy_score(y_test, y_pred)
print("Accuracy score: %.3f" % (acc))
