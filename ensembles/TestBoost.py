from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from AdaBoost import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Dane
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    flip_y=0.08,
    weights=[0.9, 0.1],
    random_state=1410
)

# Podział na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
# Klasyfikator
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=10)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc = balanced_accuracy_score(y_test, y_pred)
print("Accuracy score: %.3f" % (acc))
