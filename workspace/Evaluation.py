import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import os

clfs = {
    'SVC' : SVC(),
    'Linear SVC' : LinearSVC(),
    'kNN' : KNeighborsClassifier()
}

datasets = ['appendicitis', 'balance', 'banana', 'bupa', 'glass']


n_repeat = 5
n_split = 2
scores = np.zeros((len(datasets), n_split*n_repeat, len(clfs)))
rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=1234)

for data_id, dataset in enumerate(datasets):
    os.chdir('../datasets')
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1]

    for fold_id, (train, test) in enumerate(rskf.split(X,y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clfs[clf_name]
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[data_id, fold_id, clf_id] = accuracy_score(y[test], y_pred)
os.chdir('../results')
np.save('preexperiment', scores)
mean = np.mean(scores, axis=2)
std = np.std(scores, axis=2)
print(mean, "\n")
print(std, '\n')
for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))

