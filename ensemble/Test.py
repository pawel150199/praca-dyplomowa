import numpy as np
import os
import time
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from Bagging import BaggingClassifier
from RandomSamplePartition import RandomSamplePartition
from RandomSubspaceEnsemble import RandomSubspaceEnsemble
from sklearn.ensemble import BaggingClassifier as OBgg
from strlearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier

#Klasyfikatory
clfs = {
    'Bagging LSVC5' : BaggingClassifier(MLPClassifier(hidden_layer_sizes=10), n_estimators=1),
    'Bagging LSVC10': BaggingClassifier(MLPClassifier(hidden_layer_sizes=10), n_estimators=9),
    'Bagging LSVC15': BaggingClassifier(MLPClassifier(hidden_layer_sizes=10), n_estimators=12)
}

#Zbiór danych
datasets = ['glass']

if __name__ =='__main__':
    #Stratyfikowana, wielokrotna, walidacja krzyzowa
    n_splits = 2
    n_repeats = 5
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 1234)

    #Tablice z wynikami
    scores = np.zeros((n_splits*n_repeats, len(clfs)))
    duration = [0 for x in range(len(clfs))]
    #Eksperyment
    for data_id, data_name in enumerate(datasets):
        os.chdir('../datasets')
        dataset = np.genfromtxt("%s.csv" % (data_name) , delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                start = time.time()
                clf = clfs[clf_name]
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                scores[fold_id, clf_id] = balanced_accuracy_score(y[test],y_pred)
                dur = time.time()-start
                duration[clf_id] += dur
    
    print(duration)
    #Wyświetlenie wyników
    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))
    #Zapisanie  wyników
    os.chdir('../results')
    np.save('results', scores)
