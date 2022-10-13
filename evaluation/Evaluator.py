import numpy as np
import os, sys
import time
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier as OBgg
from strlearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier

sys.path.append('../algorithms')
from Bagging import BaggingClassifier


"""
Klasa słuzy do przeprowadzania ewaluacji eksperymentu
"""
class Evaluator():
    def __init__(self, datasets, storage_dir=None, n_splits=2, n_repeats=5, random_state=None, metrics=accuracy_score):
        self.datasets = datasets
        self.storage_dir = storage_dir
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.metrics = metrics

        #Sprawdzenie czy istnieje storage
        if self.storage_dir is not None:
            return
        else:
            raise ValueError('Directory cannot be None')

    def process(self, clfs, result_name):
        """
        Funkcja do przeprowadzania ewaluacji eksperymentu
        """
        self.clfs = clfs
        rskf = rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state = self.random_state)
        scores = np.zeros((len(datasets), self.n_splits*self.n_repeats, len(self.clfs), len(self.metrics)))

        for data_id, data_name in enumerate(self.datasets):
            os.chdir('../datasets')
            dataset = np.genfromtxt("%s.csv" % (data_name) , delimiter=',')
            X = dataset[:, :-1]
            y = dataset[:, -1].astype(int)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                clf = clfs[clf_name]
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                for metric_id, metric_name in enumerate(self.metrics):
                    #DATA X FOLD X CLASSIFIER X METRIC 
                    scores[data_id, fold_id, clf_id, metric_id] = self.metrics[metric_name](y[test],y_pred)
    
        #Wyświetlenie wyników
        mean = np.mean(scores, axis=1)
        std = np.std(scores, axis=1)
        for clf_id, clf_name in enumerate(clfs):
            print("%s: %.3f (%.3f)" % (clf_name, mean[0,clf_id,0], std[0,clf_id,0]))

        #Zapisanie  wyników
        try:
            os.chdir('../%s' % (self.storage_dir))
            np.save(result_name, scores)
        except ValueError:
            print("Incorrect directory")

        

if __name__ == '__main__':

    #Klasyfikatory
    clfs = {
        'Bagging LSVC5' : BaggingClassifier(MLPClassifier(hidden_layer_sizes=10), n_estimators=1),
        'Bagging LSVC10': BaggingClassifier(MLPClassifier(hidden_layer_sizes=10), n_estimators=9),
        'Bagging LSVC15': BaggingClassifier(MLPClassifier(hidden_layer_sizes=10), n_estimators=12)
    }

    #Zbiór danych
    datasets = ['glass']

    metrics = {
        'BAC' : balanced_accuracy_score
    }

    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=1410, metrics=metrics)
    ev.process(clfs, result_name='resultsxdd')