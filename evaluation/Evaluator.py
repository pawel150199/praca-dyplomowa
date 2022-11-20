import numpy as np
import os
import warnings
from sklearn.base import clone
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold


"""
Class is used for evaluate experiment
"""


class Evaluator():
    def __init__(self, datasets, storage_dir=None, n_splits=5, n_repeats=2, random_state=None, metrics=accuracy_score):
        self.datasets = datasets
        self.storage_dir = storage_dir
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.metrics = metrics

        # Check does exist storage
        if self.storage_dir is not None:
            return
        else:
            raise ValueError('Directory cannot be None!')

    def process(self, clfs, result_name):
        """
        Evaluation function
        """
        # Ignore warnings
        warnings.filterwarnings("ignore")

        self.clfs = clfs
        rskf = rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state = self.random_state)
        self.scores = np.zeros((len(self.datasets), self.n_splits*self.n_repeats, len(self.clfs), len(self.metrics)))

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
                    # DATA X FOLD X CLASSIFIER X METRIC 
                        self.scores[data_id, fold_id, clf_id, metric_id] = self.metrics[metric_name](y[test],y_pred)
    
        # Show outputs
        self.mean = np.mean(self.scores, axis=1)
        self.std = np.std(self.scores, axis=1)
        #for clf_id, clf_name in enumerate(clfs):
        #    print("%s: %.3f (%.3f)" % (clf_name, self.mean[0,clf_id,0], self.std[0,clf_id,0]))

        if result_name != None:
            # Save outputs
            try:
                os.chdir('../%s' % (self.storage_dir))
                np.save(result_name, self.scores)
            except ValueError:
                print("Incorrect value!")

    def process_ranks(self):
        """Calculate global ranks"""

        self.mean_ranks = []
        self.ranks = []

        for m, _ in enumerate(self.metrics):
            scores_ = self.mean[:, :, m]

            # Ranks
            ranks = []
            for row in scores_:
                ranks.append(rankdata(row).tolist())
                
            ranks = np.array(ranks)
            self.ranks.append(ranks)
            mean_ranks = np.mean(ranks, axis=0)
            self.mean_ranks.append(mean_ranks)

        self.mean_ranks = np.array(self.mean_ranks)
        self.ranks = np.array(self.ranks)
