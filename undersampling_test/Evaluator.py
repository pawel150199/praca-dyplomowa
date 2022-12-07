import warnings
import sys
import numpy as np
from scipy.stats import rankdata
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, OneSidedSelection, CondensedNearestNeighbour
from sklearn.tree import DecisionTreeClassifier
from strlearn.metrics import balanced_accuracy_score, geometric_mean_score_1, f1_score, precision, recall, specificity
sys.path.append("../preprocessing")
from ModifiedClusterCentroids import ModifiedClusterCentroids


"""Evaluation of test many undersampling method and comparision with ModifiedClusterCentroids"""

# Ignore warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 1410
# Classifiers
clfs = {
    'GNB': GaussianNB()
}

# Undersamplings methods
preprocs = {
    'RUS' : RandomUnderSampler(random_state=RANDOM_STATE),
    'CC': ClusterCentroids(random_state=RANDOM_STATE),
    'NM': NearMiss(version=1),
    'OSS' : OneSidedSelection(rrandom_state=RANDOM_STATE),
    'CNN' : CondensedNearestNeighbour(random_state=RANDOM_STATE),
    'MCC': ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='DBSCAN')
}

metrics = {
    'BAC' : balanced_accuracy_score,
    'G-mean' : geometric_mean_score_1,
    'F1-score' : f1_score,
    'precision' : precision,
    'recall' : recall,
    'specificity' : specificity
}

# Datasets
datasets = [
    'abalone-21_vs_8',
    'abalone-3_vs_11',
    'abalone9-18',
    'ecoli-0-1-4-7_vs_5-6',
    'ecoli-0-1_vs_2-3-5',
    'ecoli-0-6-7_vs_3-5',
    'ecoli2',
    'ecoli4',
    'glass-0-1-5_vs_2',
    'glass-0-1-6_vs_2',
    'glass-0-1-6_vs_5-1',
    'glass-0-1-6_vs_5',
    'glass2',
    'glass4',
    'page-blocks-1-3_vs_4',
    'popfailures',
    'shuttle-6_vs_2-3',
    'vowel0',
    'yeast-0-2-5-7-9_vs_3-6-8',
    'yeast-0-3-5-9_vs_7-8',
    'yeast-2_vs_8',
    'yeast4',
    'yeast5',
    'yeast6'
]

if __name__ =='__main__':
    # Reapeated, Stratified, Cross validation
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    # Outputs with scores
    scores = np.zeros((len(datasets), len(preprocs), n_splits*n_repeats, len(metrics)))
    
    # Experiment
    for data_id, data_name in enumerate(datasets):
        print(data_name)
        dataset = np.genfromtxt("../datasets/%s.csv" % (data_name) , delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for preproc_id, preproc_name in enumerate(preprocs):
                if preprocs[preproc_name] == None:
                    X_res, y_res = X[train], y[train]
                else:
                    X_res, y_res = preprocs[preproc_name].fit_resample(X[train],y[train])

                for clf_id, clf_name in enumerate(clfs):
                    clf = clfs[clf_name]
                    clf.fit(X_res, y_res)
                    y_pred = clf.predict(X[test])
                    # Output with format DATAxPREPROCSxFOLDxMETRICS
                    for m_id, m_name in enumerate(metrics):
                        scores[data_id, preproc_id, fold_id, m_id] = metrics[m_name](y[test],y_pred)

    mean_scores = np.mean(scores, axis=2)
    ranks = []
    mean_ranks = []
    for m_id, metric in enumerate(metrics):
        scores_ = mean_scores[:, :, m_id]

        rank = []
        for ms in scores_:
            rank.append(rankdata(ms).tolist())
        rank = np.array(rank)
        ranks.append(rank)
        mean_rank = np.mean(rank, axis=0)
        mean_ranks.append(mean_rank)
    ranks = np.array(ranks)
    mean_ranks = np.array(mean_ranks)


    # Save results 
    np.save('../results/GNB_RanksUndersampling', ranks)
    np.save('../results/GNB_Undersampling', scores)
    