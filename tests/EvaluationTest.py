import sys
import warnings
from strlearn.metrics import balanced_accuracy_score, recall
from sklearn.neighbors import KNeighborsClassifier
sys.path.append("../ensembles")
from Bagging import BaggingClassifier
sys.path.append("../evaluation")
from Evaluator import Evaluator
from StatisticTest import StatisticTest

"""Test e2e experiment evaluation"""


def main():
    # Ignore warnings
    warnings.filterwarnings("ignore")
    # Classifiers
    clfs = {
        'Bagging LSVC5' : BaggingClassifier(KNeighborsClassifier(), n_estimators=20),
    }

    # Datasets
    datasets = ['ecoli2']

    # Metrics
    metrics = {
        'BAC' : balanced_accuracy_score,
        'RECALL' : recall
    }

    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=1410, metrics=metrics)
    ev.process(clfs, 'Test')
    st = StatisticTest(ev)
    st.process('Test')

if __name__=='__main__':
    main()
    