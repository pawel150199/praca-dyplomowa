import sys
from strlearn.metrics import balanced_accuracy_score, recall
from sklearn.neural_network import MLPClassifier
sys.path.append('../ensemble')
import warnings
from Bagging import BaggingClassifier
sys.path.append('../evaluation')
from Evaluator import Evaluator
from StatisticTest import StatisticTest

"""Test e2e experiment evaluation"""

# Ignore warnings
warnings.filterwarnings("ignore")

def main():
    # Classifiers
    clfs = {
        'Bagging LSVC5' : BaggingClassifier(MLPClassifier(hidden_layer_sizes=10), n_estimators=1),
        'Bagging LSVC10': BaggingClassifier(MLPClassifier(hidden_layer_sizes=10), n_estimators=9),
        'Bagging LSVC15': BaggingClassifier(MLPClassifier(hidden_layer_sizes=10), n_estimators=12)
    }

    # Datasets
    datasets = ['glass']

    # Metrics
    metrics = {
        'BAC' : balanced_accuracy_score,
        'RECALL' : recall
    }

    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=1410, metrics=metrics)
    ev.process(clfs, 'nowy')
    st = StatisticTest(ev)
    st.process('MLP')

if __name__=='__main__':
    main()
    