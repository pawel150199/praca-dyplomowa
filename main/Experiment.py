import warnings
from strlearn.metrics import balanced_accuracy_score, recall
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from execution.ensembles.Bagging import BaggingClassifier
from execution.ensembles.RandomSubspaceEnsemble import RandomSubspaceEnsemble
from execution.ensembles.RandomSamplePartition import RandomSamplePartition
from execution.ensembles.OB import OB
from execution.ensembles.ORSE import ORSE
from execution.ensembles.ORSP import ORSP
from execution.ensembles.UB import UB
from execution.ensembles.URSE import URSE
from execution.ensembles.URSP import URSP
from execution.evaluation.Evaluator import Evaluator
from execution.evaluation.StatisticTest import StatisticTest


"""Experiment evaluation"""

# Ignore warnings
warnings.filterwarnings("ignore")

N_ESTIMATORS = 5

def evaluation(base_estimator, n_estimators, name):
    """Evaluation"""
    # Classificators
    clfs = {
        'Bagging' : BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators),
        'RSE' : RandomSubspaceEnsemble(base_estimator=base_estimator, n_estimators=n_estimators),
        'RSP' : RandomSamplePartition(base_estimator=base_estimator, n_estimators=n_estimators),
        'OB' : OB(base_estimator=base_estimator, n_estimators=n_estimators),
        'ORSE' : ORSE(base_estimator=base_estimator, n_estimators=n_estimators),
        'ORSP' : ORSP(base_estimator=base_estimator, n_estimators=n_estimators),
        'UB' : UB(base_estimator=base_estimator, n_estimators=n_estimators),
        'URSE' : URSE(base_estimator=base_estimator, n_estimators=n_estimators),
        'URSP' : URSP(base_estimator=base_estimator, n_estimators=n_estimators),
    }
    # Datasets
    datasets = ['ecoli2']

    # Metrics
    metrics = {
        'BAC' : balanced_accuracy_score,
        'Recall' : recall
    }
    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=1410, metrics=metrics)
    ev.process(clfs, name)
    st = StatisticTest(ev)
    st.process(name)

def main():
    """Main function"""
    #GaussianNB
    evaluation(base_estimator=GaussianNB(), n_estimators=N_ESTIMATORS, name='GNB')
    #kNN
    evaluation(base_estimator=KNeighborsClassifier(), n_estimators=N_ESTIMATORS, name='kNN')
    #SVC
    evaluation(base_estimator=SVC(random_state=1410), n_estimators=N_ESTIMATORS, name='SVC')
    #Linear SVC
    evaluation(base_estimator=LinearSVC(random_state=1410), n_estimators=N_ESTIMATORS, name='LinearSVC')
    #DecisionTreeClassifier
    evaluation(base_estimator=DecisionTreeClassifier(random_state=1410), n_estimators=N_ESTIMATORS, name='DecisionTreeClassifier')
    #MLP
    evaluation(base_estimator=MLPClassifier(random_state=1410), n_estimators=N_ESTIMATORS, name='MLP')

if __name__=='__main__':
    main()