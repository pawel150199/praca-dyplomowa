import sys
import warnings
from strlearn.metrics import balanced_accuracy_score, recall
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
sys.path.append("../ensembles")
from Bagging import BaggingClassifier
from RSPmod import RandomSamplePartition as RSP
from RandomSubspaceMethod import RandomSubspaceMethod
from RandomSamplePartition import RandomSamplePartition
from OB import OB
from ORSM import ORSM
from ORSP import ORSP
from UB import UB
from URSM import URSM
from URSP import URSP
sys.path.append("../evaluation")
from Evaluator import Evaluator
from StatisticTest import StatisticTest


"""Experiment evaluation"""

# Ignore warnings
warnings.filterwarnings("ignore")

N_ESTIMATORS = 5

def evaluation(base_estimator, n_estimators, name):
    """Evaluation"""
    # Classificators
    clfs = {
        #'Bagging' : BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators),
        #'RSE' : RandomSubspaceEnsemble(base_estimator=base_estimator, n_estimators=n_estimators),
        'RSP' : RandomSamplePartition(base_estimator=base_estimator, n_estimators=n_estimators),
        'RSPmod' : RSP(base_estimator=base_estimator, n_estimators=n_estimators),
        #'OB' : OB(base_estimator=base_estimator, n_estimators=n_estimators),
        #'ORSE' : ORSE(base_estimator=base_estimator, n_estimators=n_estimators),
        #'ORSP' : ORSP(base_estimator=base_estimator, n_estimators=n_estimators),
        #'UB' : UB(base_estimator=base_estimator, n_estimators=n_estimators),
        #'URSE' : URSE(base_estimator=base_estimator, n_estimators=n_estimators),
        #'URSP' : URSP(base_estimator=base_estimator, n_estimators=n_estimators),
    }
    # Datasets
    datasets = ['ecoli2']

    # Metrics
    metrics = {
        'BAC' : balanced_accuracy_score,
        'Recall' : recall
    }
    print("Evaluation")
    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=1410, metrics=metrics)
    ev.process(clfs, name)
    ev.process_ranks()
    st = StatisticTest(ev)
    st.process(name)
    print("Global rank")
    st.rank_process(name)

def main():
    """Main function"""
    #GaussianNB
    print("\n####################GaussianNB##################\n")
    evaluation(base_estimator=GaussianNB(), n_estimators=N_ESTIMATORS, name='GNB')
    #kNN
    print("\n####################kNN##################\n")
    evaluation(base_estimator=KNeighborsClassifier(), n_estimators=N_ESTIMATORS, name='kNN')
    #SVC
    print("\n####################SVC##################\n")
    evaluation(base_estimator=SVC(random_state=1410), n_estimators=N_ESTIMATORS, name='SVC')
    #Linear SVC
    print("\n####################Linear SVC##################\n")
    evaluation(base_estimator=LinearSVC(random_state=1410), n_estimators=N_ESTIMATORS, name='LinearSVC')
    #DecisionTreeClassifier
    print("\n####################DecisionTree##################\n")
    evaluation(base_estimator=DecisionTreeClassifier(random_state=1410), n_estimators=N_ESTIMATORS, name='DecisionTreeClassifier')
    #MLP
    print("\n####################MLP##################\n")
    evaluation(base_estimator=MLPClassifier(random_state=1410), n_estimators=N_ESTIMATORS, name='MLP')

if __name__=='__main__':
    main()