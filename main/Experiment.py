import sys
from strlearn.metrics import balanced_accuracy_score, recall
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB()
from sklearn.svm import SVC
sys.path.append('../ensemble')
from Bagging import BaggingClassifier
from RandomSamplePartition import RandomSamplePartition
from RandomSubspaceEnsemble import RandomSubspaceEnsemble
from UB import UB
from URSE import URSE
from URSP import URSP
from OB import OB
from ORSE import ORSE
from ORSP import ORSP
sys.path.append('../evaluation')
from Evaluator import Evaluator
from StatisticTest import StatisticTest

"""Ewaluacja eksperymentu"""

def evaluation_base_kNN():
    #Klasyfikatory
    clfs = {
        'Bagging' : BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=5),
        'RSE' : RandomSubspaceEnsemble(base_estimator=KNeighborsClassifier(), n_estimators=5),
        'RSP' : RandomSamplePartition(base_estimator=KNeighborsClassifier(), n_estimators=5),
        'OB' : OB(base_estimator=KNeighborsClassifier(), n_estimators=5),
        'ORSE' : ORSE(base_estimator=KNeighborsClassifier(), n_estimators=5),
        'ORSP' : ORSP(base_estimator=KNeighborsClassifier(), n_estimators=5),
        'UB' : UB(base_estimator=KNeighborsClassifier(), n_estimators=5),
        'URSE' : URSE(base_estimator=KNeighborsClassifier(), n_estimators=5),
        'URSP' : URSP(base_estimator=KNeighborsClassifier(), n_estimators=5),
    }

    #Zbiór danych
    datasets = ['appendicitis', 'bupa']

    #metryki
    metrics = {
        'BAC' : balanced_accuracy_score,
        'Recall' : recall
    }

    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=1410, metrics=metrics)
    ev.process(clfs, 'kNN')
    st = StatisticTest(ev)
    st.process('kNN')

def evaluation_base_SVC():
    #Klasyfikatory
    clfs = {
        'Bagging' : BaggingClassifier(base_estimator=SVC(), n_estimators=5),
        'RSE' : RandomSubspaceEnsemble(base_estimator=SVC(), n_estimators=5),
        'RSP' : RandomSamplePartition(base_estimator=SVC(), n_estimators=5),
        'OB' : OB(base_estimator=SVC(), n_estimators=5),
        'ORSE' : ORSE(base_estimator=SVC(), n_estimators=5),
        'ORSP' : ORSP(base_estimator=SVC(), n_estimators=5),
        'UB' : UB(base_estimator=SVC(), n_estimators=5),
        'URSE' : URSE(base_estimator=SVC(), n_estimators=5),
        'URSP' : URSP(base_estimator=SVC(), n_estimators=5),
    }

    #Zbiór danych
    datasets = ['appendicitis', 'bupa']

    #metryki
    metrics = {
        'BAC' : balanced_accuracy_score,
        'Recall' : recall
    }

    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=1410, metrics=metrics)
    ev.process(clfs, 'SVC')
    st = StatisticTest(ev)
    st.process('SVC')

def evaluation_base_SVC():
    #Klasyfikatory
    clfs = {
        'Bagging' : BaggingClassifier(base_estimator=SVC(), n_estimators=5),
        'RSE' : RandomSubspaceEnsemble(base_estimator=SVC(), n_estimators=5),
        'RSP' : RandomSamplePartition(base_estimator=SVC(), n_estimators=5),
        'OB' : OB(base_estimator=SVC(), n_estimators=5),
        'ORSE' : ORSE(base_estimator=SVC(), n_estimators=5),
        'ORSP' : ORSP(base_estimator=SVC(), n_estimators=5),
        'UB' : UB(base_estimator=SVC(), n_estimators=5),
        'URSE' : URSE(base_estimator=SVC(), n_estimators=5),
        'URSP' : URSP(base_estimator=SVC(), n_estimators=5),
    }

    #Zbiór danych
    datasets = ['appendicitis', 'bupa']

    #metryki
    metrics = {
        'BAC' : balanced_accuracy_score,
        'Recall' : recall
    }

    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=1410, metrics=metrics)
    ev.process(clfs, 'SVC')
    st = StatisticTest(ev)
    st.process('SVC')

if __name__=='__main__':
    main()
    