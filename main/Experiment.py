import sys
import warnings
from strlearn.metrics import balanced_accuracy_score, geometric_mean_score_1, f1_score, precision, recall, specificity
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
from HybridEnsemble import HybridEnsemble
from OHE import OHE
from UHE import UHE
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
        'Bagging' : BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators),
        'RSM' : RandomSubspaceMethod(base_estimator=base_estimator, n_estimators=n_estimators),
        'RSP' : RandomSamplePartition(base_estimator=base_estimator, n_estimators=n_estimators),
        'HE' : HybridEnsemble(base_estimator=base_estimator, n_estimators=n_estimators, boosting_estimators=n_estimators),
        'OHE' : OHE(base_estimator=base_estimator, n_estimators=n_estimators, boosting_estimators=n_estimators),
        'OB' : OB(base_estimator=base_estimator, n_estimators=n_estimators),
        'ORSE' : ORSM(base_estimator=base_estimator, n_estimators=n_estimators),
        'ORSP' : ORSP(base_estimator=base_estimator, n_estimators=n_estimators),
        'UB' : UB(base_estimator=base_estimator, n_estimators=n_estimators),
        'URSE' : URSM(base_estimator=base_estimator, n_estimators=n_estimators),
        'URSP' : URSP(base_estimator=base_estimator, n_estimators=n_estimators),
        'UHE' : UHE(base_estimator=base_estimator, n_estimators=n_estimators, boosting_estimators=n_estimators)
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
        'glass-0-1-6_vs_5',
        'glass2',
        'glass4',
        'glass5',
        'new-thyroid1',
        'newthyroid2',
        'poker-9_vs_7',
        'shuttle-6_vs_2-3',
        'winequality-white-9_vs_4',
        'yeast-0-2-5-7-9_vs_3-6-8',
        'yeast-0-3-5-9_vs_7-8',
        'yeast-2_vs_8',
        'yeast6'
    ]

    # Metrics
    metrics = {
        'BAC' : balanced_accuracy_score,
        'G-mean' : geometric_mean_score_1,
        'F1-score' : f1_score,
        'precision' : precision,
        'recall' : recall,
        'specificity' : specificity
    }   
    print("Evaluation")
    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=1410, metrics=metrics)
    ev.process(clfs, name)
    ev.process_ranks()
    st = StatisticTest(ev)
    st.process(name)
    st.rank_process(name)

def main():
    """Main function"""
    #GaussianNB
    print("\n####################GaussianNB##################\n")
    evaluation(base_estimator=GaussianNB(), n_estimators=N_ESTIMATORS, name='GNB')
    #SVC
    print("\n####################SVC##################\n")
    evaluation(base_estimator=SVC(random_state=1410), n_estimators=N_ESTIMATORS, name='SVC')
    #Linear SVC
    print("\n####################Linear SVC##################\n")
    evaluation(base_estimator=LinearSVC(random_state=1410), n_estimators=N_ESTIMATORS, name='LinearSVC')
    #DecisionTreeClassifier
    print("\n####################DecisionTree##################\n")
    evaluation(base_estimator=DecisionTreeClassifier(random_state=1410), n_estimators=N_ESTIMATORS, name='DecisionTreeClassifier')

if __name__=='__main__':
    main()