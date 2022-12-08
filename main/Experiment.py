import sys
import warnings
from strlearn.metrics import balanced_accuracy_score, geometric_mean_score_1, f1_score, precision, recall, specificity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
sys.path.append("../ensembles")
from Bagging import BaggingClassifier
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

N_ESTIMATORS = 50
RANDOM_STATE=1410
SUBSPACES=5

def evaluation(base_estimator, n_estimators, name):
    """Evaluation"""    
    # Classificators
    clfs = {
        'Bagging' : BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=RANDOM_STATE),
        'RSM' : RandomSubspaceMethod(base_estimator=base_estimator, n_estimators=n_estimators, random_state=RANDOM_STATE, n_subspace_features=SUBSPACES),
        'RSP' : RandomSamplePartition(base_estimator=base_estimator, n_estimators=n_estimators, random_state=RANDOM_STATE, n_subspace_features=SUBSPACES),
        'OB' : OB(base_estimator=base_estimator, n_estimators=n_estimators, random_state=RANDOM_STATE),
        'ORSM' : ORSM(base_estimator=base_estimator, n_estimators=n_estimators, random_state=RANDOM_STATE, n_subspace_features=SUBSPACES),
        'ORSP' : ORSP(base_estimator=base_estimator, n_estimators=n_estimators, random_state=RANDOM_STATE, n_subspace_features=SUBSPACES),
        'UB' : UB(base_estimator=base_estimator, n_estimators=n_estimators, random_state=RANDOM_STATE),
        'URSM' : URSM(base_estimator=base_estimator, n_estimators=n_estimators, random_state=RANDOM_STATE, n_subspace_features=SUBSPACES),
        'URSP' : URSP(base_estimator=base_estimator, n_estimators=n_estimators, random_state=RANDOM_STATE, n_subspace_features=SUBSPACES)
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
        'glass2',
        'glass4',
        'led7digit-0-2-4-5-6-7-8-9_vs_1',
        'new-thyroid1',
        'page-blocks-1-3_vs_4',
        'shuttle-6_vs_2-3',
        'vowel0',
        'yeast-0-2-5-7-9_vs_3-6-8',
        'yeast-0-3-5-9_vs_7-8',
        'yeast-1-2-8-9_vs_7',
        'yeast-2_vs_8',
        'yeast4',
        'yeast5',
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
    ev = Evaluator(datasets=datasets, storage_dir="results", random_state=RANDOM_STATE, metrics=metrics)
    ev.process(clfs, result_name=f"Scores_{name}")
    ev.process_ranks(result_name=f"Ranks_{name}")
    st = StatisticTest(ev)
    st.process(table_name=f"T_student_{name}")
    st.rank_process(table_name=f"Wilcoxon_{name}")

def main():
    """Main function"""
    #GaussianNB
    print("\n####################GaussianNB##################\n")
    evaluation(base_estimator=GaussianNB(), n_estimators=N_ESTIMATORS, name='GNB')

    #kNN
    print("\n####################kNN##################\n")
    evaluation(base_estimator=KNeighborsClassifier(), n_estimators=N_ESTIMATORS, name='KNN')

    #SVC
    print("\n####################SVC##################\n")
    evaluation(base_estimator=SVC(random_state=RANDOM_STATE), n_estimators=N_ESTIMATORS, name='SVC')

    #DecisionTreeClassifier
    print("\n####################DecisionTree##################\n")
    evaluation(base_estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=N_ESTIMATORS, name='DecisionTreeClassifier')



if __name__=='__main__':
    main()