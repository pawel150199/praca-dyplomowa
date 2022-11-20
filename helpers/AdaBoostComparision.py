import sys
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from strlearn.metrics import balanced_accuracy_score
from sklearn.ensemble import AdaBoostClassifier as AdaBoostSklearn
sys.path.append("../ensembles")
from AdaBoost import AdaBoostClassifier
sys.path.append("../evaluation")
from Evaluator import Evaluator
from StatisticTest import StatisticTest


# Ignore warnings
warnings.filterwarnings("ignore")

N_ESTIMATORS = 5

def evaluation(base_estimator, n_estimators, name):
    """Evaluation"""
    algorithm = ''
    print(base_estimator)
    if str(base_estimator) == "SVC(random_state=1410)" or str(base_estimator) == "LinearSVC(random_state=1410)":
        algorithm = 'SAMME'
    else:
        algorithm = 'SAMME.R'

    print(algorithm)
    # Classificators
    clfs = {
        'AdaBoost' : AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators),
        'AdaBoost-sklearn' : AdaBoostSklearn(base_estimator=base_estimator, n_estimators=n_estimators, algorithm=algorithm)
    }
    # Dataset
    datasets = ['ecoli2']

    # Metric
    metrics = {'BAC' : balanced_accuracy_score}
    ev = Evaluator(datasets=datasets, storage_dir="results", metrics=metrics)
    ev.process(clfs, name)
    st = StatisticTest(ev)
    st.process(name)

def main():
    """Main function"""
    #GaussianNB
    print("\nGaussianNB\n")
    evaluation(base_estimator=GaussianNB(), n_estimators=N_ESTIMATORS, name='AdaBoostGNB')
    #SVC
    print("\nSVC\n")
    evaluation(base_estimator=SVC(random_state=1410), n_estimators=N_ESTIMATORS, name='AdaBoostSVC')
    #Linear SVC
    print("\nLinear SVC\n")
    evaluation(base_estimator=LinearSVC(random_state=1410), n_estimators=N_ESTIMATORS, name='AdaBoostLinearSVC')
    #DecisionTreeClassifier
    print("\nDecisionTree\n")
    evaluation(base_estimator=DecisionTreeClassifier(random_state=1410, max_depth=1), n_estimators=N_ESTIMATORS, name='AdaBoostDecisionTreeClassifier')
if __name__=='__main__':
    main()