import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone 


"""Implementation of Adaptive Boosting"""


class AdaBoostClassifier(ClassifierMixin, BaseEnsemble):
    
    def __init__(self,base_estimator=DecisionTreeClassifier(max_depth=1),n_estimators=50, random_state=None):
        self.n_estimators = n_estimators
        self.models = [None]*n_estimators
        self.base_estimator = base_estimator
        self.estimator_errors_ = []
        self.random_state = random_state
        np.random.seed(self.random_state)

    def __indexToVector(self,y,k,labelDict):
        """Change indexes to vector"""
        y_new = []
        for y_i in y:
            i = labelDict[y_i]
            v = np.ones(k)*(-1/(k-1))
            v[i] = 1
            y_new.append(v)
        return np.array(y_new)

    def __indexToLabel(self,i,clf):
        """Change indexes to labels"""
        return clf.classes[i]
    
    def __createLabelDict(self,classes):
        """Create dictionary with labels"""
        self.labelDict = {}
        self.classes = classes
        for i,cl in enumerate(classes):
            self.labelDict[cl] = i
    
    def __oversample(self, X, y):
        """Oversampling"""
        preproc = SMOTE(random_state=1410)
        X_new, y_new = preproc.fit_resample(X,y)
        return X_new, y_new

    def fit(self,X,y):
        """Fitting model"""
        X, y = self.__oversample(X,y)
        X = np.float64(X)
        N = len(y)
        w = np.array([1/N for i in range(N)])
        
        self.__createLabelDict(np.unique(y))
        k = len(self.classes)
        
        for m in range(self.n_estimators):

            Gm = clone(self.base_estimator).fit(X,y,sample_weight=w).predict

            incorrect = Gm(X) != y
            errM = np.average(incorrect,weights=w,axis=0)
            self.estimator_errors_.append(errM)

            BetaM = (np.log((1 - errM)/errM) + np.log(k - 1))
            w *= np.exp(BetaM * incorrect * (w > 0))
            self.models[m] = (BetaM,Gm)

    def predict(self,X):
        "Predict labels"
        k = len(self.classes)
        y_pred = sum(Bm * self.__indexToVector(Gm(X),k,self.labelDict) for Bm,Gm in self.models)
        
        itl = np.vectorize(self.__indexToLabel)
        return itl(np.argmax(y_pred,axis=1),self)
