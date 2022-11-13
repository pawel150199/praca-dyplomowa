import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import base
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone

class AdaBoostClassifier(DecisionTreeClassifier, ClassifierMixin):
    def __init__(self, n_estimators=50, base_estimator=None):
        self.n_estimators = n_estimators
        self.models = [None]*n_estimators
        if base_estimator == None:
            base_estimator = DecisionTreeClassifier(max_depth=1)
        self.base_estimator = base_estimator
        self.estimator_errors_ = []

    def __index_to_vector(y,k,labelDict):
        y_new = []
        for yi in y:
            i = labelDict[yi]
            v = np.ones(k)*(-1/(k-1))
            v[i] = 1
            y_new.append(v)
    def index_to_label(i,clf):
        return clf.classes[i]

    def fit(self,X,y):
        
        X = np.float64(X)
        N = len(y)
        w = np.array([1/N for i in range(N)])
        
        self.createLabelDict(np.unique(y))
        k = len(self.classes)
        
        for m in range(self.n_estimators):
            
            Gm = base.clone(self.base_estimator).\
                            fit(X,y,sample_weight=w).predict
            
            incorrect = Gm(X) != y
            errM = np.average(incorrect,weights=w,axis=0)
            
            self.estimator_errors_.append(errM)
            
            BetaM = (np.log((1-errM)/errM)+np.log(k-1))
            
            w *= np.exp(BetaM*incorrect*(w > 0))
            
            self.models[m] = (BetaM,Gm)
            
    def createLabelDict(self,classes):
        self.labelDict = {}
        self.classes = classes
        for i,cl in enumerate(classes):
            self.labelDict[cl] = i

    def predict(self,X):
        k = len(self.classes)
        y_pred = sum(Bm*self.__index_to_vector(Gm(X),k,self.labelDict) \
                             for Bm,Gm in self.models)
        
        iTL = np.vectorize(self.index_to_label)
        return iTL(np.argmax(y_pred,axis=1),self)
