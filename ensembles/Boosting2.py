import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone

class AdaBoost(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=DecisionTreeClassifier(), M=100):
        self.base_estimator = base_estimator
        self.M = M
        self.ensembles = []
        self.alphas = []
    
    def fit(self, X, y):
        N, D = X.shape
        
        # Instantiate stuff
        weights = np.repeat(1/N, N)
        self.yhats = np.empty((N, self.M))
        
        for t in range(self.M):
            
            # Calculate stuff
            clf = self.base_estimator
            clf.fit(X, y, sample_weight=weights)
            yhat_t = clf.predict(X)
            epsilon = sum(weights*(yhat_t != y))/sum(weights)
            alpha = np.log( (1-epsilon)/epsilon )
            weights = np.array([w*(1-epsilon)/epsilon if yhat_t[i] != y[i]
                                    else w for i, w in enumerate(weights)])
            # Append stuff
            self.ensembles.append(clf)
            self.alphas.append(alpha)
            self.yhats[:,t] = yhat_t
            
        self.yhat = np.sign(np.dot(self.yhats, self.alphas))
        
    def predict(self, X):
        yhats = np.zeros(len(X))
        for e, e_name in enumerate(self.ensembles):
            yhats_tree = e_name.predict(X)
            yhats += yhats_tree * self.alphas[e]
        return np.sign(yhats)