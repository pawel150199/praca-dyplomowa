import sys
import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone
sys.path.append("../preprocessing")
from ModifiedClusterCentroids import ModifiedClusterCentroids

"""Undersampled AdaBoost - przykładowa implementacja Boostingu z undersamplingiem"""

class UBO(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator=DecisionTreeClassifier(), M=10):
        self.base_estimator = base_estimator
        self.M = M
        self.alphas = []
        self.ensemble = []
        self.training_errors = []
        self.prediction_errors = []
    
    def compute_error(self, y, y_pred, w_i):
        """Obliczenie błędu słabych klasyfikatorów
        Argumenty:
        y: aktualna wartośc
        y_pred: wartośc oczekiwana
        w_i: wagi poszczególnych obserwacji
        """
        return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

    def error_rates(self, X, y):
        """Stosunek błędów dla kazdego slabego klasyfikatora"""
        self.prediction_errors = []
        for m in range(self.M):
            y_pred_m = self.ensemble[m].predict(X)
            error_m = self.compute_error(y=y, y_pred=y_pred_m, w_i=np.ones(len(y)))
            self.prediction_errors.append(error_m)

    def compute_alpha(self, error):
        """Obliczanie wartości alfa
        Argumenty:
        error: liczebnośc błędów pochodzących z klasyfikatorów podstawowych"""
        return np.log((1 - error) / error)

    def update_weights(self, w_i, alpha, y, y_pred):
        """Uaktualnienia wag po kadej iteracji boostingu"""
        return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
    
    def undersample(self, X, y):
        preproc = ModifiedClusterCentroids()
        X_new, y_new = preproc.fit_resample(X,y)
        return X_new, y_new
    
    def fit(self, X, y):
        """Trening"""

        # Czyszczenie przed wywołaniem
        self.alphas = []
        self.training_errors = []

        # Undersampling
        X, y = self.undersample(X, y)

        # Boosting
        for m in range(self.M):
            # Ustawienie wag dla pierwszej iteracji
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                # Uaktualnienie wag
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)

            G_m = clone(self.base_estimator)
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            
            # Lista z wynikami dla klasyfikatora podstawowego
            self.ensemble.append(G_m)

            # Obliczenie błedu
            error_m = self.compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            #Obliczenie alfa
            alpha_m = self.compute_alpha(error_m)
            self.alphas.append(alpha_m)
            
    def predict(self, X):
        """Predykcja"""
        weak_preds = np.zeros((len(X), self.M))
        for m in range(self.M):
            y_pred_m = self.ensemble[m].predict(X) * self.alphas[m]
            weak_preds[:,m] = y_pred_m
        
        # Predykcja
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred
