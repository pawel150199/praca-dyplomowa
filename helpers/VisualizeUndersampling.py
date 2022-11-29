
import numpy as np
import sys
from sklearn.datasets import make_blobs, make_classification, make_moons
sys.path.append('../preprocessing')
from ModifiedClusterCentroids import ModifiedClusterCentroids
import matplotlib.pyplot as plt 
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, OneSidedSelection, CondensedNearestNeighbour

"""Kod umozliwia zaprezentowanie wizualizacji działania algorytmów wykorzystanych w eksperymencie"""
# Pobieranie danych
#datasets = 'sonar'
#dataset = np.genfromtxt("datasets/%s.csv" % (datasets), delimiter=',')
#X = dataset[:, :-1]
#y = dataset[:, -1].astype(int)

# Generowanie syntetycznego zbioru danych
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights= [0.2, 0.8]
)

# MCC - DBSCAN - const
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='DBSCAN')
X_DBSCAN, y_DBSCAN= preproc.fit_resample(X,y)

# MCC - OPTICS - const
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='OPTICS')
X_OPTICS, y_OPTICS= preproc.fit_resample(X,y)

# MCC - DBSCAN - auto
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='DBSCAN')
X_DBSCAN_auto, y_DBSCAN_auto= preproc.fit_resample(X,y)

# MCC - OPTICS - auto
preproc = ModifiedClusterCentroids(CC_strategy='const', cluster_algorithm='OPTICS')
X_OPTICS_auto, y_OPTICS_auto= preproc.fit_resample(X,y)

# RUS
preproc = RandomUnderSampler(random_state=1234)
X_RUS, y_RUS = preproc.fit_resample(X,y)

# CC
preprocs = ClusterCentroids(random_state=1234)
X_CC, y_CC = preproc.fit_resample(X,y)

# NearMiss
preprocs = NearMiss(version=1)
X_NM, y_NM = preproc.fit_resample(X,y)

# OSS
preprocs = OneSidedSelection(random_state=1234)
X_OSS, y_OSS = preproc.fit_resample(X,y)

# CNN
preprocs = CondensedNearestNeighbour(random_state=1234)
X_CNN, y_CNN = preproc.fit_resample(X,y)

# Vizualization
fig, ax = plt.subplots(2,5, figsize=(18,9))
# Original dataset
ax[0,0].scatter(*X.T, c=y)
ax[0,0].set_xlim(-5,5)
ax[0,0].set_ylim(-5,5)
ax[0,0].set_xlabel('x1')
ax[0,0].set_ylabel('x2')
ax[0,0].set_title('Oryginalny zbiór' )

# MCC DBSCAN-const
ax[0,1].scatter(*X_DBSCAN.T, c=y_DBSCAN)
ax[0,1].set_xlim(-5,5)
ax[0,1].set_ylim(-5,5)
ax[0,1].set_xlabel('x1')
ax[0,1].set_ylabel('x2')
ax[0,1].set_title('MCC DBSCAN-const')

# MCC OPTICS-const
ax[0,3].scatter(*X_OPTICS.T, c=y_OPTICS)
ax[0,3].set_xlim(-5,5)
ax[0,3].set_ylim(-5,5)
ax[0,3].set_xlabel('x1')
ax[0,3].set_ylabel('x2')
ax[0,3].set_title('MCC OPTICS-const')

# RUS
ax[1,0].scatter(*X_RUS.T, c=y_RUS)
ax[1,0].set_xlim(-5,5)
ax[1,0].set_ylim(-5,5)
ax[1,0].set_xlabel('x1')
ax[1,0].set_ylabel('x2')
ax[1,0].set_title('RUS')

# CC
ax[1,1].scatter(*X_CC.T, c=y_CC)
ax[1,1].set_xlim(-5,5)
ax[1,1].set_ylim(-5,5)
ax[1,1].set_xlabel('x1')
ax[1,1].set_ylabel('x2')
ax[1,1].set_title('CC')

# NearMiss
ax[1,2].scatter(*X_NM.T, c=y_NM)
ax[1,2].set_xlim(-5,5)
ax[1,2].set_ylim(-5,5)
ax[1,2].set_xlabel('x1')
ax[1,2].set_ylabel('x2')
ax[1,2].set_title('NearMiss')

# MCC OPTICS-auto
ax[0,4].scatter(*X_OPTICS_auto.T, c=y_OPTICS_auto)
ax[0,4].set_xlim(-5,5)
ax[0,4].set_ylim(-5,5)
ax[0,4].set_xlabel('x1')
ax[0,4].set_ylabel('x2')
ax[0,4].set_title('MCC OPTICS-auto')

# CNN
ax[1,3].scatter(*X_CNN.T, c=y_CNN)
ax[1,3].set_xlim(-5,5)
ax[1,3].set_ylim(-5,5)
ax[1,3].set_xlabel('x1')
ax[1,3].set_ylabel('x2')
ax[1,3].set_title('CNN')

# MCC DBSCAN-auto
ax[0,2].scatter(*X_DBSCAN_auto.T, c=y_DBSCAN_auto)
ax[0,2].set_xlim(-5,5)
ax[0,2].set_ylim(-5,5)
ax[0,2].set_xlabel('x1')
ax[0,2].set_ylabel('x2')
ax[0,2].set_title('MCC DBSCAN-auto')

# OSS
ax[1,4].scatter(*X_OSS.T, c=y_OSS)
ax[1,4].set_xlim(-5,5)
ax[1,4].set_ylim(-5,5)
ax[1,4].set_xlabel('x1')
ax[1,4].set_ylabel('x2')
ax[1,4].set_title('OSS')

plt.tight_layout()
plt.savefig("../images/UnersamplingVisualization.png", dpi=600)
#y_new = np.reshape(y_new, (X_new.shape[0], 1))