
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

dim = np.arange(-4,5,1)

# Vizualization
fig, ax = plt.subplots(5,2, figsize=(14,18))
# Original dataset
ax[0,0].scatter(*X.T, c=y, cmap='RdBu')
ax[0,0].set_xlim(-4,4)
ax[0,0].set_ylim(-4,4)
ax[0,0].set_xlabel('x1',fontsize=16)
ax[0,0].set_ylabel('x2',fontsize=16)
ax[0,0].set_title('Oryginalny zbiór', fontsize=18)
ax[0,0].grid()

# MCC DBSCAN-const
ax[0,1].scatter(*X_DBSCAN.T, c=y_DBSCAN, cmap='RdBu')
ax[0,1].set_xlim(-4,4)
ax[0,1].set_ylim(-4,4)
ax[0,1].set_xlabel('x1',fontsize=16)
ax[0,1].set_ylabel('x2',fontsize=16)
ax[0,1].set_title('MCC DBSCAN-const', fontsize=18)
ax[0,1].grid()

# MCC OPTICS-const
ax[1,0].scatter(*X_OPTICS.T, c=y_OPTICS, cmap='RdBu')
ax[1,0].set_xlim(-4,4)
ax[1,0].set_ylim(-4,4)
ax[1,0].set_xlabel('x1',fontsize=16)
ax[1,0].set_ylabel('x2',fontsize=16)
ax[1,0].set_title('MCC OPTICS-const', fontsize=18)
ax[1,0].grid()

# MCC DBSCAN-auto
ax[1,1].scatter(*X_DBSCAN_auto.T, c=y_DBSCAN_auto, cmap='RdBu')
ax[1,1].set_xlim(-4,4)
ax[1,1].set_ylim(-4,4)
ax[1,1].set_xlabel('x1',fontsize=16)
ax[1,1].set_ylabel('x2',fontsize=16)
ax[1,1].set_title('MCC DBSCAN-auto', fontsize=18)
ax[1,1].grid()

# MCC OPTICS-auto
ax[2,0].scatter(*X_OPTICS_auto.T, c=y_OPTICS_auto, cmap='RdBu')
ax[2,0].set_xlim(-4,4)
ax[2,0].set_ylim(-4,4)
ax[2,0].set_xlabel('x1',fontsize=16)
ax[2,0].set_ylabel('x2',fontsize=16)
ax[2,0].set_title('MCC OPTICS-auto', fontsize=18)
ax[2,0].grid()

# RUS
ax[2,1].scatter(*X_RUS.T, c=y_RUS, cmap='RdBu')
ax[2,1].set_xlim(-4,4)
ax[2,1].set_ylim(-4,4)
ax[2,1].set_xlabel('x1',fontsize=16)
ax[2,1].set_ylabel('x2',fontsize=16)
ax[2,1].set_title('RUS', fontsize=18)
ax[2,1].grid()

# CC
ax[3,0].scatter(*X_CC.T, c=y_CC, cmap='RdBu')
ax[3,0].set_xlim(-4,4)
ax[3,0].set_ylim(-4,4)
ax[3,0].set_xlabel('x1',fontsize=16)
ax[3,0].set_ylabel('x2',fontsize=16)
ax[3,0].set_title('CC', fontsize=18)
ax[3,0].grid()

# NearMiss
ax[3,1].scatter(*X_NM.T, c=y_NM, cmap='RdBu')
ax[3,1].set_xlim(-4,4)
ax[3,1].set_ylim(-4,4)
ax[3,1].set_xlabel('x1',fontsize=16)
ax[3,1].set_ylabel('x2',fontsize=16)
ax[3,1].set_title('NearMiss', fontsize=18)
ax[3,1].grid()

# CNN
ax[4,0].scatter(*X_CNN.T, c=y_CNN, cmap='RdBu')
ax[4,0].set_xlim(-4,4)
ax[4,0].set_ylim(-4,4)
ax[4,0].set_xlabel('x1',fontsize=16)
ax[4,0].set_ylabel('x2',fontsize=16)
ax[4,0].set_title('CNN', fontsize=18)
ax[4,0].grid()

# OSS
ax[4,1].scatter(*X_OSS.T, c=y_OSS, cmap='RdBu')
ax[4,1].set_xlim(-4,4)
ax[4,1].set_ylim(-4,4)
ax[4,1].set_xlabel('x1',fontsize=16)
ax[4,1].set_ylabel('x2',fontsize=16)
ax[4,1].set_title('OSS', fontsize=18)
ax[4,1].grid()


plt.tight_layout()
plt.savefig("../images/UnersamplingVisualization.png", dpi=600)
#y_new = np.reshape(y_new, (X_new.shape[0], 1))