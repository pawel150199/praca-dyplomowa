import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.base import ClusterMixin
from sklearn.utils.validation import  check_X_y

class ModifiedClusterCentroids(ClusterMixin):
    """
    Modified Cluster Centroids algorithm based on clustering methods like: DBSCAN and OPTICS
    Parameters:
    n_cluster, eps, min_samples, metric, algorithm -> the same like in DBSCAN algorithm
    CC_strategy -> ('const','auto'): 
    *'const' -> reduce major class to minor ones
    *'auto' -> reduce major class using std.
    cluster_algorithm -> define clustering method
    min_samples -> parameter used in OPTICS method
    max_eps -> parameter used in OPTICS method and define analyzing area (natively all area is analyzing)
    """

    def __init__(self, CC_strategy='const', eps=0.5, metric='euclidean', algorithm='auto', min_samples=5, cluster_algorithm='DBSCAN', max_eps=np.inf):
        self.eps = eps
        self.cluster_algorithm = cluster_algorithm
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.CC_strategy = CC_strategy
        self.max_eps = max_eps

    def rus(self, X, y, n_samples):
        """Choose random samples"""
        X_inc = np.random.choice(len(X), size=n_samples, replace=False)
        return X[X_inc], y[X_inc]

    def fit_resample(self, X, y):
        """Resample"""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.n_features = X.shape[1]

        # Define major and minor class
        l, c = np.unique(y, return_counts=True)
        minor_probas = np.amin(c)
        minor_class = l[minor_probas==c]
        major_class = l[minor_probas!=c]
        # Table for data after resampling
        X_resampled = []
        y_resampled = []

        if self.CC_strategy == 'const':
            """In case of 'const' parameter major class will be reduce to minor class"""

            # Clustering DBSCAN or OPTICS
            if self.cluster_algorithm == 'DBSCAN':
                clustering = DBSCAN(eps=self.eps, metric=self.metric, algorithm=self.algorithm).fit(X[y!=minor_class])
            elif self.cluster_algorithm == 'OPTICS':
                clustering = OPTICS(min_samples=self.min_samples).fit(X[y!=minor_class])
            else:
                raise ValueError('Incorrect cluster_algorithm!')

            # Define apriori probability between clusters
            l, c = np.unique(clustering.labels_, return_counts=True)

            # Define how much data will be remove
            if len(l)==1:
                new_c=int(minor_probas)
                X_selected, y_selected = self.rus(X[y!=minor_class][clustering.labels_==l], y[y!=minor_class][clustering.labels_==l], n_samples=new_c)
                X_resampled.append(X_selected)
                y_resampled.append(y_selected)

                # Add minor class
                X_resampled.append(X[y==minor_class])
                y_resampled.append(y[y==minor_class])
                X_resampled=np.concatenate(X_resampled)
                y_resampled=np.concatenate(y_resampled)
                return X_resampled, y_resampled

            else:
                # Calculate apriori probability
                prob = [i/c.sum() for i in c]
                new_c = [prob[i]*minor_probas for i in range(0, len(c))]
                new_c = np.ceil(new_c)
                
                # Undersampling inside clusters
                for label, n_samples in zip(l, new_c):
                    n_samples = int(n_samples)
                    X_selected, y_selected = self.rus(X[y!=minor_class][clustering.labels_==label], y[y!=minor_class][clustering.labels_==label], n_samples=n_samples)
                    X_resampled.append(X_selected)
                    y_resampled.append(y_selected)

                # Add minor class
                X_resampled.append(X[y==minor_class])
                y_resampled.append(y[y==minor_class])
                X_resampled=np.concatenate(X_resampled)
                y_resampled=np.concatenate(y_resampled)
                return X_resampled, y_resampled

        elif self.CC_strategy == 'auto':
            """In case of 'auto' parameter algorithm choose reduction level by std value"""

            # Clustering
            if self.cluster_algorithm == 'DBSCAN':
                clustering = DBSCAN(eps=self.eps, metric=self.metric, algorithm=self.algorithm).fit(X[y!=minor_class])
            elif self.cluster_algorithm == 'OPTICS':
                clustering = OPTICS(min_samples=self.min_samples, max_eps=self.max_eps).fit(X[y!=minor_class])
            else:
                raise ValueError('Niepoprawna wartośc cluster_algorithm!')

            # STD calculations
            l, c = np.unique(clustering.labels_, return_counts=True)
            # If exist one cluster data will be reduce to amount of minor class 
            if len(l)==1:
                X_selected, y_selected = self.rus(X[y!=minor_class][clustering.labels_==l], y[y!=minor_class][clustering.labels_==l], n_samples=int(minor_probas))
                X_resampled.append(X_selected)
                y_resampled.append(y_selected)

                # Add minor class
                X_resampled.append(X[y==minor_class])
                y_resampled.append(y[y==minor_class])
                X_resampled=np.concatenate(X_resampled)
                y_resampled=np.concatenate(y_resampled)
                return X_resampled, y_resampled 

            else:
                std = []
                for i in l:
                    std.append(np.std(X[y!=minor_class][clustering.labels_==i].flatten()))
                std=np.array(std)
                std=std/std.sum()
                # Choose samples frorm clusters with lower std
                std = [1 - i for i in std]
                std = np.array(std)

                # Higher std is equal to less samples from clusters
                new_c = std*c
                new_c = np.ceil(new_c)
                for label, n_samples in zip(l, new_c):
                    n_samples = int(n_samples)
                    X_selected, y_selected = self.rus(X[y!=minor_class][clustering.labels_==label], y[y!=minor_class][clustering.labels_==label], n_samples=n_samples)
                    X_resampled.append(X_selected)
                    y_resampled.append(y_selected)

                # Add minor class
                X_resampled.append(X[y==minor_class])
                y_resampled.append(y[y==minor_class])
                X_resampled=np.concatenate(X_resampled)
                y_resampled=np.concatenate(y_resampled)
                return X_resampled, y_resampled 

        else:
            raise ValueError("Incorrect CC_strategy!")
            