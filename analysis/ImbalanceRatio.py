import numpy as np
import os
from tabulate import tabulate

class IR:
    """
    Class is used for generate table with imbalacement ratio of choosen datasets

    Usage:
        obj = IR(['datasetname'])
        obj.calculate()
        obj.tab(true)
    """
    def __init__(self, datasets):
        self.datasets = datasets
    
    def calculate(self):
        """Calculate IR"""
        self.scores = []
        self.y = []
        self.dataset_name = []

        for data_id, dataset in enumerate(self.datasets):
            os.chdir('../datasets')
            dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=',')
            self.y = dataset[:, -1].astype(int)
            l, c = np.unique(self.y, return_counts=True)
            minor_probas = np.amin(c)
            idx = np.where(minor_probas!=c)
            Nmax = sum(c[idx])
            Nmin = minor_probas
            IR = round((Nmax/Nmin), 2)
            self.scores.append([Nmin, Nmax, IR])
        return self

    def tab(self, save):
        """Generate tables"""
        # save -> parameter is used for choose store tables or not
        t = []
        for data_id, data_name in enumerate(self.datasets):
            t.append(['%s' % data_name] + ['%.3f' % v for v in self.scores[data_id]])
        headers = ['datasets', 'N_min', 'N_maj', 'IR']

        if save == True:
            os.chdir('../latexTable')
            with open('IR.txt', 'w') as f:
                f.write(tabulate(t, headers, tablefmt='latex'))
        else:
            print(tabulate(t, headers))

if __name__ == '__main__':
    # Simple usage
    obj = IR(['balance','appendicitis', 'banana', 'bupa', 'glass'])
    obj.calculate()
    obj.tab(True)