import numpy as np
from tabulate import tabulate

class IR:
    """
    Klasa słuzy do wygenerowania tabeli ze statystykami zbiorów danych
    Im większy współczynnik IR tym większa rónica pomiędzy
    zbilansowaniem klas w przestrzeni problemu

    Usage:
        obj = IR('datasetname')
        obj.calculate()
        obj.tab(true)
    """
    def __init__(self, datasets):
        self.datasets = datasets
    
    def calculate(self):
        self.scores = []
        self.y = []
        self.dataset_name = []
        for data_id, dataset in enumerate(self.datasets):
            dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=',')
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
        # save -> parametr mówiący czy zapisac wyniki w postaci tablicy w formacie tex
        t = []
        for data_id, data_name in enumerate(self.datasets):
            t.append(['%s' % data_name] + ['%.3f' % v for v in self.scores[data_id]])
        headers = ['datasets', 'N_min', 'N_maj', 'IR']

        if save == True:
            with open('latexTable/IR.txt', 'w') as f:
                f.write(tabulate(t, headers, tablefmt='latex'))
        else:
            print(tabulate(t, headers))
