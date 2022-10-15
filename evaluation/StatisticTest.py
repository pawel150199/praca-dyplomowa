from distutils.log import error
import numpy as np
import os
from tabulate import tabulate
from scipy.stats import ttest_ind

""" 
Klasa generuje wyniki testów statystycznych, które zapisywane są w katalogu results
"""

class StatisticTest():
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def process(self, table_name, alpha=0.05, m_fmt="%3f", std_fmt=None, nc="---", db_fmt="%s", tablefmt="plain"):
        try:
            #DATA X FOLD X CLASSIFIER X METRIC 
            scores = self.evaluator.scores
            mean_scores = self.evaluator.mean
            std = self.evaluator.std
            metrics = list(self.evaluator.metrics.keys())
            clfs = list(self.evaluator.clfs.keys())
            datasets = self.evaluator.datasets
            n_clfs = len(clfs)

            #Generowanie tabel
            for m_id, m_name in enumerate(metrics):
                #
                t = []
                for db_idx, db_name in enumerate(datasets):
                    #Wiersz z wartoscia srednia
                    t.append([db_fmt % db_name] + [m_fmt % v for v in mean_scores[db_idx, :]])
                    #Jesli podamy std_fmt w zmiennych globalnych zostanie do tabeli dodany wiersz z odchyleniem standardowym
                    if std_fmt:
                        t.append( [std_fmt % v for v in std[db_idx, :]])
                    #Obliczenie wartosci T i P z testu T-studenta
                    T, p = np.array(
                        [[ttest_ind(scores[db_idx, i, :],
                            scores[db_idx, j, :])
                        for i in range(len(clfs))]
                        for j in range(len(clfs))]
                    ).swapaxes(0, 2)
                    _ = np.where((p < alpha) * (T > 0))
                    conclusions = [list(1 + _[1][_[0] == i])
                                for i in range(n_clfs)]
            
                    t.append([''] + [", ".join(["%i" % i for i in c])
                                    if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)
                                    for c in conclusions])

                #Prezentacja wyników
                print('\n\n\n', m_name, '\n')  
                headers = ['datasets']
                for i in clfs:
                    headers.append(i)
                print(tabulate(t, headers))

                #Zapisanie wyników w formacie .tex
                os.chdir('../latexTable')
                with open('%s_%s.txt' % (table_name, m_name), 'w') as f:
                    f.write(tabulate(t, headers, tablefmt='latex'))
        except ValueError:
            error('Operacja nie powiodła się!')
