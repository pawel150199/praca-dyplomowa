import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind, ranksums
from Evaluator import preprocs, datasets, metrics

""" 
Pair tests
"""

# Vars
alpha=.05
m_fmt="%.3f"
std_fmt=None
nc="---"
db_fmt="%s"
tablefmt="plain"
n_preprocs = len(preprocs)

if __name__=="__main__":
    # Generate table
    t = []

    for m_id, m_name in enumerate(metrics):
        # Load scores
        scores = np.load("../results/GNB_Undersampling.npy")
        scores = scores[:,:,:,m_id]
        mean_scores = np.mean(scores, axis=2)
        stds = np.std(scores, axis=2)
        # Calculate mean and std score
        t.append([db_fmt % m_name])

        for db_idx, db_name in enumerate(datasets):
            # Mean score
            t.append([''] + [db_fmt % db_name] + [m_fmt % v for v in mean_scores[db_idx, :]])
            # Std score
            if std_fmt:
                t.append([''] + [std_fmt % v for v in stds[db_idx, :]])
            # t-student
            T, p = np.array(
                [[ttest_ind(scores[db_idx, i, :],
                    scores[db_idx, j, :])
                for i in range(len(preprocs))]
                for j in range(len(preprocs))]
            ).swapaxes(0, 2)
            _ = np.where((p < alpha) * (T > 0))
            conclusions = [list(1 + _[1][_[0] == i])
                        for i in range(n_preprocs)]
    
            t.append([''] + [''] + [", ".join(["%i" % i for i in c])
                            if len(c) > 0 and len(c) < len(preprocs)-1 else ("all" if len(c) == len(preprocs)-1 else nc)
                            for c in conclusions])

    # Wilcoxon - global ranks
    ranks = np.load("../results/GNB_RanksUndersampling.npy")

    preprocs = list(preprocs.keys())
    mean_ranks = np.mean(ranks, axis=1)
    alpha = .05
    r = []

    for m_id, metric in enumerate(metrics):
        metric_ranks = ranks[m_id,:,:]
        length = len(preprocs)
        s = np.zeros((len(preprocs), len(preprocs)))
        p = np.zeros((len(preprocs), len(preprocs)))

        for i in range(length):
            for j in range(length):
                s[i, j], p[i, j] = ranksums(metric_ranks.T[i], metric_ranks.T[j])

        _ = np.where((p < alpha) * (s > 0))
        conclusions = [list(1 + _[1][_[0] == i]) for i in range(length)]
        r.append(["%s" % metric] + ["%.3f" % v for v in mean_ranks[m_id]])
        r.append([''] + [", ".join(["%i" % i for i in c]) if len(c) > 0 and len(c) < len(preprocs)-1 else ("all" if len(c) == len(preprocs)-1 else nc) for c in conclusions])
        
    ################################# T-Student ######################################
    # Show outputs 
    headers = ['metrics','datasets']
    for i in preprocs:
        headers.append(i)
    print(tabulate(t, headers))

    # Save outputs
    with open('../latexTable/GNB_TstudentUndersampling.txt', 'w') as f:
        f.write(tabulate(t, headers, tablefmt='latex'))
        
    ################################### Ranks #########################################

    # Show outputs
    print(tabulate(r, headers=(preprocs), tablefmt='plain'))

    # Save outputs
    with open('../latexTable/GNB_RanksUndersampling.txt', 'w') as f:
        f.write(tabulate(r, headers=(preprocs), tablefmt='latex'))
