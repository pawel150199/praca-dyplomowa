import matplotlib.pyplot as plt
from math import pi
import numpy as np

clfs = ["Bagging", "RSM", "RSP", "OB", "ORSM", "ORSP", "UB", "URSM", "URSP"]
metrics = ["BAC", "G-mean", "F1-score", "Precision", "Recall", "Specificity"]

ranks = np.load("../results/ENSEMBLES/Ranks_KNN.npy")
mean_ranks = np.mean(ranks, axis=1)

N = mean_ranks.shape[0]
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.spines["polar"].set_visible(False)

plt.xticks(angles[:-1], metrics)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], color="grey", size=9)
plt.ylim(0,9)
plt.title("kNN", loc="center")

# Line style
ls = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
lw = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

for method_id, method in enumerate(clfs):
    values = mean_ranks[:, method_id].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=lw[method_id], linestyle=ls[method_id], label=method)

# Legend
plt.legend(bbox_to_anchor=(1.2, -0.08), ncol=5, fontsize=9)
plt.tight_layout()

# Save image
plt.savefig("../images/KNNEnsmblesRadar", dpi=600, bbox_inches='tight')


