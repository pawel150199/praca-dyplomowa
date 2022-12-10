import matplotlib.pyplot as plt
from math import pi
import numpy as np

preprocs= ['RUS', 'CC', 'NM', 'OSS', 'CNN', 'MCC']
metrics = ["BAC", "G-mean", "F1-score", "Precision", "Recall", "Specificity"]

ranks = np.load("../results/UNDERSAMPLING/SVC_RanksUndersampling.npy")
mean_ranks = np.mean(ranks, axis=1)
N = len(preprocs)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.spines["polar"].set_visible(False)

plt.xticks(angles[:-1], metrics, fontsize=13)
plt.yticks([0,1, 2, 3, 4, 5, 6], ["0", "1", "2", "3", "4", "5", "6"], color="grey", size=10)
plt.ylim(0,6)
plt.title("SVC", fontsize=15)

ls = ["-", "--", "-.", ":", "--", "-"]
lw = [1, 1, 1, 1, 1, 1.5]

for method_id, method in enumerate(preprocs):
    values = mean_ranks[:, method_id].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=lw[method_id], linestyle=ls[method_id], label=method)

# Legend
plt.legend(bbox_to_anchor=(1, -0.06), ncol=3, fontsize=13)

# Save image
plt.savefig("../images/SVC_UndersamplingRadar", dpi=600, bbox_inches='tight')
