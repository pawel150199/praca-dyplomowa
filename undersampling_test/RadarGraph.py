import matplotlib.pyplot as plt
from math import pi
import numpy as np

preprocs= ['RUS', 'CC', 'NM', 'OSS', 'CNN', 'MCC']
metrics = ["BAC", "G-mean", "F1", "Precision", "Recall", "Specificity"]

ranks = np.load("../results/GNB_RanksUndersampling.npy")
mean_ranks = np.mean(ranks, axis=1)
N = len(preprocs)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

plt.xticks(angles[:-1], metrics)
plt.yticks([0,1, 2, 3, 4, 5, 6], ["0", "1", "2", "3", "4", "5", "6"], color="grey", size=7)
plt.ylim(0,6)
plt.title("GNB")

for method_id, method in enumerate(preprocs):
    values = mean_ranks[:, method_id].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1.3, linestyle='solid', label=method)

# Legend
plt.legend(bbox_to_anchor=(1.15, -0.06), ncol=8, fontsize=9)

# Save image
plt.savefig("../images/GNB_UndersamplingRadar", dpi=1000)
