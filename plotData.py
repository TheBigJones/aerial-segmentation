import matplotlib.pyplot as plt
import numpy as np
from data.config import LABELS, LABELMAP

labels_inc = ["IGNORE"] + LABELS

def norm(occurrences):
    return occurrences / np.sum(occurrences[1:])


l, occurrences = np.loadtxt("class_occurrences_dataset-medium.dat", usecols=(0,2,), unpack=True)

frequency = norm(occurrences)

l = l[1:]
frequency = frequency[1:]


l_s = [int(e) for _, e in sorted(zip(frequency, l))]
frequencies_s = sorted(frequency)

fig, axes = plt.subplots()
axes.set_yscale("log")
axes.set_ylim(1E-3, 1E0)
axes.grid(axis="y", zorder=5)

axes.bar([i for i in range(len(frequencies_s))], frequencies_s)


axes.set_ylabel("Relative frequency")
axes.set_axisbelow(True)

axes.set_xticks([i for i in range(len(frequencies_s))])

def to_normal(s):
    return s[0] + s[1:].lower()

axes.set_xticklabels([to_normal(labels_inc[l_s[i]]) for i in range(len(frequencies_s))], rotation=0)


plt.tight_layout()

plt.savefig("relative_frequency.pdf")


fig, axes = plt.subplots(2,3)

for i, l in enumerate(LABELS):
    bins, elev = np.loadtxt(f"elev_dist_dataset-medium_{l}.dat", usecols=(0,1,), unpack=True)
    elev = elev/np.sum(elev)
    axes[i // 3][i % 3].plot(bins, elev)
    axes[i // 3][i % 3].set_title(to_normal(l))
    
    axes[i // 3][i % 3].set_yscale("log")
    axes[i // 3][i % 3].set_ylim(1E-9, 1E0)
    axes[i // 3][i % 3].set_xlim(min(bins), max(bins))
    
    axes[i // 3][i % 3].set_xticks([i*64 for i in range(4)] + [255])
    
    axes[i // 3][i % 3].grid(axis="y", zorder=5)
    
    
axes[0][0].set_ylabel("Relative frequency")
axes[1][0].set_ylabel("Relative frequency")
axes[1][0].set_xlabel("Elevation")
axes[1][1].set_xlabel("Elevation")
axes[1][2].set_xlabel("Elevation")

plt.tight_layout()

plt.savefig("elevation_distribution.pdf")

plt.show()


