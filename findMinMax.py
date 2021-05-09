import os
from PIL import Image
import numpy as np
import sys
from data.config import LABELS, LABELMAP

def load_elevation(fname):
    elevation = np.asarray(Image.open(fname))
    return elevation

def load_mask(fname):
    mask = np.asarray(Image.open(fname), dtype=np.uint16)
    return mask

#img_folder = "elevations"
img_folder = "eleva-chips"
mask_folder = "label-chips"
dataset = "dataset-sample"

folder = os.path.join(os.path.dirname(__file__), dataset)

minimum = 1E10
maximum = -1E10
num_bins = 1000

eleva_path = os.path.join(folder, img_folder)
mask_path = os.path.join(folder, mask_folder)

elev_files = os.listdir(eleva_path)
mask_files = os.listdir(mask_path)

maximum = dict(zip(LABELS, [0]*len(LABELS)))
minimum = dict(zip(LABELS, [0]*len(LABELS)))

for elev, mask in zip(elev_files, mask_files):
    elevation = load_elevation(os.path.join(eleva_path, elev))
    mask = load_mask(os.path.join(mask_path, mask))
    for i, label in enumerate(LABELS, 1):
        masking = mask[:,:,0]==i
        masked_elev = elevation[masking]
        if masked_elev.shape[0] == 0:
            continue
        if np.max(masked_elev) > maximum[label]:
            maximum[label] = np.max(masked_elev)
        if np.min(masked_elev) < minimum[label]:
            minimum[label]  = np.min(masked_elev)

print(f"Maximum elevation found: {maximum}")
print(f"Minimum elevation found: {minimum}")


bins = np.linspace(0, 255, num_bins)

histos = dict()
for label in LABELS:
    histos[label] = np.zeros((len(elev_files), num_bins-1))

for i, (elev, mask) in enumerate(zip(elev_files, mask_files)):
    elevation = load_elevation(os.path.join(eleva_path, elev))

    mask = load_mask(os.path.join(mask_path, mask))
    
    # LABELMAP key 0 are ignore -> start at 1
    for j, label in enumerate(LABELS, 1):
        masking = mask[:,:,0]==j
        histos[label][i] = np.histogram(elevation[masking], bins=bins)[0]

for label in LABELS:
    np.savetxt(f"elev_dist_{dataset}_{label}.dat", np.column_stack((bins[:-1]+(bins[1]-bins[0])/2, np.mean(histos[label], axis=0))), header=f"#maximum = {maximum[label]}\n#minimum = {minimum[label]}", comments="")