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



labels_inc = ["IGNORE"] + LABELS

img_folder = "elevations"
#img_folder = "eleva-chips"
#mask_folder = "label-chips"
mask_folder = "labels"
dataset = "dataset-medium"

folder = os.path.join(os.path.dirname(__file__), dataset)

minimum = 1E10
maximum = -1E10
num_bins = 257

eleva_path = os.path.join(folder, img_folder)
mask_path = os.path.join(folder, mask_folder)

files = os.listdir(eleva_path)

occurrences = {c: 0 for c in labels_inc}


bins = np.linspace(0, 256, 257)
histos = {c: np.zeros((len(files), num_bins-1)) for c in labels_inc}


print(LABELMAP)

for count, f in enumerate(files):
    print(f"{count}/{len(files)}")

    elevation = load_elevation(os.path.join(eleva_path, f))
    mask = load_mask(os.path.join(mask_path, f.replace("elev.tif", "label.png")))

    for i, label in enumerate(labels_inc):
    
        if "chips" in img_folder:
            masking = (mask[:,:,0]==i)
        else:
            color = LABELMAP[i]
            masking = np.where( (mask[:, :, 0] == color[2]) & (mask[:, :, 1] == color[1]) & (mask[:, :, 2] == color[0]) )
            
        occurrences[label] += np.sum(masking)
        masked_elev = elevation[masking]
    
        histos[label][count] = np.histogram(masked_elev, bins=bins)[0]

for label in LABELS:
    np.savetxt(f"elev_dist_{dataset}_{label}.dat", np.column_stack((bins[:-1], np.mean(histos[label], axis=0))), comments="")



#np.savetxt(f"class_occurrences_{dataset}.dat", np.column_stack(([i for i in range(len(labels_inc))], [occurrences[l] for l in labels_inc])), comments="")

np.savetxt(f"class_occurrences_{dataset}.dat", [(i, labels_inc[i], occurrences[labels_inc[i]]) for i in range(len(labels_inc))], fmt="%s", comments="")


