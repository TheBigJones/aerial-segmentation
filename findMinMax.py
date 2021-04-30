import os
from PIL import Image
import numpy as np
import sys

def load_elevation(fname):
    elevation = Image.open(fname)
    return elevation


#img_folder = "elevations"
img_folder = "eleva-chips"


folder = sys.argv[1]


minimum = 1E10
maximum = -1E10
num_bins = 1000

path = os.path.join(os.getcwd(), folder, img_folder)

files = os.listdir(path)

for f in files:
    elevation = load_elevation(os.path.join(path, f))
    if np.max(elevation) > maximum:
        maximum = np.max(elevation)
    if np.min(elevation) < minimum:
        minimum = np.min(elevation)

print(f"Maximum elevation found: {maximum}")
print(f"Minimum elevation found: {minimum}")


bins = np.linspace(minimum, maximum, num_bins)
histos = np.zeros((len(files), num_bins-1))

for i, f in enumerate(files):
    elevation = load_elevation(os.path.join(path, f))
    histos[i] = np.histogram(elevation, bins=bins)[0]

np.savetxt(f"elev_dist_{folder}.dat", np.column_stack((bins[:-1]+(bins[1]-bins[0])/2, np.mean(histos, axis=0))), header="#maximum = {maximum}\n#minimum = {minimum}", comments="")
# test comment


