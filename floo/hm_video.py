#
#   Imports
#

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#
#   Constants
#

PATH = "datasets/gait-in-parkinsons-disease-1.0.0/preprocessed/GaPt05_hm.npz"
OUTPUT_PATH = "test.mp4"

FPS = 50

#
#   Main
#

hm = np.load(PATH)["heatmaps"]
print(hm.shape)
print("Heatmap matrix loaded loaded !")

hm = hm[:, 0, :, :] # Shape : (T, H, W)

w, h = hm.shape[2], hm.shape[1]

cmap = plt.get_cmap("hsv")

# Normalising the data to uint8 0-255
hm = hm.astype(np.float32)

hm -= hm.min()
hm /= hm.max()

fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter(OUTPUT_PATH, fourcc, FPS, (w, h))

for frame in hm:
    out.write((cmap(frame)[:, :, :3] * 255).astype(np.uint8))

out.release()