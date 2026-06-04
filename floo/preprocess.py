#
#   Imports
#

import os
from typing import Any
import pandas as p # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.interpolate import Rbf # type: ignore

#
#   Constants
#

INPUT_PATH = "datasets/gait-in-parkinsons-disease-1.0.0/"
OUTPUT_PATH = "datasets/gait-in-parkinsons-disease-1.0.0/preprocessed/"

# TODO: Determine which size is the most appropriate
# Often used sizes are 224x224 for pretrained models
HM_W = 224
HM_H = 224

# TODO: See if we keep only the patient with _01 at the end or _02 for now no just keeping the _01

# The padding around the footprints and the radius of the points
# TODO : Determine the best padding and radius
HEATMAP_PADDING = 12 * 224 / 64 # Because it the 11 is from the original 64x64 heatmap
POINT_RADIUS = 8.5 * 224 / 64 # Because it the 9 is from the original 64x64 heatmap

SENSOR_XY = np.array([
    [-500, -800],
    [-700, -400],
    [-300, -400],
    [-700, 0],
    [-300, 0],
    [-700, 400],
    [-300, 400],
    [-500, 800],
    [500, -800],
    [700, -400],
    [300, -400],
    [700, 0],
    [300, 0],
    [700, 400],
    [300, 400],
    [500, 800],
])

FPS = 100
TIME_TO_KEEP = 5 # seconds
LOG_INTERVAL = 9.99 # seconds

#
#   Cleaning the metadata
#

# Loading the demographics
demo = p.read_csv(INPUT_PATH + "demographics.csv", sep=";")

demo.info()

# Dropping the useless columns
# Removing speed_10 cause not enough values that are not NaN
# Removing the scores cause it probably introduces a bias
demo = demo.drop(columns=[
    "Subjnum", "Study", "HoehnYahr", "UPDRS", "UPDRSM", "TUAG", "Speed_10"
]) # type: ignore

# Dropping the rows with NaN
demo = demo.dropna()

# Dropping the row Juc010 cause it does not exist in data
demo = demo[demo["ID"] != "Juc010"]

# Remapping the values of the columns to numeric ones
demo["Group"] = demo["Group"].map({"PD": 1, "CO": 0})
demo["Gender"] = demo["Gender"].map({"male": 1, "female": 0})
demo["Height"] = demo["Height"].str.replace(",", ".").astype(float)
demo["Height"] = demo["Height"].apply(lambda x: x * 100 if x < 100 else x)
demo["Speed_01 (m/sec)"] = demo["Speed_01 (m/sec)"].str.replace(",", ".").astype(float)

demo.info()
demo.to_csv(OUTPUT_PATH + "demographics.csv", index=False, sep=";")

# Displaying the corr matrix
corr_matrix = demo.drop(columns=["ID"]).corr()
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="none")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Matrix")
plt.show()

#
#   Preparing the heatmaps
#

def normalize(T: np.ndarray, axis: int = 0) -> np.ndarray:
    """
        Normalise the tensor T along the axis axis

        Params:
            - T (np.ndarray): The tensor to normalise
            - axis (int): The axis along which to normalise

        Returns:
            - The normalised tensor
    """
    return (
        T - T.min(axis=axis, keepdims=True)
    ) / (
        T.max(axis=axis, keepdims=True) - T.min(axis=axis, keepdims=True)
    )

# Preparing the coordinates grid
Xg, Yg = np.meshgrid(np.arange(HM_W), np.arange(HM_H))
x, y = SENSOR_XY[:, 0], SENSOR_XY[:, 1]
x = normalize(x) * (HM_W - 1 - 2 * HEATMAP_PADDING) + HEATMAP_PADDING
y = normalize(y) * (HM_H - 1 - 2 * HEATMAP_PADDING) + HEATMAP_PADDING

blanck_column = np.zeros((HM_H, HM_W))

# Preparing the heatmaps for each patient
for i, patient in enumerate(demo["ID"]):
    print(f"Processing patient {i + 1} / {len(demo['ID'])}")
    
    # Loading the patient grf file
    patient_path = INPUT_PATH + patient + "_01.txt"
    if os.path.isfile(patient_path):
        with open(patient_path, "r") as file:

            # Creating a heatmap for each line
            heatmaps: Any = []
            time_stamp = time.time()
            n = 0 # Number of logs printed
            i = 0 # Index of the heatmap
            weight = demo[demo["ID"] == patient]["Weight (kg)"].values[0]
            for line in file:

                # Keeping the heatmaps for the last TIME_TO_KEEP seconds                
                if len(heatmaps) >= TIME_TO_KEEP * FPS:
                    break

                splits = line.strip("\n").split("\t")
                if len(splits) != 19:
                    continue

                forces = np.array([float(x) for x in splits[1:17]], dtype=np.float32)

                # TODO: Normalise with weight of the patient
                # Normalizing the with the weight of the patient (N/kg)
                forces = forces / weight

                # Creating the heatmap ass an image with size (W, H)
                rbf = Rbf(
                    x, y,
                    forces,
                    function="gaussian",
                    epsilon=POINT_RADIUS,
                )
                heatmap = rbf(Xg, Yg)

                # Giving the shape (C, H, W)
                # C = 1 because we have only one channel, intensity
                # and reversing H because of the different referentials
                # and then duplicating the canals for compatibility purposes to
                # standard pre trained models => C = 3
                heatmap = heatmap[::-1, :] # Reversing the vertical axis
                heatmap = np.stack([heatmap, blanck_column, blanck_column], axis=0)
                heatmap = (heatmap).astype(np.float16) # Converting to uint8 0-255
                heatmaps.append(heatmap)
                
                i += 1

                if time.time() - time_stamp >= LOG_INTERVAL * n:
                    n += 1
                    print(
                        f" -> Processed {i} heatmaps in {time.time() - time_stamp:.2f} "
                        "seconds !"
                    )

            heatmaps = np.array(heatmaps) # Shape : (T, 3, H, W)
            
            # Saving the heatmaps
            print(
                "\n"
                "=================================================\n"
                f" => Saving heatmaps for patient {patient} in "
                f"{time.time() - time_stamp:.2f} seconds !"
            )
            time_stamp = time.time()
            np.savez_compressed(OUTPUT_PATH + patient + "_hm.npz", heatmaps=heatmaps)
            print(
                f" => Saved heatmaps for patient {patient} in "
                f"{time.time() - time_stamp:.2f} seconds !\n"
                "=================================================\n"
            )

    else:
        print(f"No file found for patient {patient} !")