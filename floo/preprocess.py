#
#   Imports
#

import pandas as p # type: ignore
import matplotlib.pyplot as plt

#
#   Constants
#

INPUT_PATH = "datasets/gait-in-parkinsons-disease-1.0.0/"
OUTPUT_PATH = "datasets/gait-in-parkinsons-disease-1.0.0/preprocessed/"

#
#   Main
#

# Loading the demographics
demo = p.read_csv(INPUT_PATH + "demographics.csv", sep=";")

demo.info()

# Dropping the useless columns
demo = demo.drop(columns=["ID", "Study", "Speed_10"]) # type: ignore

# Dropping the rows withNaN
demo = demo.dropna()

# Remapping the values of the columns to numeric ones
demo["Group"] = demo["Group"].map({"PD": 1, "CO": 0})
demo["Gender"] = demo["Gender"].map({"male": 1, "female": 0})
demo["Height"] = demo["Height"].str.replace(",", ".").astype(float)
demo["Height"] = demo["Height"].apply(lambda x: x * 100 if x < 100 else x)
demo["HoehnYahr"] = demo["HoehnYahr"].str.replace(",", ".").astype(float)
demo["TUAG"] = demo["TUAG"].str.replace(",", ".").astype(float)
demo["Speed_01 (m/sec)"] = demo["Speed_01 (m/sec)"].str.replace(",", ".").astype(float)

demo.info()

# Displaying the corr matrix
corr_matrix = demo.corr()
plt.figure(figsize=(7, 710))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="none")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Matrix")
plt.show()
