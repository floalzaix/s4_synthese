#
#   Imports
#

import torch
import pandas # type: ignore
import numpy as np

from torch import nn
from typing import Dict, Any
from torch.utils.data import DataLoader, Dataset, random_split
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#
#   Constants
#

# Should be the path of the preprocessed dataset
# including the demographics.csv file and the heatmaps
# to the format (T, C, W, H) in a file patient_hm.npz
DATA_PATH = "datasets/gait-in-parkinsons-disease-1.0.0/preprocessed/"

# Ratio of the dataset to be used for training
TRAIN_VALIDATION_RATIO = 0.8

BATCH_SIZE = 1

SEED = 42

#
#   Dataloading
#

class GaitDataset(Dataset[Dict[str, Any]]):
    """
        Contains the dataset for the training and evaluation of
        the model. It handles the loading the heatmaps and the
        metadata of the patients.

        Then DataLoader will handle the batching logic using
        different workers.
    """

    def __init__(self, data_path: str):
        self.metadata = pandas.read_csv(data_path + "demographics.csv", sep=";")

    #
    #   Overrides
    #
    
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        patient = self.metadata.iloc[index] # type: ignore

        # Getting the heatmaps and the label
        patient_id = patient["ID"] # type: ignore
        heatmap = np.load(DATA_PATH + patient_id + "_hm.npz")["heatmaps"] # type: ignore
        label = patient["Group"] # type: ignore

        # Processing the heatmaps
        heatmap = torch.from_numpy(heatmap).float() # type: ignore
        heatmap /= 65535.0 # uint16 to 0 1 float

        # Removing the group and id to avoid biaised analysis
        patient = patient.drop(columns=["ID", "Group"]) # type: ignore

        return {
            "heatmap": heatmap,
            "metadata": patient.to_dict(), # type: ignore
            "label": label
        }

#
#   Model definition
#

class DeepV1(nn.Module):
    """
        The model is a simple convolutional neural network
        that takes in a heatmap and then go through a time
        transformer block to extract time data and outputs
        a classification of the patient being 0 or 1 (PD or CO).

        Params:
            - batch_size: int

        Inputs:
            - heatmap: Tensor of shape (batch_size, T, C, H, W)
            T = number of time steps (30 seconds at 100 FPS)
            C = number of channels (3 for the RGB channels but
            only one is used)
            H = height of the heatmap (224)
            W = width of the heatmap (224)

        Outputs:
            - classification: Tensor of shape (batch_size, 1)
    """
    
    def __init__(self):
        super().__init__()

        self.

    #
    #   Overrides
    #
    
    def forward(self, heatmap: Tensor) -> Tensor:
        """
            Forward pass of the model.
        """
        return self.model(heatmap)

#
#   Functions
#

#
#   Training loop
#

#
#   Evaluation
#

#
#   Main
#

if __name__ == "__main__":

    #
    #   Handling dataloading and separating train and vaalidation
    #

    dataset = GaitDataset(DATA_PATH)

    train_size = int(TRAIN_VALIDATION_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    trainset, validationset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    trainloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=True
    )
    validationloader = DataLoader(
        validationset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=False
    )