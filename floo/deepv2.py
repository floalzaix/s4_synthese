#
#   Imports
#

import time

import torch
import pandas # type: ignore
import numpy as np

from torch import nn
from typing import Any, Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
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

# Temporal subsampling  3000 frames @ stride 10 => 300 depth
TEMPORAL_STRIDE = 10

FEATURE_DIM = 256
DROPOUT = 0.3

EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LOG_BATCH_INTERVAL = 10
LOG_DIR = "./floo/runs/deepv2-1"
MODEL_SAVE_PATH = "./floo/models/deepv2/deepv2.pt"

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
        self.data_path = data_path
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
        heatmap = np.load(
            self.data_path + patient_id + "_hm.npz"
        )["heatmaps"] # type: ignore
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


def collate_gait_batch(
    batch: List[Dict[str, Any]],
) -> Dict[str, Tensor]:
    """
        Pads heatmap sequences to the longest sample in the batch.

        Params:
            - batch: List of samples from GaitDataset.

        Returns:
            Dict with stacked heatmaps and float labels.
    """
    heatmaps = [item["heatmap"] for item in batch]
    max_t = max(h.shape[0] for h in heatmaps)
    c, h, w = heatmaps[0].shape[1:]

    padded = torch.zeros(len(batch), max_t, c, h, w)
    for i, heatmap in enumerate(heatmaps):
        padded[i, : heatmap.shape[0]] = heatmap

    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.float32,
    )

    return {"heatmap": padded, "label": labels}


#
#   Model definition
#

class ResidualBlock3D(nn.Module):
    """
        Residual block with 3D convolutions for spatiotemporal
        feature extraction.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(channels)

    #
    #   Overrides
    #

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = torch.relu(out)
        return out


class VolumeEncoder3D(nn.Module):
    """
        3D CNN backbone that maps a gait heatmap volume to a
        fixed-size feature vector.

        Params:
            - feature_dim: Output feature dimension.

        Inputs:
            - x: Tensor of shape (N, 1, T, H, W)

        Outputs:
            - features: Tensor of shape (N, feature_dim)
    """

    def __init__(self, feature_dim: int):
        super().__init__()

        # Stem — spatial 224 -> 56, temporal depth preserved
        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

        self.layer1 = nn.Sequential(
            ResidualBlock3D(64),
            ResidualBlock3D(64),
        )

        # Spatiotemporal down — T/2, H/2, W/2
        self.down1 = nn.Sequential(
            nn.Conv3d(
                64,
                128,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock3D(128),
            ResidualBlock3D(128),
        )

        # Spatial down only — H/2, W/2
        self.down2 = nn.Sequential(
            nn.Conv3d(
                128,
                256,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock3D(256),
            ResidualBlock3D(256),
        )

        # Temporal down — T/2
        self.down3 = nn.Sequential(
            nn.Conv3d(
                256,
                256,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Linear(256, feature_dim)

    #
    #   Overrides
    #

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.down3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class DeepV2(nn.Module):
    """
        End-to-end 3D CNN for binary gait classification.
        Spatiotemporal patterns are learned directly from the
        heatmap volume without a per-frame 2D encoder or
        temporal transformer.

        Inputs:
            - heatmap: Tensor of shape (batch_size, T, C, H, W)
            T = number of time steps (30 seconds at 100 FPS)
            C = number of channels (3 for the RGB channels but
            only one is used)
            H = height of the heatmap (224)
            W = width of the heatmap (224)

        Outputs:
            - classification: Tensor of shape (batch_size,)
    """

    def __init__(
        self,
        *,
        feature_dim: int = FEATURE_DIM,
        temporal_stride: int = TEMPORAL_STRIDE,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.temporal_stride = temporal_stride
        self.volume_encoder = VolumeEncoder3D(feature_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

        print("DeepV2 initialized !")

    #
    #   Overrides
    #

    def forward(self, heatmap: Tensor) -> Tensor:
        """
            Forward pass of the model.

            Params:
                - heatmap: Tensor of shape (B, T, C, H, W)

            Returns:
                Logits of shape (B,) for binary classification.
        """
        
        # Keeping only the pressure intensity channel
        x = heatmap[:, :, 0:1, :, :]

        # Subsampling the temporal axis to reduce memory usage
        x = x[:, :: self.temporal_stride, :, :, :]

        # (B, T, 1, H, W) -> (B, 1, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)

        features = self.volume_encoder(x)
        logits = self.classifier(features).squeeze(-1)
        return logits

    #
    #   Methods
    #

    def infer(self, heatmap: Tensor) -> Tensor:
        """
            Do not use for training.


            Returns:
                Class probabilities after sigmoid.
        """
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(heatmap))

    def train_one_epoch(
        self,
        train_loader: DataLoader[Dict[str, Tensor]],
        train_set_length: int,
        optimizer: Any,
        criterion: Any,
        epoch_num: int,
        writer: SummaryWriter,
        tag: str,
    ) -> Tuple[List[float], int, int, float]:
        """
            Training loop for one epoch.


            Returns:
                - Loss values per batch
                - Number of correct predictions
                - Total number of samples
                - Elapsed time in seconds
        """
        time_stamp = time.time()
        self.train()

        loss_tensor: List[float] = []
        corrects = 0
        total = 0
        n = 0

        for batch in train_loader:
            heatmaps = batch["heatmap"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = self(heatmaps)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = heatmaps.size(0)
            loss_tensor.append(loss.item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            corrects += (preds == labels).sum().item()
            total += batch_size

            if n % LOG_BATCH_INTERVAL == 0:
                print(
                    f" => [{tag}] Epoch {epoch_num} "
                    f"Total: {total} / {train_set_length}"
                )

            n += 1

        writer.add_scalar(
            f"Loss/train_{tag}", np.mean(loss_tensor), epoch_num
        )
        writer.add_scalar(
            f"Accuracy/train_{tag}", corrects / total, epoch_num
        )
        writer.flush()

        return loss_tensor, corrects, total, time.time() - time_stamp

    def evaluate(
        self,
        val_loader: DataLoader[Dict[str, Tensor]],
        criterion: Any,
    ) -> Tuple[List[float], int, int]:
        """
            Validation loop.


            Returns:
                - Loss values per batch
                - Number of correct predictions
                - Total number of samples
        """
        self.eval()

        loss_tensor: List[float] = []
        corrects = 0
        total = 0

        for batch in val_loader:
            heatmaps = batch["heatmap"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with torch.no_grad():
                logits = self(heatmaps)
                loss = criterion(logits, labels)

                batch_size = heatmaps.size(0)
                loss_tensor.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).float()
                corrects += (preds == labels).sum().item()
                total += batch_size

        return loss_tensor, corrects, total

    def save(self, path: str = MODEL_SAVE_PATH) -> None:
        """
            Saves model weights to disk.

            Params:
                - path: Destination file path.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path} !")


#
#   Functions
#

def run_training(
    model: DeepV2,
    train_loader: DataLoader[Dict[str, Tensor]],
    val_loader: DataLoader[Dict[str, Tensor]],
    *,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    log_dir: str = LOG_DIR,
) -> None:
    """
        Full training loop with TensorBoard logging.

        Params:
            - model: DeepV2 instance already on DEVICE.
            - train_loader: Training DataLoader.
            - val_loader: Validation DataLoader.
            - epochs: Number of training epochs.
            - lr: Adam learning rate.
            - weight_decay: Adam weight decay.
            - log_dir: TensorBoard log directory.
    """
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(log_dir=log_dir)

    train_len = len(train_loader.dataset) # type: ignore

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch} / {epochs} ---")

        train_loss, train_correct, train_total, elapsed = (
            model.train_one_epoch(
                train_loader,
                train_len,
                optimizer,
                criterion,
                epoch,
                writer,
                "deepv2",
            )
        )
        val_loss, val_correct, val_total = model.evaluate(
            val_loader,
            criterion,
        )

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        writer.add_scalar(
            "Loss/val_deepv2", np.mean(val_loss), epoch
        )
        writer.add_scalar(
            "Accuracy/val_deepv2", val_acc, epoch
        )
        writer.flush()

        print(
            f" Train loss: {np.mean(train_loss):.4f}, "
            f"acc: {train_acc:.4f}, "
            f"time: {elapsed:.2f}s"
        )
        print(
            f" Val   loss: {np.mean(val_loss):.4f}, "
            f"acc: {val_acc:.4f}"
        )

    model.save()
    writer.close()


#
#   Main
#

if __name__ == "__main__":

    #
    #   Handling dataloading and separating train and validation
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
        trainset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=True,
        collate_fn=collate_gait_batch,
    )
    validationloader = DataLoader(
        validationset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=False,
        collate_fn=collate_gait_batch,
    )

    #
    #   Training
    #

    model = DeepV2().to(DEVICE)
    run_training(model, trainloader, validationloader)
