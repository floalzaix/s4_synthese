#
#   Imports
#

import os
import queue
import sys
import threading
import time

import torch
import pandas # type: ignore
import numpy as np

from torch import nn
from typing import Any, Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

#
#   Split
#

SPLIT_MODE = "stratified"
HOLDOUT_STUDY = "Ju"
TRAIN_VALIDATION_RATIO = 0.8
SEED = 42

#
#   Training hyperparameters
#

BATCH_SIZE = 1
NUM_DATALOADER_WORKERS = 0
PIN_MEMORY = False
EPOCHS = 20
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.0001
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 5
BEST_MIN_DELTA = 1e-3
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 2
ENABLE_LIVE_EPOCH_INPUT = True

#
#   Model
#

# Must match preprocess.py (TIME_TO_KEEP * FPS)
GAIT_FPS = 100
GAIT_DURATION_SEC = 5
MAX_RAW_FRAMES = GAIT_FPS * GAIT_DURATION_SEC

TEMPORAL_STRIDE = 10

# Frames per CNN forward chunk (limits peak VRAM on long sequences)
FRAME_CNN_CHUNK_SIZE = 16
EMBED_DIM = 64
NUM_TRANSFORMER_LAYERS = 1
NUM_ATTENTION_HEADS = 4
TRANSFORMER_FF_DIM = 128
NUM_GROUPS = 8
DROPOUT = 0.2


def seq_len_after_stride(
    raw_frames: int,
    stride: int,
) -> int:
    """
        Sequence length after heatmap[::stride] (ceil division).
    """
    if stride < 1:
        raise ValueError("TEMPORAL_STRIDE must be >= 1")
    return (raw_frames + stride - 1) // stride


# Positional encoding size — tied to stride so T never exceeds max_len
MAX_SEQ_LEN = seq_len_after_stride(MAX_RAW_FRAMES, TEMPORAL_STRIDE)

#
#   Metadata (numeric columns from preprocess.py)
#

METADATA_COLS = [
    "Gender",
    "Age",
    "Height",
    "Weight (kg)",
    "Speed_01 (m/sec)",
]
N_METADATA = len(METADATA_COLS)
METADATA_EMBED_DIM = 16

#
#   Logging and checkpoints
#

LOG_BATCH_INTERVAL = 10
LOG_DIR = "./floo/runs/deepv1-2"
MODEL_SAVE_PATH = "./floo/models/deepv1/deepv1.pt"
BEST_MODEL_PATH = "./floo/models/deepv1/deepv1_best.pt"

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
        self.meta_mean: Tensor | None = None
        self.meta_std: Tensor | None = None

    #
    #   Metadata
    #

    def set_metadata_normalization(
        self,
        mean: Tensor,
        std: Tensor,
    ) -> None:
        """
            Z-score stats fitted on the training split only.
        """
        self.meta_mean = mean
        self.meta_std = std

    def _metadata_tensor(self, patient: Any) -> Tensor:
        """
            Builds a normalized feature vector from demographics.
        """
        vals = [float(patient[col]) for col in METADATA_COLS] # type: ignore
        features = torch.tensor(vals, dtype=torch.float32)

        # Raw features until set_metadata_normalization is called
        if self.meta_mean is None or self.meta_std is None:
            return features

        return (features - self.meta_mean) / self.meta_std

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

        # Demographics vector (z-scored when stats are set)
        meta = self._metadata_tensor(patient)

        # Processing the heatmaps
        heatmap = torch.from_numpy(heatmap).float() # type: ignore
        heatmap /= 65535.0 # uint16 to 0 1 float

        # Align with preprocess window (30 s @ 100 Hz)
        heatmap = heatmap[:MAX_RAW_FRAMES]

        # Pressure channel only + temporal stride (saves GPU RAM in the loader)
        heatmap = heatmap[::TEMPORAL_STRIDE, 0:1, :, :]

        return {
            "heatmap": heatmap,
            "metadata": meta,
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
            Dict with heatmaps, metadata vectors and labels.
    """
    heatmaps = [item["heatmap"] for item in batch]
    max_t = max(h.shape[0] for h in heatmaps)
    c, h, w = heatmaps[0].shape[1:]

    # Pad time dimension to the longest sequence in the batch
    padded = torch.zeros(len(batch), max_t, c, h, w)
    for i, heatmap in enumerate(heatmaps):
        padded[i, : heatmap.shape[0]] = heatmap

    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.float32,
    )

    # Stack per-patient demographic vectors
    metadata = torch.stack([item["metadata"] for item in batch])

    return {
        "heatmap": padded,
        "metadata": metadata,
        "label": labels,
    }


#
#   Train / validation split
#

def study_from_patient_id(patient_id: str) -> str:
    """
        Infers the study site from the patient ID prefix.
    """
    if patient_id.startswith("Ga"):
        return "Ga"
    if patient_id.startswith("Ju"):
        return "Ju"
    if patient_id.startswith("Si"):
        return "Si"
    return patient_id[:2]


def stratified_train_val_indices(
    labels: List[int],
    train_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
        Stratified train / validation index split per class.
    """
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []

    label_arr = np.array(labels)

    # Split each class separately then merge indices
    for label in (0, 1):
        cls_idx = np.flatnonzero(label_arr == label)
        rng.shuffle(cls_idx)
        n_train = int(train_ratio * len(cls_idx))
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train:].tolist())

    return train_idx, val_idx


def study_holdout_indices(
    studies: List[str],
    holdout: str,
) -> Tuple[List[int], List[int]]:
    """
        Train on all studies except holdout; validate on holdout only.
    """
    train_idx = [i for i, s in enumerate(studies) if s != holdout]
    val_idx = [i for i, s in enumerate(studies) if s == holdout]
    return train_idx, val_idx


def build_train_val_subsets(
    dataset: GaitDataset,
) -> Tuple[Subset, Subset]:
    """
        Builds train and validation subsets from SPLIT_MODE.
    """
    labels = dataset.metadata["Group"].astype(int).tolist()
    patient_ids = dataset.metadata["ID"].astype(str).tolist()
    studies = [study_from_patient_id(pid) for pid in patient_ids]

    # SPLIT_MODE: stratified (PD/CO) or study (cohort hold-out)
    if SPLIT_MODE == "study":
        train_idx, val_idx = study_holdout_indices(studies, HOLDOUT_STUDY)
        print(
            f"Study hold-out split: train={len(train_idx)}, "
            f"val={len(val_idx)} (holdout={HOLDOUT_STUDY})"
        )
    else:
        train_idx, val_idx = stratified_train_val_indices(
            labels,
            TRAIN_VALIDATION_RATIO,
            SEED,
        )
        print(
            f"Stratified split: train={len(train_idx)}, "
            f"val={len(val_idx)}"
        )

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


#
#   Metadata normalization
#

def fit_metadata_stats(
    dataset: GaitDataset,
    train_indices: List[int],
) -> Tuple[Tensor, Tensor]:
    """
        Mean and std of metadata columns on the training split.
    """
    rows = dataset.metadata.iloc[train_indices][METADATA_COLS].astype(float)
    mean = torch.tensor(rows.mean().to_numpy(), dtype=torch.float32)
    std = torch.tensor(rows.std().to_numpy(), dtype=torch.float32)
    # Avoid division by zero on constant columns
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def compute_pos_weight(labels: List[int]) -> Tensor:
    """
        Weight for the positive class (PD) in BCEWithLogitsLoss.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor([1.0], device=DEVICE)
    return torch.tensor([n_neg / n_pos], device=DEVICE)


#
#   Model definition
#

class ResidualBlock(nn.Module):
    """
        Residual block from the ResNet paper.
    """

    def __init__(self, channels: int, num_groups: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups, channels)

    #
    #   Overrides
    #

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + identity
        out = torch.relu(out)
        return out


class FrameEncoder(nn.Module):
    """
        CNN backbone that maps one heatmap frame to a fixed-size
        embedding vector.
    """

    def __init__(self, embed_dim: int, num_groups: int):
        super().__init__()

        # Stem — 224 -> 56
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = ResidualBlock(32, num_groups)

        # 56 -> 28
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = ResidualBlock(64, num_groups)

        # 28 -> 14
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = ResidualBlock(128, num_groups)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(128, embed_dim)

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
        x = self.pool(x).flatten(1)
        return self.head(x)


class PositionalEncoding(nn.Module):
    """
        Sinusoidal positional embeddings for the temporal transformer.

        Params:
            - embed_dim: Feature dimension.
            - max_len: Maximum sequence length after subsampling.
    """

    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()

        # Fixed sinusoidal table (no learned temporal encoding)
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-np.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pos_embed_table", pe.unsqueeze(0), persistent=False)

    #
    #   Overrides
    #

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        max_len = self.pos_embed_table.size(1)
        if seq_len > max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds positional "
                f"encoding max_len {max_len}. Increase MAX_SEQ_LEN "
                f"(e.g. lower TEMPORAL_STRIDE)."
            )
        return x + self.pos_embed_table[:, :seq_len, :]


class DeepV1(nn.Module):
    """
        The model is a simple convolutional neural network
        that takes in a heatmap and then go through a time
        transformer block to extract time data and outputs
        a classification of the patient being 0 or 1 (PD or CO).

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
        embed_dim: int = EMBED_DIM,
        temporal_stride: int = TEMPORAL_STRIDE,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        num_heads: int = NUM_ATTENTION_HEADS,
        ff_dim: int = TRANSFORMER_FF_DIM,
        num_groups: int = NUM_GROUPS,
        n_metadata: int = N_METADATA,
        metadata_embed_dim: int = METADATA_EMBED_DIM,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()

        self.temporal_stride = temporal_stride

        #
        #   Per-frame CNN and temporal transformer
        #

        self.frame_encoder = FrameEncoder(embed_dim, num_groups)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        #
        #   Demographics branch
        #

        self.metadata_encoder = nn.Sequential(
            nn.Linear(n_metadata, metadata_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        #
        #   Fused classification head
        #

        fused_dim = embed_dim + metadata_embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1),
        )

        print(
            f"DeepV1 initialized ! "
            f"max_seq_len={max_seq_len} (stride={temporal_stride})"
        )

    #
    #   Overrides
    #

    def forward(self, heatmap: Tensor, metadata: Tensor) -> Tensor:
        """
            Forward pass of the model.
        """
        # (B, T, 1, H, W) — stride already applied in GaitDataset
        x = heatmap

        batch_size, seq_len, _, height, width = x.shape

        # Flattening time into the batch dimension for the CNN
        x = x.reshape(batch_size * seq_len, 1, height, width)

        # Chunked encode to avoid OOM on 300+ frames at once
        n_frames = x.size(0)
        chunks: List[Tensor] = []
        for start in range(0, n_frames, FRAME_CNN_CHUNK_SIZE):
            end = min(start + FRAME_CNN_CHUNK_SIZE, n_frames)
            chunks.append(self.frame_encoder(x[start:end]))
        features = torch.cat(chunks, dim=0).reshape(batch_size, seq_len, -1)

        # Temporal transformer over frame embeddings
        features = self.pos_encoding(features)
        encoded = self.temporal_transformer(features)

        # Global average pooling over time
        gait_features = encoded.mean(dim=1)
        meta_features = self.metadata_encoder(metadata)
        fused = torch.cat([gait_features, meta_features], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits

    #
    #   Methods
    #

    def infer(self, heatmap: Tensor, metadata: Tensor) -> Tensor:
        """
            Do not use for training.


            Returns:
                Class probabilities after sigmoid.
        """
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(heatmap, metadata))

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
        epoch_start = time.time()
        self.train()

        loss_tensor: List[float] = []
        corrects = 0
        total = 0
        n = 0
        n_batches = len(train_loader)

        for batch in train_loader:
            heatmaps = batch["heatmap"].to(DEVICE)
            metadata = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = self(heatmaps, metadata)
            loss = criterion(logits, labels)
            loss.backward()

            # Stabilize updates on long sequences
            if MAX_GRAD_NORM > 0:
                nn.utils.clip_grad_norm_(
                    self.parameters(),
                    MAX_GRAD_NORM,
                )

            optimizer.step()

            batch_size = heatmaps.size(0)
            loss_tensor.append(loss.item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            corrects += (preds == labels).sum().item()
            total += batch_size

            if n > 0 and n % LOG_BATCH_INTERVAL == 0:
                elapsed = time.time() - epoch_start
                batches_done = n + 1
                sec_per_batch = elapsed / batches_done
                batches_left = n_batches - batches_done
                eta = sec_per_batch * batches_left
                print(
                    f" => [{tag}] Epoch {epoch_num} "
                    f"Batch: {batches_done} / {n_batches}, "
                    f"Samples: {total} / {train_set_length}, "
                    f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
                )

            n += 1

        writer.add_scalar(
            "Loss/train", np.mean(loss_tensor), epoch_num
        )
        writer.add_scalar(
            "Accuracy/train", corrects / total, epoch_num
        )
        writer.flush()

        return loss_tensor, corrects, total, time.time() - epoch_start

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
            metadata = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with torch.no_grad():
                logits = self(heatmaps, metadata)
                loss = criterion(logits, labels)

                batch_size = heatmaps.size(0)
                loss_tensor.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).float()
                corrects += (preds == labels).sum().item()
                total += batch_size

        return loss_tensor, corrects, total

    #
    #   Persistence
    #

    def save(self, path: str = MODEL_SAVE_PATH) -> None:
        """
            Saves model weights to disk.

            Params:
                - path: Destination file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path} !")


#
#   Training loop
#

def run_training_phase(
    model: DeepV1,
    train_loader: DataLoader[Dict[str, Tensor]],
    val_loader: DataLoader[Dict[str, Tensor]],
    train_len: int,
    optimizer: Adam,
    criterion: nn.Module,
    scheduler: ReduceLROnPlateau,
    writer: SummaryWriter,
    *,
    epochs: int,
    feedback_queue: queue.Queue[int],
    tag: str = "deepv1",
) -> None:
    """
        Training loop with early stopping, LR schedule and
        optional live epoch target override.
    """
    i = 0
    current_epochs = epochs
    best_val_loss = float("inf")
    patience_counter = 0

    while True:
        print(f"\n--- Epoch {i + 1} / {current_epochs} ---")

        train_loss, train_correct, train_total, elapsed = (
            model.train_one_epoch(
                train_loader,
                train_len,
                optimizer,
                criterion,
                i + 1,
                writer,
                tag,
            )
        )
        val_loss, val_correct, val_total = model.evaluate(
            val_loader,
            criterion,
        )

        mean_train_loss = float(np.mean(train_loss))
        mean_val_loss = float(np.mean(val_loss))
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        #
        #   LR schedule and TensorBoard
        #

        scheduler.step(mean_val_loss)

        writer.add_scalar("Loss/val", mean_val_loss, i + 1)
        writer.add_scalar("Accuracy/val", val_acc, i + 1)
        writer.add_scalar(
            f"LR/{tag}",
            optimizer.param_groups[0]["lr"],
            i + 1,
        )
        writer.flush()

        print(
            f" Train loss: {mean_train_loss:.4f}, "
            f"acc: {train_acc:.4f}, "
            f"time: {elapsed:.2f}s"
        )
        print(
            f" Val   loss: {mean_val_loss:.4f}, "
            f"acc: {val_acc:.4f}, "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        #
        #   Early stopping — keep best val checkpoint
        #

        # Keep a best checkpoint only when improvement is meaningful
        if (best_val_loss - mean_val_loss) > BEST_MIN_DELTA:
            best_val_loss = mean_val_loss
            patience_counter = 0
            model.save(BEST_MODEL_PATH)
            print(
                f" New best val loss (< -{BEST_MIN_DELTA:.1e}) "
                "— checkpoint updated."
            )
        else:
            patience_counter += 1
            print(
                f" No val improvement "
                f"({patience_counter} / {EARLY_STOP_PATIENCE})"
            )
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

        i += 1

        #
        #   Optional live epoch target (stdin thread)
        #

        try:
            new_epochs = feedback_queue.get_nowait()
            current_epochs = new_epochs
            print(f"Updated epoch target to {current_epochs}")
        except queue.Empty:
            pass

        if i >= current_epochs:
            break

    #
    #   Last epoch weights (best val is in BEST_MODEL_PATH)
    #

    model.save(MODEL_SAVE_PATH)


#
#   Live epoch input
#

def start_epoch_feedback_thread(
    feedback_queue: queue.Queue[int],
) -> None:
    """
        Reads new epoch targets from stdin while training runs.
    """

    def user_feedback() -> None:
        while True:
            try:
                raw = input(
                    "New epoch count (applies to current run): \n"
                )
                new_epochs = int(raw)
                feedback_queue.put(new_epochs)
                print(f"Epoch target queued: {new_epochs}")
            except ValueError:
                print("Please enter an integer.")
            except EOFError:
                print(
                    "stdin closed; live epoch changes disabled "
                    "(training continues)."
                )
                break
            except KeyboardInterrupt:
                break

    thread = threading.Thread(target=user_feedback, daemon=True)
    thread.start()


def run_training(
    model: DeepV1,
    train_loader: DataLoader[Dict[str, Tensor]],
    val_loader: DataLoader[Dict[str, Tensor]],
    train_labels: List[int],
    *,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    log_dir: str = LOG_DIR,
) -> None:
    """
        Sets up optimizer, scheduler, guards and runs training.
    """

    #
    #   Loss and optimizer
    #

    pos_weight = compute_pos_weight(train_labels)
    print(
        f"BCE pos_weight (PD): {pos_weight.item():.3f} "
        f"[n_train={len(train_labels)}]"
    )

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    writer = SummaryWriter(log_dir=log_dir)

    #
    #   Background stdin thread for epoch target changes
    #

    feedback_queue: queue.Queue[int] = queue.Queue()
    if ENABLE_LIVE_EPOCH_INPUT and sys.stdin.isatty():
        start_epoch_feedback_thread(feedback_queue)
    elif ENABLE_LIVE_EPOCH_INPUT:
        print("Non-interactive terminal — live epoch input disabled.")

    train_len = len(train_loader.dataset) # type: ignore

    run_training_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_len=train_len,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        writer=writer,
        epochs=epochs,
        feedback_queue=feedback_queue,
    )

    writer.close()
    print(f"Best weights (lowest val loss): {BEST_MODEL_PATH}")


#
#   Main
#

if __name__ == "__main__":

    #
    #   Handling dataloading and separating train and validation
    #

    dataset = GaitDataset(DATA_PATH)
    trainset, validationset = build_train_val_subsets(dataset)

    #
    #   Metadata z-score — fit on train indices only
    #

    train_idx = trainset.indices # type: ignore
    meta_mean, meta_std = fit_metadata_stats(dataset, list(train_idx))
    dataset.set_metadata_normalization(meta_mean, meta_std)
    print(
        "Metadata z-score (train): "
        f"{dict(zip(METADATA_COLS, meta_mean.tolist()))}"
    )

    train_labels = [
        int(dataset.metadata.iloc[i]["Group"]) # type: ignore
        for i in train_idx
    ]

    #
    #   DataLoaders
    #

    loader_kwargs: Dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_DATALOADER_WORKERS,
        "pin_memory": PIN_MEMORY and DEVICE.type == "cuda",
        "collate_fn": collate_gait_batch,
    }
    if NUM_DATALOADER_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2

    trainloader = DataLoader(
        trainset,
        shuffle=True,
        **loader_kwargs,
    )
    validationloader = DataLoader(
        validationset,
        shuffle=False,
        **loader_kwargs,
    )

    #
    #   Training
    #

    print(
        f"Sequence tokens per patient: ~{MAX_SEQ_LEN} "
        f"(raw={MAX_RAW_FRAMES}, stride={TEMPORAL_STRIDE})"
    )

    model = DeepV1().to(DEVICE)
    run_training(
        model,
        trainloader,
        validationloader,
        train_labels,
    )
