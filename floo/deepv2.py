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

# Held-out test set (never used during training / CV hyper-tuning)
TEST_RATIO = 0.15
TEST_SEED = 43

# Stratified k-fold on train+val pool (~158 patients -> ~32 per fold)
USE_CROSS_VALIDATION = True
N_FOLDS = 5
CV_LOG_DIR = "./floo/runs/deepv2-cv"
CV_MODEL_DIR = "./floo/models/deepv2/cv"

#
#   Training hyperparameters
#

BATCH_SIZE = 10
NUM_DATALOADER_WORKERS = 4
PIN_MEMORY = True
EPOCHS = 20 
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 7
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

FEATURE_DIM = 64
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


MAX_SEQ_DEPTH = seq_len_after_stride(MAX_RAW_FRAMES, TEMPORAL_STRIDE)

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

# Fuse demographics into the classifier (False = gait heatmaps only)
USE_METADATA = False

#
#   Logging and checkpoints
#

LOG_BATCH_INTERVAL = 2
LOG_DIR = "./floo/runs/deepv2-3"
MODEL_SAVE_PATH = "./floo/models/deepv2/deepv2.pt"
BEST_MODEL_PATH = "./floo/models/deepv2/deepv2_best.pt"

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
    pool_indices: List[int] | None = None,
) -> Tuple[List[int], List[int]]:
    """
        Stratified train / validation index split per class.
    """
    if pool_indices is None:
        pool_indices = list(range(len(labels)))

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []

    # Split each class separately then merge indices
    for label in (0, 1):
        cls_idx = [i for i in pool_indices if labels[i] == label]
        rng.shuffle(cls_idx)
        n_train = int(train_ratio * len(cls_idx))
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:])

    return train_idx, val_idx


def stratified_holdout_indices(
    labels: List[int],
    pool_indices: List[int],
    holdout_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
        Stratified hold-out from a pool (e.g. untouched test set).
    """
    rng = np.random.default_rng(seed)
    remaining: List[int] = []
    holdout: List[int] = []

    for label in (0, 1):
        cls_idx = [i for i in pool_indices if labels[i] == label]
        rng.shuffle(cls_idx)
        n_holdout = int(holdout_ratio * len(cls_idx))
        if n_holdout < 1 and len(cls_idx) > 1:
            n_holdout = 1
        holdout.extend(cls_idx[:n_holdout])
        remaining.extend(cls_idx[n_holdout:])

    rng.shuffle(remaining)
    rng.shuffle(holdout)
    return remaining, holdout


def stratified_kfold_indices(
    labels: List[int],
    n_folds: int,
    seed: int,
    pool_indices: List[int] | None = None,
) -> List[Tuple[List[int], List[int]]]:
    """
        Stratified k-fold index splits (one list entry per fold).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    if pool_indices is None:
        pool_indices = list(range(len(labels)))

    rng = np.random.default_rng(seed)
    fold_buckets: List[List[int]] = [[] for _ in range(n_folds)]

    # Round-robin per class keeps class balance in each fold
    for label in (0, 1):
        cls_idx = [i for i in pool_indices if labels[i] == label]
        rng.shuffle(cls_idx)
        for i, idx in enumerate(cls_idx):
            fold_buckets[i % n_folds].append(int(idx))

    splits: List[Tuple[List[int], List[int]]] = []
    for fold_id in range(n_folds):
        val_idx = list(fold_buckets[fold_id])
        train_idx: List[int] = []
        for j in range(n_folds):
            if j != fold_id:
                train_idx.extend(fold_buckets[j])
        rng.shuffle(train_idx)
        splits.append((train_idx, val_idx))

    return splits


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
#   Classification metrics
#

class BinaryMetricAccumulator:
    """
        Aggregates binary counts and logits for multi-metric
        reporting (accuracy, precision, recall, F1, AUC, etc.).
    """

    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.logits: List[Tensor] = []
        self.labels: List[Tensor] = []

    def update(
        self,
        preds: Tensor,
        labels: Tensor,
        logits: Tensor | None = None,
    ) -> None:
        labels_i = labels.long()
        preds_i = preds.long()
        self.tp += ((preds_i == 1) & (labels_i == 1)).sum().item()
        self.tn += ((preds_i == 0) & (labels_i == 0)).sum().item()
        self.fp += ((preds_i == 1) & (labels_i == 0)).sum().item()
        self.fn += ((preds_i == 0) & (labels_i == 1)).sum().item()
        if logits is not None:
            self.logits.append(logits.detach().cpu())
            self.labels.append(labels.detach().cpu())

    def compute(self) -> Dict[str, float]:
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "specificity": 0.0,
                "f1": 0.0,
                "auc": float("nan"),
                "pred_positive_rate": 0.0,
            }

        accuracy = (self.tp + self.tn) / total
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        specificity = (
            self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
        )
        precision = (
            self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        )
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        balanced_accuracy = (recall + specificity) / 2.0
        pred_positive_rate = (self.tp + self.fp) / total

        auc = float("nan")
        if self.logits:
            logits_cat = torch.cat(self.logits)
            labels_cat = torch.cat(self.labels)
            auc = binary_auc_from_logits(logits_cat, labels_cat)

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "auc": auc,
            "pred_positive_rate": pred_positive_rate,
        }


def binary_auc_from_logits(
    logits: Tensor,
    labels: Tensor,
) -> float:
    """
        ROC-AUC via rank statistic (no sklearn dependency).
    """
    probs = torch.sigmoid(logits).numpy()
    y = labels.numpy().astype(np.int64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = np.argsort(np.argsort(probs))
    rank_sum_pos = float(ranks[pos].sum())
    auc = (rank_sum_pos - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def print_cv_summary(
    fold_metrics: List[Dict[str, float]],
    n_folds: int,
) -> None:
    """
        Prints mean ± std of each metric across CV folds.
    """
    if not fold_metrics:
        return

    print(f"\n{'=' * 60}")
    print(f"Cross-validation summary ({n_folds} folds)")
    print(f"{'=' * 60}")

    metric_keys = fold_metrics[0].keys()
    for key in metric_keys:
        values = [
            m[key] for m in fold_metrics
            if key in m and not np.isnan(m[key])
        ]
        if not values:
            continue
        mean_v = float(np.mean(values))
        std_v = float(np.std(values))
        print(f" {key:22s}  {mean_v:.4f}  ±  {std_v:.4f}")

    for fold_id, metrics in enumerate(fold_metrics, start=1):
        print(
            f" fold {fold_id}: "
            f"{format_metric_line(metrics)}"
        )


def format_metric_line(metrics: Dict[str, float]) -> str:
    """
        Compact one-line summary for batch / epoch logs.
    """
    auc = metrics["auc"]
    auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
    return (
        f"acc={metrics['accuracy']:.3f}, "
        f"bal={metrics['balanced_accuracy']:.3f}, "
        f"prec={metrics['precision']:.3f}, "
        f"rec={metrics['recall']:.3f}, "
        f"spec={metrics['specificity']:.3f}, "
        f"f1={metrics['f1']:.3f}, "
        f"auc={auc_str}, "
        f"pred_PD={metrics['pred_positive_rate']:.2f}"
    )


def log_metrics_to_tensorboard(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    step: int,
    prefix: str,
) -> None:
    """
        Logs all classification metrics to TensorBoard.
    """
    for key, value in metrics.items():
        if np.isnan(value):
            continue
        writer.add_scalar(f"{prefix}/{key}", value, step)


#
#   Model definition
#

class ResidualBlock3D(nn.Module):
    """
        Residual block with 3D convolutions for spatiotemporal
        feature extraction.
    """

    def __init__(self, channels: int, num_groups: int):
        super().__init__()

        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv3d(
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


class VolumeEncoder3D(nn.Module):
    """
        3D CNN backbone that maps a gait heatmap volume to a
        fixed-size feature vector.
    """

    def __init__(self, feature_dim: int, num_groups: int):
        super().__init__()

        # Stem — spatial 224 -> 56, temporal depth preserved
        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                32,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

        self.layer1 = ResidualBlock3D(32, num_groups)

        # Spatiotemporal down — T/2, H/2, W/2
        self.down1 = nn.Sequential(
            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = ResidualBlock3D(64, num_groups)

        # Spatial down only — H/2, W/2
        self.down2 = nn.Sequential(
            nn.Conv3d(
                64,
                128,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = ResidualBlock3D(128, num_groups)

        # Temporal down — T/2
        self.down3 = nn.Sequential(
            nn.Conv3d(
                128,
                128,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups, 128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Linear(128, feature_dim)

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
        heatmap volume; demographics are fused before the head.

        Inputs:
            - heatmap: Tensor of shape (batch_size, T, C, H, W)
            - metadata: Tensor of shape (batch_size, N_METADATA)

        Outputs:
            - classification: Tensor of shape (batch_size,)
    """

    def __init__(
        self,
        *,
        feature_dim: int = FEATURE_DIM,
        num_groups: int = NUM_GROUPS,
        n_metadata: int = N_METADATA,
        metadata_embed_dim: int = METADATA_EMBED_DIM,
        dropout: float = DROPOUT,
        max_seq_depth: int = MAX_SEQ_DEPTH,
    ):
        super().__init__()

        self.max_seq_depth = max_seq_depth
        self.use_metadata = USE_METADATA

        #
        #   3D volume encoder
        #

        self.volume_encoder = VolumeEncoder3D(feature_dim, num_groups)

        #
        #   Demographics branch (optional)
        #

        if self.use_metadata:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(n_metadata, metadata_embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            fused_dim = feature_dim + metadata_embed_dim
        else:
            self.metadata_encoder = None
            fused_dim = feature_dim

        #
        #   Classification head
        #

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1),
        )

        meta_status = "on" if self.use_metadata else "off"
        print(
            f"DeepV2 initialized ! metadata={meta_status}, "
            f"max_seq_depth={max_seq_depth} "
            f"(stride={TEMPORAL_STRIDE})"
        )

    #
    #   Overrides
    #

    def forward(self, heatmap: Tensor, metadata: Tensor) -> Tensor:
        """
            Forward pass of the model.
        """
        # (B, T, 1, H, W) — stride already applied in GaitDataset
        seq_len = heatmap.size(1)
        if seq_len > self.max_seq_depth:
            raise ValueError(
                f"Sequence depth {seq_len} exceeds max_seq_depth "
                f"{self.max_seq_depth}. Increase MAX_SEQ_DEPTH "
                f"(e.g. lower TEMPORAL_STRIDE)."
            )

        x = heatmap.permute(0, 2, 1, 3, 4) # (B, 1, T, H, W) for Conv3d

        gait_features = self.volume_encoder(x)

        if self.use_metadata:
            meta_features = self.metadata_encoder(metadata) # type: ignore
            fused = torch.cat([gait_features, meta_features], dim=1)
        else:
            fused = gait_features

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
    ) -> Tuple[List[float], Dict[str, float], float]:
        """
            Training loop for one epoch.


            Returns:
                - Loss values per batch
                - Epoch classification metrics
                - Elapsed time in seconds
        """
        epoch_start = time.time()
        self.train()

        loss_tensor: List[float] = []
        epoch_metrics = BinaryMetricAccumulator()
        n = 0
        n_batches = len(train_loader)
        samples_seen = 0

        for batch in train_loader:
            heatmaps = batch["heatmap"].to(DEVICE)
            metadata = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = self(heatmaps, metadata)
            loss = criterion(logits, labels)
            loss.backward()

            # Stabilize updates on long 3D volumes
            if MAX_GRAD_NORM > 0:
                nn.utils.clip_grad_norm_(
                    self.parameters(),
                    MAX_GRAD_NORM,
                )

            optimizer.step()

            batch_size = heatmaps.size(0)
            loss_tensor.append(loss.item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            epoch_metrics.update(preds, labels, logits)
            samples_seen += batch_size

            if n > 0 and n % LOG_BATCH_INTERVAL == 0:
                elapsed = time.time() - epoch_start
                batches_done = n + 1
                sec_per_batch = elapsed / batches_done
                batches_left = n_batches - batches_done
                eta = sec_per_batch * batches_left
                print(
                    f" => [{tag}] Epoch {epoch_num} "
                    f"Batch: {batches_done} / {n_batches}, "
                    f"Samples: {samples_seen} / {train_set_length}, "
                    f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
                )

            n += 1

        mean_loss = float(np.mean(loss_tensor))
        metrics = epoch_metrics.compute()

        writer.add_scalar("Loss/train", mean_loss, epoch_num)
        log_metrics_to_tensorboard(writer, metrics, epoch_num, "train")
        writer.flush()

        return loss_tensor, metrics, time.time() - epoch_start

    def evaluate(
        self,
        val_loader: DataLoader[Dict[str, Tensor]],
        criterion: Any,
        epoch_num: int = 0,
        writer: SummaryWriter | None = None,
        tag: str = "deepv2",
    ) -> Tuple[List[float], Dict[str, float]]:
        """
            Validation loop.


            Returns:
                - Loss values per batch
                - Epoch classification metrics
        """
        self.eval()

        loss_tensor: List[float] = []
        epoch_metrics = BinaryMetricAccumulator()
        n = 0

        for batch in val_loader:
            heatmaps = batch["heatmap"].to(DEVICE)
            metadata = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with torch.no_grad():
                logits = self(heatmaps, metadata)
                loss = criterion(logits, labels)

                loss_tensor.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).float()
                epoch_metrics.update(preds, labels, logits)

            n += 1

        return loss_tensor, epoch_metrics.compute()

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
    model: DeepV2,
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
    tag: str = "deepv2",
    model_save_path: str = MODEL_SAVE_PATH,
    best_model_path: str = BEST_MODEL_PATH,
) -> Dict[str, float]:
    """
        Training loop with early stopping, LR schedule and
        optional live epoch target override.


        Returns:
            Best-fold validation metrics (at lowest val loss).
    """
    i = 0
    current_epochs = epochs
    best_val_loss = float("inf")
    best_val_metrics: Dict[str, float] = {}
    patience_counter = 0

    while True:
        print(f"\n--- Epoch {i + 1} / {current_epochs} ---")

        train_loss, train_metrics, elapsed = model.train_one_epoch(
            train_loader,
            train_len,
            optimizer,
            criterion,
            i + 1,
            writer,
            tag,
        )
        val_loss, val_metrics = model.evaluate(
            val_loader,
            criterion,
            epoch_num=i + 1,
            writer=writer,
            tag=tag,
        )

        mean_train_loss = float(np.mean(train_loss))
        mean_val_loss = float(np.mean(val_loss))

        #
        #   LR schedule and TensorBoard
        #

        scheduler.step(mean_val_loss)

        writer.add_scalar("Loss/val", mean_val_loss, i + 1)
        log_metrics_to_tensorboard(writer, val_metrics, i + 1, "val")
        writer.add_scalar(
            f"LR/{tag}",
            optimizer.param_groups[0]["lr"],
            i + 1,
        )
        writer.flush()

        print(
            f" Train loss: {mean_train_loss:.4f}, "
            f"time: {elapsed:.2f}s"
        )
        print(f" Train metrics: {format_metric_line(train_metrics)}")
        print(
            f" Val   loss: {mean_val_loss:.4f}, "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )
        print(f" Val   metrics: {format_metric_line(val_metrics)}")

        #
        #   Early stopping — keep best val checkpoint
        #

        if (best_val_loss - mean_val_loss) > BEST_MIN_DELTA:
            best_val_loss = mean_val_loss
            best_val_metrics = dict(val_metrics)
            patience_counter = 0
            model.save(best_model_path)
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

    model.save(model_save_path)
    return best_val_metrics


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


def make_loader_kwargs() -> Dict[str, Any]:
    """
        Shared DataLoader options for train / val / test.
    """
    loader_kwargs: Dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_DATALOADER_WORKERS,
        "pin_memory": PIN_MEMORY and DEVICE.type == "cuda",
        "persistent_workers": NUM_DATALOADER_WORKERS > 0,
        "collate_fn": collate_gait_batch,
    }
    if NUM_DATALOADER_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2
    return loader_kwargs


def build_test_dataloader(
    dataset: GaitDataset,
    test_idx: List[int],
    train_idx_for_meta: List[int],
) -> DataLoader:
    """
        Test loader — metadata z-score fitted on train indices only.
    """
    meta_mean, meta_std = fit_metadata_stats(dataset, train_idx_for_meta)
    dataset.set_metadata_normalization(meta_mean, meta_std)

    return DataLoader(
        Subset(dataset, test_idx),
        shuffle=False,
        **make_loader_kwargs(),
    )


def evaluate_saved_model_on_test(
    checkpoint_path: str,
    dataset: GaitDataset,
    test_idx: List[int],
    train_idx_for_meta: List[int],
    *,
    tag: str = "test",
) -> Dict[str, float]:
    """
        Loads a checkpoint and runs a single pass on the test set.
    """
    if not test_idx:
        print("Test set is empty — skipping test evaluation.")
        return {}

    test_loader = build_test_dataloader(
        dataset,
        test_idx,
        train_idx_for_meta,
    )

    model = DeepV2().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    _, metrics = model.evaluate(test_loader, criterion, tag=tag)

    print(f"\n--- Test evaluation ({tag}) ---")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Test size: {len(test_idx)}")
    print(f" Test metrics: {format_metric_line(metrics)}")
    return metrics


def build_fold_dataloaders(
    dataset: GaitDataset,
    train_idx: List[int],
    val_idx: List[int],
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
        Metadata z-score, subsets and DataLoaders for one fold.
    """
    meta_mean, meta_std = fit_metadata_stats(dataset, train_idx)
    dataset.set_metadata_normalization(meta_mean, meta_std)
    print(
        "Metadata z-score (train fold): "
        f"{dict(zip(METADATA_COLS, meta_mean.tolist()))}"
    )

    train_labels = [
        int(dataset.metadata.iloc[i]["Group"]) # type: ignore
        for i in train_idx
    ]

    loader_kwargs = make_loader_kwargs()

    trainloader = DataLoader(
        Subset(dataset, train_idx),
        shuffle=True,
        **loader_kwargs,
    )
    valloader = DataLoader(
        Subset(dataset, val_idx),
        shuffle=False,
        **loader_kwargs,
    )
    return trainloader, valloader, train_labels


def run_training(
    model: DeepV2,
    train_loader: DataLoader[Dict[str, Tensor]],
    val_loader: DataLoader[Dict[str, Tensor]],
    train_labels: List[int],
    *,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    log_dir: str = LOG_DIR,
    model_save_path: str = MODEL_SAVE_PATH,
    best_model_path: str = BEST_MODEL_PATH,
    enable_live_input: bool | None = None,
    tag: str = "deepv2",
) -> Dict[str, float]:
    """
        Sets up optimizer, scheduler, guards and runs training.


        Returns:
            Best validation metrics for this run / fold.
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
    use_live = (
        ENABLE_LIVE_EPOCH_INPUT
        if enable_live_input is None
        else enable_live_input
    )
    if use_live and sys.stdin.isatty():
        start_epoch_feedback_thread(feedback_queue)
    elif use_live:
        print("Non-interactive terminal — live epoch input disabled.")

    train_len = len(train_loader.dataset) # type: ignore

    best_metrics = run_training_phase(
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
        tag=tag,
        model_save_path=model_save_path,
        best_model_path=best_model_path,
    )

    writer.close()
    print(f"Best weights (lowest val loss): {best_model_path}")
    return best_metrics


def run_cross_validation(
    dataset: GaitDataset,
    train_val_idx: List[int],
    test_idx: List[int],
    *,
    n_folds: int = N_FOLDS,
    epochs: int = EPOCHS,
) -> None:
    """
        Stratified k-fold training with per-fold checkpoints and
        a summary of validation metrics across folds.
    """
    labels = dataset.metadata["Group"].astype(int).tolist()
    folds = stratified_kfold_indices(
        labels,
        n_folds,
        SEED,
        pool_indices=train_val_idx,
    )
    fold_metrics: List[Dict[str, float]] = []
    test_metrics: List[Dict[str, float]] = []

    os.makedirs(CV_MODEL_DIR, exist_ok=True)
    os.makedirs(CV_LOG_DIR, exist_ok=True)

    print(
        f"Starting {n_folds}-fold stratified CV "
        f"on {len(train_val_idx)} patients "
        f"(test hold-out: {len(test_idx)} untouched)."
    )

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n{'=' * 60}")
        print(
            f"Fold {fold_id} / {n_folds} — "
            f"train={len(train_idx)}, val={len(val_idx)}"
        )
        print(f"{'=' * 60}")

        trainloader, valloader, train_labels = build_fold_dataloaders(
            dataset,
            train_idx,
            val_idx,
        )

        best_path = f"{CV_MODEL_DIR}/deepv2_fold{fold_id}_best.pt"

        model = DeepV2().to(DEVICE)
        metrics = run_training(
            model,
            trainloader,
            valloader,
            train_labels,
            epochs=epochs,
            log_dir=f"{CV_LOG_DIR}/fold_{fold_id}",
            model_save_path=(
                f"{CV_MODEL_DIR}/deepv2_fold{fold_id}.pt"
            ),
            best_model_path=best_path,
            enable_live_input=False,
            tag=f"deepv2_fold{fold_id}",
        )
        fold_metrics.append(metrics)

        if test_idx:
            test_fold_metrics = evaluate_saved_model_on_test(
                best_path,
                dataset,
                test_idx,
                train_idx,
                tag=f"fold_{fold_id}_test",
            )
            test_metrics.append(test_fold_metrics)

    print_cv_summary(fold_metrics, n_folds)

    if test_metrics:
        print(f"\n{'=' * 60}")
        print("Test set summary (best checkpoint per fold)")
        print_cv_summary(test_metrics, len(test_metrics))


#
#   Main
#

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    #
    #   Dataset
    #

    dataset = GaitDataset(DATA_PATH)
    all_labels = dataset.metadata["Group"].astype(int).tolist()
    all_indices = list(range(len(all_labels)))

    #
    #   Held-out test set (untouched until final evaluation)
    #

    train_val_idx, test_idx = stratified_holdout_indices(
        all_labels,
        all_indices,
        TEST_RATIO,
        TEST_SEED,
    )
    print(
        f"Split: train+val={len(train_val_idx)}, "
        f"test={len(test_idx)} (held out, ratio={TEST_RATIO})"
    )

    print(
        f"Sequence depth per patient: ~{MAX_SEQ_DEPTH} "
        f"(raw={MAX_RAW_FRAMES}, stride={TEMPORAL_STRIDE})"
    )

    #
    #   Cross-validation or single stratified split
    #

    if USE_CROSS_VALIDATION:
        run_cross_validation(
            dataset,
            train_val_idx,
            test_idx,
            n_folds=N_FOLDS,
            epochs=EPOCHS,
        )
    else:
        train_idx, val_idx = stratified_train_val_indices(
            all_labels,
            TRAIN_VALIDATION_RATIO,
            SEED,
            pool_indices=train_val_idx,
        )
        print(
            f"Stratified split (train+val pool): "
            f"train={len(train_idx)}, val={len(val_idx)}"
        )

        trainloader, validationloader, train_labels = (
            build_fold_dataloaders(dataset, train_idx, val_idx)
        )

        model = DeepV2().to(DEVICE)
        run_training(
            model,
            trainloader,
            validationloader,
            train_labels,
        )

        evaluate_saved_model_on_test(
            BEST_MODEL_PATH,
            dataset,
            test_idx,
            train_idx,
            tag="test",
        )
