#
#   Imports
#

import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import Tensor, nn

from deepv1 import (
    DEVICE,
    FRAME_CNN_CHUNK_SIZE,
    DeepV1,
    GaitDataset,
    fit_metadata_stats,
)

#
#   Constants
#

MODEL_PATH = "./floo/models/deepv1-2/deepv1_fold2_best_ever.pt"
DATA_PATH = "datasets/gait-in-parkinsons-disease-1.0.0/preprocessed/"
OUTPUT_DIR = "./floo/xai_outputs/"

# Correctly classified samples to export (balanced across classes)
MAX_CORRECT_SAMPLES = 4
SAMPLES_PER_CLASS = MAX_CORRECT_SAMPLES // 2

FPS = 100

# Rendering — hsv colormap (same as hm_video.py)
HEATMAP_CMAP = "hsv"
HEATMAP_BRIGHTNESS = 0.45
OVERLAY_ALPHA = 0.72
SHADOW_STRENGTH = 0.55
IMPORTANCE_GAMMA = 2.2

# Grad-CAM target: last conv of the frame CNN
GRADCAM_TARGET_LAYER = "frame_encoder.layer3.conv2"

#
#   Loading
#


def load_model(checkpoint_path: str) -> DeepV1:
    """
        Loads a DeepV1 checkpoint from disk.

        Params:
            - checkpoint_path: Path to the .pt weights file.

        Returns:
            Evaluated model on DEVICE.
    """
    model = DeepV1().to(DEVICE)
    state = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {checkpoint_path} !")
    return model


def setup_dataset(data_path: str) -> GaitDataset:
    """
        Builds the dataset and fits metadata z-score on all patients.

        Params:
            - data_path: Preprocessed dataset root.

        Returns:
            GaitDataset ready for inference.
    """
    dataset = GaitDataset(data_path)
    all_idx = list(range(len(dataset)))
    mean, std = fit_metadata_stats(dataset, all_idx)
    dataset.set_metadata_normalization(mean, std)
    print(f"Dataset ready — {len(dataset)} patients !")
    return dataset


#
#   XAI — spatial (Grad-CAM) and temporal (attention)
#


def _resolve_gradcam_layer(model: DeepV1, dotted_path: str) -> nn.Module:
    """
        Resolves a dotted attribute path on the model.

        Params:
            - model: DeepV1 instance.
            - dotted_path: e.g. "frame_encoder.layer3.conv2".

        Returns:
            Target nn.Module for Grad-CAM hooks.
    """
    module: nn.Module = model
    for part in dotted_path.split("."):
        module = getattr(module, part)
    return module


def _upsample_cam_maps(
    cam: Tensor,
    height: int,
    width: int,
) -> np.ndarray:
    """
        Resizes low-res CAM maps back to heatmap resolution.

        Params:
            - cam: (T, h, w) tensor.
            - height: Target height.
            - width: Target width.

        Returns:
            (T, H, W) float32 array.
    """
    cam_np = cam.detach().cpu().numpy()
    out = np.zeros((cam_np.shape[0], height, width), dtype=np.float32)

    for t in range(cam_np.shape[0]):
        out[t] = cv.resize(
            cam_np[t],
            (width, height),
            interpolation=cv.INTER_LINEAR,
        )

    return out


def extract_grad_cam_maps(
    model: DeepV1,
    heatmap: Tensor,
    metadata: Tensor,
    pred_class: int,
) -> np.ndarray:
    """
        Spatial importance per frame via Grad-CAM on the frame CNN.

        Hook-based Grad-CAM (Selvaraju et al.) — one backward pass
        over the full sequence, yielding T spatial maps.

        Params:
            - model: DeepV1 instance.
            - heatmap: (1, T, 1, H, W) tensor.
            - metadata: (1, N_META) tensor.
            - pred_class: Predicted label used as backward target.

        Returns:
            Array of shape (T, H, W) with per-frame spatial saliency.
    """
    target_layer = _resolve_gradcam_layer(model, GRADCAM_TARGET_LAYER)
    activations_list: list[Tensor] = []
    gradients_list: list[Tensor] = []

    def forward_hook(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        # Frame encoder runs in chunks — accumulate every pass
        activations_list.append(output)

    def backward_hook(
        _module: nn.Module,
        _grad_input: tuple[Tensor | None, ...],
        grad_output: tuple[Tensor, ...],
    ) -> None:
        # Backward visits chunks in reverse order
        gradients_list.insert(0, grad_output[0])

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    model.zero_grad()
    heatmap = heatmap.clone().detach().requires_grad_(True)

    logits = model(heatmap, metadata)
    score = logits if pred_class == 1 else -logits
    score.backward()

    acts = torch.cat(activations_list, dim=0)
    grads = torch.cat(gradients_list, dim=0)

    # Channel-weighted maps — one per encoded frame
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1)
    cam = torch.relu(cam)

    _, _, _, height, width = heatmap.shape
    cam_maps = _upsample_cam_maps(cam, height, width)

    fwd_handle.remove()
    bwd_handle.remove()
    model.zero_grad()

    return cam_maps


def extract_transformer_frame_importance(
    model: DeepV1,
    heatmap: Tensor,
) -> np.ndarray:
    """
        Temporal importance from transformer self-attention.

        Params:
            - model: DeepV1 instance.
            - heatmap: (1, T, 1, H, W) tensor.

        Returns:
            1-D array (T,) — attention mass received per frame.
    """
    model.eval()

    with torch.no_grad():
        batch_size, seq_len, _, height, width = heatmap.shape
        flat = heatmap.reshape(batch_size * seq_len, 1, height, width)

        # Chunked frame encoding (same as DeepV1.forward)
        chunks: list[Tensor] = []
        for start in range(0, flat.size(0), FRAME_CNN_CHUNK_SIZE):
            end = min(start + FRAME_CNN_CHUNK_SIZE, flat.size(0))
            chunks.append(model.frame_encoder(flat[start:end]))
        features = torch.cat(chunks, dim=0).reshape(
            batch_size, seq_len, -1
        )

        features = model.pos_encoding(features)
        layer = model.temporal_transformer.layers[0]

        # Mirror TransformerEncoderLayer pre-attention norm
        if layer.norm_first:
            src = layer.norm1(features)
        else:
            src = features

        _, attn_weights = layer.self_attn(
            src,
            src,
            src,
            need_weights=True,
            average_attn_weights=True,
        )

        # Sum attention received by each key position (frame)
        received = attn_weights[0].sum(dim=0).cpu().numpy()

    return received.astype(np.float32)


#
#   Normalization pipeline
#


def normalize_per_frame_spatial(maps: np.ndarray) -> np.ndarray:
    """
        Min-max each frame's spatial map to [0, 1].

        Params:
            - maps: (T, H, W) saliency array.

        Returns:
            Per-frame normalized array.
    """
    out = maps.astype(np.float32)
    for t in range(out.shape[0]):
        frame = out[t]
        lo, hi = frame.min(), frame.max()
        if hi - lo < 1e-8:
            out[t] = 0.0
        else:
            out[t] = (frame - lo) / (hi - lo)
    return out


def normalize_frame_weights(weights: np.ndarray) -> np.ndarray:
    """
        Min-max transformer frame weights to [0, 1].

        Params:
            - weights: (T,) temporal importance.

        Returns:
            Normalized 1-D weights.
    """
    lo, hi = weights.min(), weights.max()
    if hi - lo < 1e-8:
        return np.zeros_like(weights)
    return (weights - lo) / (hi - lo)


def apply_frame_weighting(
    spatial: np.ndarray,
    frame_weights: np.ndarray,
) -> np.ndarray:
    """
        Scales each spatial map by its temporal importance.

        Params:
            - spatial: (T, H, W) maps.
            - frame_weights: (T,) weights in [0, 1].

        Returns:
            Temporally weighted (T, H, W) array.
    """
    return spatial * frame_weights[:, np.newaxis, np.newaxis]


def normalize_global_video(maps: np.ndarray) -> np.ndarray:
    """
        Min-max over the full video volume (all frames and pixels).

        Params:
            - maps: (T, H, W) array.

        Returns:
            Globally normalized array in [0, 1].
    """
    out = maps.astype(np.float32)
    lo, hi = out.min(), out.max()
    if hi - lo < 1e-8:
        return np.zeros_like(out)
    return (out - lo) / (hi - lo)


#
#   Video rendering
#


def normalize_heatmap_volume(frames: np.ndarray) -> np.ndarray:
    """
        Global min-max on the full sequence (like hm_video.py).

        Params:
            - frames: (T, H, W) pressure channel.

        Returns:
            Values in [0, 0.85] ready for the hsv colormap.
    """
    out = frames.astype(np.float32)
    lo, hi = out.min(), out.max()
    if hi - lo > 1e-8:
        out = (out - lo) / (hi - lo)
    return out * 0.85


def heatmap_to_rgb(
    frame: np.ndarray,
    cmap,
    brightness: float = HEATMAP_BRIGHTNESS,
) -> np.ndarray:
    """
        Applies the hsv colormap to a pre-normalized frame.

        Same channel order as hm_video.py: matplotlib RGB
        written directly to VideoWriter (no RGB/BGR swap).

        Params:
            - frame: (H, W) float array in [0, 0.85].
            - cmap: Matplotlib colormap (hsv).
            - brightness: Global darken factor in [0, 1].

        Returns:
            uint8 image (H, W, 3) ready for VideoWriter.
    """
    return (
        cmap(frame)[:, :, :3] * 255.0 * brightness
    ).astype(np.uint8)


def prepare_importance_display(importance: np.ndarray) -> np.ndarray:
    """
        Stretches importance contrast so peak zones stand out.

        Params:
            - importance: (T, H, W) globally normalized volume.

        Returns:
            Display-ready importance in [0, 1].
    """
    imp = np.power(
        np.clip(importance.astype(np.float32), 0.0, 1.0),
        IMPORTANCE_GAMMA,
    )
    return normalize_global_video(imp)


def blend_importance_overlay(
    base_img: np.ndarray,
    importance: np.ndarray,
    max_alpha: float,
    shadow_strength: float,
) -> np.ndarray:
    """
        White highlight on important pixels, shadow elsewhere.

        Params:
            - base_img: Background frame (H, W, 3) uint8.
            - importance: (H, W) values in [0, 1].
            - max_alpha: Peak white overlay opacity.
            - shadow_strength: Darkening on low-importance pixels.

        Returns:
            Blended uint8 frame (same channel order as hm_video).
    """
    imp = np.clip(importance.astype(np.float32), 0.0, 1.0)

    # Extra shadow where the model assigns low importance
    shadow = 1.0 - shadow_strength * (1.0 - imp)
    base = base_img.astype(np.float32)
    base *= shadow[:, :, np.newaxis]

    # White surcouche on high-importance zones
    alpha = np.clip(imp * max_alpha, 0.0, 1.0)
    alpha_3 = alpha[:, :, np.newaxis]
    white = np.full_like(base, 255.0)

    blended = base * (1.0 - alpha_3) + white * alpha_3
    return blended.astype(np.uint8)


def _write_video_frames(
    frames: list[np.ndarray],
    output_path: str,
    fps: int,
    width: int,
    height: int,
) -> None:
    """
        Writes frames to an MP4 file (same as hm_video.py).

        Params:
            - frames: List of (H, W, 3) uint8 images.
            - output_path: Destination .mp4 path.
            - fps: Video frame rate.
            - width: Frame width.
            - height: Frame height.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()


def export_heatmap_video(
    heatmap_frames: np.ndarray,
    output_path: str,
    fps: int,
) -> None:
    """
        Plain heatmap video (same style as hm_video.py).

        Params:
            - heatmap_frames: (T, H, W) pressure channel for display.
            - output_path: Destination .mp4 path.
            - fps: Video frame rate.
    """
    h, w = heatmap_frames.shape[1], heatmap_frames.shape[2]
    cmap = plt.get_cmap(HEATMAP_CMAP)
    normed = normalize_heatmap_volume(heatmap_frames)

    frames = [
        heatmap_to_rgb(normed[t], cmap, brightness=1.0)
        for t in range(normed.shape[0])
    ]
    _write_video_frames(frames, output_path, fps, w, h)
    print(f"Heatmap video saved to {output_path} !")


def export_xai_video(
    heatmap_frames: np.ndarray,
    importance: np.ndarray,
    output_path: str,
    fps: int,
) -> None:
    """
        Writes an MP4 with dark heatmap and white XAI overlay.

        Params:
            - heatmap_frames: (T, H, W) pressure channel for display.
            - importance: (T, H, W) normalized importance in [0, 1].
            - output_path: Destination .mp4 path.
            - fps: Video frame rate.
    """
    h, w = heatmap_frames.shape[1], heatmap_frames.shape[2]
    cmap = plt.get_cmap(HEATMAP_CMAP)
    normed = normalize_heatmap_volume(heatmap_frames)
    display_imp = prepare_importance_display(importance)

    frames = []
    for t in range(normed.shape[0]):
        base = heatmap_to_rgb(normed[t], cmap)
        frame = blend_importance_overlay(
            base,
            display_imp[t],
            OVERLAY_ALPHA,
            SHADOW_STRENGTH,
        )
        frames.append(frame)

    _write_video_frames(frames, output_path, fps, w, h)
    print(f"XAI video saved to {output_path} !")


#
#   Sample processing
#


def predict_sample(
    model: DeepV1,
    heatmap: Tensor,
    metadata: Tensor,
) -> tuple[int, float]:
    """
        Runs inference on a single patient sample.

        Params:
            - model: DeepV1 instance.
            - heatmap: (1, T, 1, H, W) tensor.
            - metadata: (1, N_META) tensor.

        Returns:
            Tuple (predicted class 0/1, probability of class 1).
        """
    with torch.no_grad():
        prob = model.infer(heatmap, metadata).item()
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


def build_xai_maps(
    model: DeepV1,
    heatmap: Tensor,
    metadata: Tensor,
    pred_class: int,
) -> np.ndarray:
    """
        Full XAI pipeline: Grad-CAM, attention, normalizations.

        Params:
            - model: DeepV1 instance.
            - heatmap: (1, T, 1, H, W) tensor.
            - metadata: (1, N_META) tensor.
            - pred_class: Predicted label for Grad-CAM target.

        Returns:
            (T, H, W) globally normalized importance volume.
    """
    spatial = extract_grad_cam_maps(
        model, heatmap, metadata, pred_class
    )
    temporal = extract_transformer_frame_importance(model, heatmap)

    # Per-frame spatial min-max, then temporal min-max
    spatial = normalize_per_frame_spatial(spatial)
    temporal = normalize_frame_weights(temporal)

    # Weight spatial maps by transformer frame importance
    combined = apply_frame_weighting(spatial, temporal)

    # Global min-max across the whole video
    return normalize_global_video(combined)


def _class_quota_reached(
    class_counts: dict[int, int],
    per_class: int,
) -> bool:
    """
        True when both classes reached their export quota.
    """
    return (
        class_counts[0] >= per_class
        and class_counts[1] >= per_class
    )


def process_correct_samples(
    model: DeepV1,
    dataset: GaitDataset,
    max_samples: int,
    output_dir: str,
    per_class: int = SAMPLES_PER_CLASS,
) -> None:
    """
        Exports XAI videos for correct classifications with
        balanced classes (CO and PD).

        Params:
            - model: DeepV1 instance.
            - dataset: GaitDataset with normalized metadata.
            - max_samples: Total samples to export.
            - output_dir: Directory for output .mp4 files.
            - per_class: Quota per class (CO=0, PD=1).
    """
    os.makedirs(output_dir, exist_ok=True)
    class_counts: dict[int, int] = {0: 0, 1: 0}
    found = 0

    for index in range(len(dataset)):
        if _class_quota_reached(class_counts, per_class):
            break

        sample = dataset[index]
        patient_id = dataset.metadata.iloc[index]["ID"] # type: ignore
        label = int(sample["label"])

        if class_counts[label] >= per_class:
            continue

        heatmap = sample["heatmap"].unsqueeze(0).to(DEVICE)
        metadata = sample["metadata"].unsqueeze(0).to(DEVICE)

        pred, prob = predict_sample(model, heatmap, metadata)

        if pred != label:
            continue

        class_name = "PD" if label == 1 else "CO"
        print(
            f"\n[{found + 1}/{max_samples}] {patient_id} — "
            f"{class_name} — label={label}, pred={pred}, "
            f"prob={prob:.3f}"
        )

        importance = build_xai_maps(
            model, heatmap, metadata, pred
        )

        # Display channel: pressure heatmaps (T, H, W)
        display_frames = (
            sample["heatmap"][:, 0, :, :].cpu().numpy()
        )

        hm_path = os.path.join(
            output_dir,
            f"{patient_id}_hm.mp4",
        )
        xai_path = os.path.join(
            output_dir,
            f"{patient_id}_xai.mp4",
        )

        export_heatmap_video(display_frames, hm_path, FPS)
        export_xai_video(
            display_frames,
            importance,
            xai_path,
            FPS,
        )

        class_counts[label] += 1
        found += 1

    if found == 0:
        print("No correct classification found — no video exported.")
    elif found < max_samples:
        print(
            f"Only {found} correct sample(s) found "
            f"(requested {max_samples}: "
            f"{per_class} CO + {per_class} PD)."
        )
        print(
            f"Class balance: CO={class_counts[0]}, "
            f"PD={class_counts[1]}."
        )


#
#   Main
#

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    model = load_model(MODEL_PATH)
    dataset = setup_dataset(DATA_PATH)

    process_correct_samples(
        model,
        dataset,
        MAX_CORRECT_SAMPLES,
        OUTPUT_DIR,
    )
