# src/localizer.py
"""
Heatmap-based tampering localization.

Fuses multiple detector outputs (ELA, residual, blockiness, copy-move)
into a single binary tampering mask with region statistics.

This is NOT a trained segmentation network — it's a principled fusion
of handcrafted detector heatmaps. With only 84 training images and no
pixel-level ground truth masks, this approach is more honest and useful
than a poorly trained U-Net.
"""

import numpy as np
import cv2


def _normalize_map(m):
    """Normalize a 2D map to [0, 1]. Returns zeros if max is 0."""
    m = m.astype(np.float64)
    mx = m.max()
    if mx > 0:
        return m / mx
    return m


def generate_tampering_mask(ela_img, residual_map, block_map, copymove_mask,
                            weights=(0.4, 0.3, 0.1, 0.2)):
    """
    Fuse detector heatmaps into a binary tampering mask.

    Args:
        ela_img:        HxWx3 ELA output (uint8 or float)
        residual_map:   HxW residual magnitude (float)
        block_map:      HxW blockiness std map (float)
        copymove_mask:  HxW binary mask from ORB detector (uint8, 0 or 255)
        weights:        (ela, residual, blockiness, copymove) fusion weights

    Returns:
        (binary_mask, stats_dict)
        binary_mask: HxW uint8, 0 or 255
        stats_dict:  tampered_area_pct, num_regions, largest_region_pct
    """
    h, w = residual_map.shape[:2]

    # Convert ELA to single-channel (mean across RGB)
    if ela_img.ndim == 3:
        ela_gray = ela_img.astype(np.float64).mean(axis=2)
    else:
        ela_gray = ela_img.astype(np.float64)

    # Resize all maps to same shape (use residual_map as reference)
    ela_resized   = cv2.resize(ela_gray, (w, h))
    block_resized = cv2.resize(block_map.astype(np.float64), (w, h))
    cm_resized    = cv2.resize(copymove_mask.astype(np.float64), (w, h))

    # Normalize each to [0, 1]
    n_ela  = _normalize_map(ela_resized)
    n_res  = _normalize_map(residual_map)
    n_blk  = _normalize_map(block_resized)
    n_cm   = _normalize_map(cm_resized)

    # Weighted fusion
    w_ela, w_res, w_blk, w_cm = weights
    fused = w_ela * n_ela + w_res * n_res + w_blk * n_blk + w_cm * n_cm

    # Normalize fused to [0, 255] for thresholding
    fused_uint8 = (255 * _normalize_map(fused)).astype(np.uint8)

    # Otsu threshold
    _, binary = cv2.threshold(fused_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: remove noise, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # remove small specks
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # fill small holes

    # Region statistics
    total_pixels = h * w
    tampered_pixels = int(np.sum(binary > 0))
    tampered_pct = round(100.0 * tampered_pixels / total_pixels, 2)

    # Count connected components (regions)
    num_labels, labels = cv2.connectedComponents(binary)
    num_regions = num_labels - 1  # label 0 is background

    largest_region_pct = 0.0
    for lbl in range(1, num_labels):
        region_size = int(np.sum(labels == lbl))
        region_pct = 100.0 * region_size / total_pixels
        if region_pct > largest_region_pct:
            largest_region_pct = region_pct

    stats = {
        "tampered_area_pct":    tampered_pct,
        "num_regions":          num_regions,
        "largest_region_pct":   round(largest_region_pct, 2),
    }

    return binary, fused_uint8, stats


def overlay_mask(image_rgb, binary_mask, color=(255, 0, 0), alpha=0.4):
    """
    Overlay a red-tinted mask on the original image.

    Args:
        image_rgb:    HxWx3 uint8
        binary_mask:  HxW uint8 (0 or 255)
        color:        overlay color (R, G, B)
        alpha:        overlay transparency

    Returns:
        HxWx3 uint8 overlay image
    """
    overlay = image_rgb.copy()
    mask_bool = binary_mask > 0

    # Resize mask if dimensions don't match
    if binary_mask.shape[:2] != image_rgb.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (image_rgb.shape[1], image_rgb.shape[0]))
        mask_bool = binary_mask > 0

    # Blend
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            np.clip(overlay[:, :, c] * (1 - alpha) + color[c] * alpha, 0, 255).astype(np.uint8),
            overlay[:, :, c]
        )

    return overlay
