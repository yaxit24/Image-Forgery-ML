# Method:1 ELA Error Level Analysis: to detedct the tempered images by detedcting the differences in JPEG compression history.
from PIL import Image, ImageChops, ImageEnhance
import io, numpy as np
    # Scale = brighness amplification
    # quality = jpeg recompression quality
def compute_ela(pil_img, quality=90, scale=10):
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)

    recompressed = Image.open(buf).convert('RGB')
    diff = ImageChops.difference(pil_img.convert('RGB'), recompressed)
    ela = ImageEnhance.Brightness(diff).enhance(scale)

    ela_arr = np.array(ela)

    # --- statistics for ML ---
    gray = ela_arr.mean(axis=2)
    stats = {
        "ela_mean": float(gray.mean()),
        "ela_std": float(gray.std()),
        "ela_max": float(gray.max())
    }

    return ela_arr, stats

# Mrthod 2: PRNU- Photo Response Non-Uniformity:Every camera sensor has:vSlight manufacturing imperfections, Pixel-to-pixel sensitivity variations. Image = True Scene + Sensor Noise it is like as A fingerprint left by the camera sensor.

from skimage.restoration import denoise_wavelet
import numpy as np

def extract_residual(arr_rgb):
    img = arr_rgb.astype('float32') / 255.0

    den = denoise_wavelet(
        img,
        channel_axis=-1,   # updated for new skimage
        convert2ycbcr=False,
        rescale_sigma=True
    )

    residual = img - den
    residual_mag = np.mean(np.abs(residual), axis=2)

    stats = {
        "res_mean": float(residual_mag.mean()),
        "res_std": float(residual_mag.std()),
        "res_max": float(residual_mag.max())
    }

    return residual_mag, stats

#Methof 3: 
import numpy as np

def blockiness_map(gray_arr, block=8):
    h, w = gray_arr.shape
    map_out = np.zeros_like(gray_arr, dtype=float)

    for i in range(0, h - block + 1, block):
        for j in range(0, w - block + 1, block):
            blk = gray_arr[i:i+block, j:j+block]
            map_out[i:i+block, j:j+block] = blk.std()

    stats = {
        "block_mean": float(map_out.mean()),
        "block_std": float(map_out.std())
    }

    return map_out, stats

#Method 4:
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def copy_move_orb_mask(rgb_arr):
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    kps, desc = orb.detectAndCompute(gray, None)

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if desc is None or len(kps) < 10:
        return mask, {
            "n_keypoints": 0,
            "n_good_matches": 0,
            "n_clusters": 0
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc, desc, k=2)

    good_matches = []

    for m, n in matches:
        if m.trainIdx == m.queryIdx:
            continue
        if m.distance < 0.75 * n.distance:
            pt1 = kps[m.queryIdx].pt
            pt2 = kps[m.trainIdx].pt

            # avoid matching same region
            if np.linalg.norm(np.array(pt1) - np.array(pt2)) > 10:
                good_matches.append(pt1)

    if len(good_matches) < 8:
        return mask, {
            "n_keypoints": len(kps),
            "n_good_matches": len(good_matches),
            "n_clusters": 0
        }

    pts = np.array(good_matches)

    clustering = DBSCAN(eps=20, min_samples=3).fit(pts)
    labels = clustering.labels_

    clusters = 0

    for lbl in set(labels):
        if lbl == -1:
            continue

        clusters += 1
        cluster_pts = pts[labels == lbl].astype(int)

        x, y, wc, hc = cv2.boundingRect(cluster_pts)
        mask[y:y+hc, x:x+wc] = 255

    stats = {
        "n_keypoints": len(kps),
        "n_good_matches": len(good_matches),
        "n_clusters": clusters
    }

    return mask, stats
    # BFMatcher + knnMatch, ratio test, remove self-matches, cluster source pts
    # produce mask (HxW uint8 0/255)
    # (See detailed implementation in detectors.py)
    
    