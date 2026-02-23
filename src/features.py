# src/features.py
import numpy as np

# define the canonical feature key order here so vectorization is stable
FEATURE_KEYS = [
    "ela_mean","ela_std","ela_max",
    "res_mean","res_std","res_max",
    "block_mean","block_std",
    "n_keypoints","n_good_matches","n_clusters",
]

def dict_to_vector(ela_stats, res_stats, block_stats, copy_stats):
    """
    Turn detector stats dicts into a fixed-length numpy vector.
    If a key is missing, fill with 0.0
    """
    merged = {}
    merged.update(ela_stats or {})
    merged.update(res_stats or {})
    merged.update(block_stats or {})
    merged.update(copy_stats or {})

    vec = []
    for k in FEATURE_KEYS:
        val = merged.get(k, 0.0)
        # ensure numeric
        try:
            v = float(val)
        except Exception:
            v = 0.0
        vec.append(v)
    return np.array(vec, dtype=float)

def feature_names():
    return FEATURE_KEYS.copy()