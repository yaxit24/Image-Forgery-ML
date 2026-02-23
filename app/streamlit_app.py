import streamlit as st
import joblib
import numpy as np
import cv2
import os
import sys

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_image_pil
from src.detectors import compute_ela, extract_residual, blockiness_map, copy_move_orb_mask
from src.features import dict_to_vector

MODEL_PATH = "forensics_model.pkl"
SCALER_PATH = "forensics_scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("Image Forgery Detection - Milestone 1")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","tif"])

if uploaded is not None:

    # Load image
    arr, pil = load_image_pil(uploaded)

    st.image(pil, caption="Original Image")

    # ---- Run Detectors ----
    ela_img, ela_stats = compute_ela(pil)
    res_map, res_stats = extract_residual(arr)
    block_map, block_stats = blockiness_map(arr[:,:,0])
    mask, copy_stats = copy_move_orb_mask(arr)

    # ---- Display Visuals ----
    st.image(ela_img, caption="ELA Output")

    st.image(res_map / res_map.max(), caption="Residual Map")

    if mask.sum() > 0:
        overlay = arr.copy()
        overlay[mask == 255] = [255, 0, 0]
        st.image(overlay, caption="Copy-Move Mask Overlay")

    # ---- Feature Extraction ----
    feat = dict_to_vector(ela_stats, res_stats, block_stats, copy_stats)

    feat_scaled = scaler.transform([feat])
    pred = model.predict(feat_scaled)[0]
    prob = model.predict_proba(feat_scaled)[0][1]

    # ---- Prediction ----
    if pred == 1:
        st.error(f"Forged Image (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"Authentic Image (Confidence: {(1-prob)*100:.2f}%)")