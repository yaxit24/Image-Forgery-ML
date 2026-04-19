import streamlit as st
import joblib
import numpy as np
import os
import sys
import traceback

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_image_pil
from src.detectors import compute_ela, extract_residual, blockiness_map, copy_move_orb_mask
from src.features import dict_to_vector
from src.ensemble import ensemble_predict
from src.localizer import generate_tampering_mask, overlay_mask
from src.report_generator import generate_report
from src.report_exporter import export_pdf, export_markdown

# ---- Page Config ----
st.set_page_config(
    page_title="Image Forgery Detection",
    page_icon="🔍",
    layout="wide",
)

# ---- Model Loading (cached) ----

@st.cache_resource
def load_rf_model():
    """Load the Random Forest model and scaler from Milestone 1."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "forensics_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "..", "forensics_scaler.pkl")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.warning(f"RF model not found: {e}")
        return None, None


@st.cache_resource
def load_cnn_model():
    """Load the CNN classifier (MobileNetV2 head)."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "cnn_model.pth")
    try:
        from src.cnn_classifier import load_model
        model = load_model(model_path)
        return model
    except Exception as e:
        st.warning(f"CNN model not found: {e}")
        return None


# ---- Session State ----

if "history" not in st.session_state:
    st.session_state.history = []  # list of {filename, verdict, confidence, timestamp}


# ---- Sidebar: History ----

with st.sidebar:
    st.header("🕵️ Analysis History")

    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            icon = "🔴" if entry["verdict"] == "forged" else "🟢"
            st.markdown(
                f"{icon} **{entry['filename']}** — "
                f"{entry['verdict'].upper()} ({entry['confidence']:.0%})"
            )
    else:
        st.caption("No analyses yet. Upload an image to begin.")

    st.divider()
    st.caption("Milestone 2 — AI Forensic Assistant")
    st.caption("Models: MobileNetV2 CNN + Random Forest")


# ---- Main Area ----

st.title("🔍 Image Forgery Detection")
st.markdown("*AI-powered forensic analysis with deep learning and structured reporting*")

uploaded = st.file_uploader(
    "Upload an image for analysis",
    type=["jpg", "jpeg", "png", "tif", "bmp"],
    help="Supported formats: JPEG, PNG, TIFF, BMP"
)

if uploaded is not None:
    filename = uploaded.name

    # ---- Load Image ----
    try:
        arr, pil = load_image_pil(uploaded)
    except Exception as e:
        st.error(f"❌ Failed to load image: {e}")
        st.stop()

    # ---- Progress Bar ----
    progress = st.progress(0, text="Starting analysis...")

    # ---- Run Detectors ----
    analysis_data = {}
    detector_errors = []

    # ELA
    progress.progress(10, text="Running ELA analysis...")
    try:
        ela_img, ela_stats = compute_ela(pil)
        analysis_data["ela_stats"] = ela_stats
    except Exception as e:
        detector_errors.append(f"ELA: {e}")
        ela_img, ela_stats = np.zeros_like(arr), {}

    # Residual (PRNU)
    progress.progress(25, text="Extracting noise residual...")
    try:
        res_map, res_stats = extract_residual(arr)
        analysis_data["residual_stats"] = res_stats
    except Exception as e:
        detector_errors.append(f"Residual: {e}")
        res_map, res_stats = np.zeros(arr.shape[:2]), {}

    # Blockiness
    progress.progress(40, text="Analyzing block artifacts...")
    try:
        gray = arr[..., 0] if arr.ndim == 3 else arr
        block_map, block_stats = blockiness_map(gray)
        analysis_data["block_stats"] = block_stats
    except Exception as e:
        detector_errors.append(f"Blockiness: {e}")
        block_map, block_stats = np.zeros(arr.shape[:2]), {}

    # Copy-Move
    progress.progress(55, text="Detecting copy-move forgery...")
    try:
        cm_mask, copy_stats = copy_move_orb_mask(arr)
        analysis_data["copy_stats"] = copy_stats
    except Exception as e:
        detector_errors.append(f"Copy-Move: {e}")
        cm_mask, copy_stats = np.zeros(arr.shape[:2], dtype=np.uint8), {}

    # ---- CNN Prediction ----
    progress.progress(65, text="Running CNN classifier...")
    cnn_model = load_cnn_model()
    if cnn_model is not None:
        try:
            from src.cnn_classifier import predict as cnn_predict
            cnn_label, cnn_conf = cnn_predict(cnn_model, pil)
        except Exception as e:
            detector_errors.append(f"CNN: {e}")
            cnn_label, cnn_conf = "authentic", 0.5
    else:
        cnn_label, cnn_conf = "authentic", 0.5

    # ---- RF Prediction ----
    progress.progress(75, text="Running Random Forest classifier...")
    rf_model, rf_scaler = load_rf_model()
    if rf_model is not None and rf_scaler is not None:
        try:
            feat = dict_to_vector(ela_stats, res_stats, block_stats, copy_stats)
            feat_scaled = rf_scaler.transform([feat])
            rf_pred = rf_model.predict(feat_scaled)[0]
            rf_prob = rf_model.predict_proba(feat_scaled)[0][1]
        except Exception as e:
            detector_errors.append(f"RF: {e}")
            rf_pred, rf_prob = 0, 0.5
    else:
        rf_pred, rf_prob = 0, 0.5

    # ---- Ensemble ----
    progress.progress(80, text="Computing ensemble decision...")
    ensemble = ensemble_predict(cnn_label, cnn_conf, rf_pred, rf_prob)
    analysis_data["ensemble"] = ensemble

    # ---- Localization ----
    progress.progress(85, text="Generating localization mask...")
    try:
        binary_mask, fused_heatmap, loc_stats = generate_tampering_mask(
            ela_img, res_map, block_map, cm_mask
        )
        overlay_img = overlay_mask(arr, binary_mask)
        analysis_data["localization"] = loc_stats
    except Exception as e:
        detector_errors.append(f"Localizer: {e}")
        binary_mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        fused_heatmap = np.zeros(arr.shape[:2], dtype=np.uint8)
        overlay_img = arr.copy()
        analysis_data["localization"] = {"tampered_area_pct": 0, "num_regions": 0, "largest_region_pct": 0}

    # ---- Generate Report ----
    progress.progress(90, text="Generating forensic report...")
    report_text, report_source = generate_report(analysis_data, filename)

    progress.progress(100, text="Analysis complete!")

    # ---- Save to History ----
    st.session_state.history.append({
        "filename": filename,
        "verdict": ensemble["verdict"],
        "confidence": ensemble["confidence"],
    })

    # ---- Display Errors (if any) ----
    if detector_errors:
        with st.expander("⚠️ Some detectors encountered errors", expanded=False):
            for err in detector_errors:
                st.warning(err)

    # ---- Verdict Banner ----
    st.divider()
    verdict = ensemble["verdict"]
    confidence = ensemble["confidence"]

    if verdict == "forged":
        st.error(f"🔴 **FORGED** — {confidence:.0%} confidence")
    else:
        st.success(f"🟢 **AUTHENTIC** — {confidence:.0%} confidence")

    st.caption(ensemble["details"])

    # ---- Side-by-Side: Original vs Overlay ----
    st.subheader("📸 Visual Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.image(pil, caption="Original Image", use_container_width=True)

    with col2:
        st.image(overlay_img, caption="Tampering Localization Overlay", use_container_width=True)

    # ---- Detector Details (expandable) ----
    st.subheader("🔬 Detector Results")

    with st.expander("ELA (Error Level Analysis)", expanded=False):
        st.image(ela_img, caption="ELA Output", use_container_width=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mean", f"{ela_stats.get('ela_mean', 0):.2f}")
        col_b.metric("Std Dev", f"{ela_stats.get('ela_std', 0):.2f}")
        col_c.metric("Max", f"{ela_stats.get('ela_max', 0):.2f}")

    with st.expander("Noise Residual (PRNU)", expanded=False):
        if res_map.max() > 0:
            st.image(res_map / res_map.max(), caption="Residual Map", use_container_width=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mean", f"{res_stats.get('res_mean', 0):.4f}")
        col_b.metric("Std Dev", f"{res_stats.get('res_std', 0):.4f}")
        col_c.metric("Max", f"{res_stats.get('res_max', 0):.4f}")

    with st.expander("Copy-Move Detection", expanded=False):
        if cm_mask.sum() > 0:
            cm_overlay = arr.copy()
            cm_overlay[cm_mask == 255] = [255, 0, 0]
            st.image(cm_overlay, caption="Copy-Move Regions", use_container_width=True)
        else:
            st.info("No copy-move regions detected.")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Keypoints", copy_stats.get("n_keypoints", 0))
        col_b.metric("Matches", copy_stats.get("n_good_matches", 0))
        col_c.metric("Clusters", copy_stats.get("n_clusters", 0))

    with st.expander("Localization Statistics", expanded=False):
        st.image(fused_heatmap, caption="Fused Heatmap (pre-threshold)", use_container_width=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Tampered Area", f"{loc_stats.get('tampered_area_pct', 0)}%")
        col_b.metric("Regions", loc_stats.get("num_regions", 0))
        col_c.metric("Largest Region", f"{loc_stats.get('largest_region_pct', 0)}%")

    # ---- Model Transparency ----
    with st.expander("🔧 Models Used", expanded=False):
        st.markdown(f"""
| Model | Type | Result | Confidence |
|-------|------|--------|------------|
| MobileNetV2 (CNN) | Deep Learning | {ensemble.get('cnn_label', 'N/A')} | {ensemble.get('cnn_conf', 0):.0%} |
| Random Forest (M1) | Classical ML | {ensemble.get('rf_label', 'N/A')} | {ensemble.get('rf_conf', 0):.0%} |
| **Ensemble** | **Weighted Avg** | **{verdict.upper()}** | **{confidence:.0%}** |
""")
        st.caption(f"Report generated via: {report_source}")

    # ---- Full Report ----
    st.subheader("📋 Forensic Report")
    with st.expander("View Full Report", expanded=True):
        st.markdown(report_text)

    # ---- Download Buttons ----
    st.subheader("📥 Export Report")
    col_dl1, col_dl2 = st.columns(2)

    images_for_export = {
        "Original Image": pil,
        "ELA Output": ela_img,
        "Localization Overlay": overlay_img,
    }

    with col_dl1:
        md_bytes = export_markdown(report_text, images_for_export)
        st.download_button(
            label="📄 Download Markdown",
            data=md_bytes,
            file_name=f"forensic_report_{filename}.md",
            mime="text/markdown",
        )

    with col_dl2:
        try:
            pdf_bytes = export_pdf(report_text, images_for_export, title=f"Forensic Analysis: {filename}")
            st.download_button(
                label="📕 Download PDF",
                data=pdf_bytes,
                file_name=f"forensic_report_{filename}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(f"PDF export unavailable: {e}")