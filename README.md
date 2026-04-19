---
title: Image Forgery Detection
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.30.0"
app_file: app/streamlit_app.py
pinned: false
---

# 🔍 Image Forgery Detection — Milestone 2

AI-powered forensic image analysis system combining deep learning (MobileNetV2) with classical ML (Random Forest) for forgery detection, localization, and structured reporting.

## Features

- **Dual-Model Classification**: MobileNetV2 CNN + Random Forest ensemble
- **Tampering Localization**: Fused heatmap (ELA, PRNU residual, blockiness, copy-move) with Otsu thresholding
- **Structured Reports**: AI-generated forensic reports via Groq API (Llama 3.1)
- **Export**: PDF and Markdown report downloads
- **Analysis History**: Session-based tracking of analyzed images

## Detectors

| Method | What it catches |
|--------|----------------|
| ELA (Error Level Analysis) | JPEG recompression inconsistencies |
| PRNU Residual | Sensor noise pattern anomalies |
| Blockiness Map | Block artifact grid inconsistencies |
| ORB Copy-Move | Duplicated regions via keypoint matching |

## Setup

```bash
pip install -r requirements.txt
python train_cnn.py --data_dir data/train
streamlit run app/streamlit_app.py
```

## Environment Variables

- `GROQ_API_KEY`: (Optional) Groq API key for AI-generated reports. Falls back to template-based reports if not set.
