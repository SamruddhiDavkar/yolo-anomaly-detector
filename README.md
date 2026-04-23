# 🔍 YOLO Real-Time Anomaly Detector

> Real-Time AI Systems — Jetson Nano Project  
> YOLOv8 · OpenCV · ipywidgets · JupyterLab

---

## 📌 Project Overview

This project implements a **real-time object anomaly detection system** using YOLOv8 running on a **NVIDIA Jetson Nano**. The system captures live webcam frames, runs inference with a custom confidence threshold, and classifies each detection as either **NORMAL** or an **ANOMALY** — displaying results live inside a JupyterLab notebook UI.

---

## 🎯 Features

- ✅ Real-time webcam detection via YOLOv8n
- ✅ Anomaly vs. Normal classification with visual alert banners
- ✅ Live bounding boxes with class labels and confidence scores
- ✅ Interactive ipywidgets UI (Start / Stop / Snapshot / Log)
- ✅ Adjustable confidence threshold slider
- ✅ Frame counter, detection count, anomaly count, FPS display
- ✅ Snapshot save button for capturing detection frames
- ✅ Runs on Jetson Nano (JetPack 4.6 / Python 3.8)

---

## 🖥️ Demo Screenshots

### ⚠️ Anomaly Detected
> The system flagged an anomalous frame — displayed in red with "ANOMALY DETECTED!" overlay.

![Anomaly Detected](screenshots/anomaly_detected.png)

---

### ✅ Normal Detection (Bottle)
> YOLO correctly identified a bottle with green bounding boxes — classified as NORMAL.

![Normal Detection](screenshots/detected.png)

---

## 🏗️ Architecture

```
Webcam Input (cv2.VideoCapture)
        ↓
  YOLOv8n Inference
        ↓
  Detected Labels List
        ↓
  Anomaly Check (label ∈ ANOMALY_CLASSES?)
   ┌────────────┬────────────────┐
  YES           NO
   ↓             ↓
Red Banner    Green Banner
"ANOMALY      "X DETECTED
DETECTED!"     (NORMAL)"
        ↓
  ipywidgets Image widget
  (frame displayed in Jupyter)
```

---

## 📁 Project Structure

```
anomaly_detector/
├── anomaly_detector.ipynb   # Main notebook with UI
├── anomaly.py               # Standalone script version
├── yolov8n.pt               # YOLOv8 nano weights
├── screenshots/
│   ├── anomaly_detected.png
│   └── detected.png
└── README.md
```

---

## ⚙️ Setup & Installation

### Requirements

```bash
pip install ultralytics ipywidgets opencv-python-headless
```

### On Jetson Nano (JetPack 4.6)

```bash
# Use the Jetson-compatible torch build
pip install ultralytics --no-deps
pip install ipywidgets
```

### Run

```bash
jupyter lab
# Open anomaly_detector.ipynb
# Run all cells → Use the UI buttons
```

---

## 🔧 Configuration

In **Cell 3**, edit these variables:

```python
MODEL_PATH = "yolov8n.pt"              # path to your model weights
ANOMALY_CLASSES = ["knife", "scissors"] # what counts as an anomaly
CONF_THRESHOLD = 0.40                  # default confidence (also adjustable via slider)
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Model | YOLOv8n |
| Device | NVIDIA Jetson Nano 4GB |
| Avg FPS (Jetson) | ~4–6 FPS |
| Avg FPS (Laptop) | ~18–24 FPS |
| Confidence Threshold | 0.40 |
| Anomaly trigger | Label match in defined class list |

---

## 📸 System Output

The notebook displays two detection states:

| State | Banner Color | Label |
|---|---|---|
| Anomaly detected | 🔴 Red | `ANOMALY DETECTED!` |
| Normal object detected | 🟢 Green | `[CLASS] DETECTED (NORMAL)` |
| No detection | 🟢 Green | `No detection` |

---

## 👩‍💻 Author

**Samruddhi Davkar** — M.S. Data Science, UMBC  
Real-Time AI Systems | Spring 2025  
GitHub: [your-github-username]

---

## 📄 License

MIT License
