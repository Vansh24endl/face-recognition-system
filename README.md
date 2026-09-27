# AI Face Intelligence System

An upgraded, full-stack, offline **AI Face Intelligence System** featuring real-time Face Detection, LBPH Face Recognition, Smart Image Dataset Capture, and AI Facial Expression Analysis. Built with Python 3.10+, OpenCV, NumPy, and Streamlit.

---

## 🌟 Upgraded Features

1. **AI Dashboard:** Modern AI vision theme with statistics cards for registered people, dataset samples, LBPH model readiness, camera status, and 7 facial expression classes.
2. **Smart Face Registration:**
   - 5-step registration workflow (Information -> Alignment -> Pose Guidance -> Capture -> Gallery).
   - Real-time image quality evaluation (blur detection, face size check, single face enforcement).
   - Pose guidance instructions (*"Look straight"*, *"Tilt head left/right"*, *"Tilt chin up/down"*).
   - Sample gallery preview upon completing 30 image captures.
3. **Model Training:** One-click training pipeline for OpenCV `LBPHFaceRecognizer` producing `trainer/trainer.yml` and `trainer/labels.json`.
4. **Dual Live Recognition & Expression Stream:**
   - Dual-pipeline inference running LBPH Face Recognition and AI Facial Expression Analysis in parallel.
   - Dual-bounding box overlays displaying `Name`, `Recognition %`, `Predicted Expression`, and `Exp Conf %`.
   - Real-time facial expression probability panel with progress bars.
   - Session-based expression log history table.
5. **Facial Expression Analysis Lab:** Dedicated lab to upload images or take snapshots for expression probability breakdowns across 7 classes (*Happy, Sad, Neutral, Angry, Surprise, Fear, Disgust*).
6. **Registered People Directory:** Summary table, dataset metadata audit, and secure deletion capability.
7. **100% Offline & Private:** Operates entirely locally on CPU with zero cloud API dependencies.

---

## 📁 Project Structure

```text
face-recognition-system/
│
├── app.py                  # Streamlit AI Dashboard & Multi-Module App
├── requirements.txt        # Project Dependencies
├── README.md               # Setup & User Documentation
├── PRACTICAL.md            # Detailed College Lab Report & Viva Q&A
├── setup.bat               # One-click Windows Setup Script
├── run.bat                 # One-click Execution Script
│
├── dataset/                # Face Sample Directories (dataset/PersonName_ID/)
│
├── trainer/                # Model Artifacts
│   ├── trainer.yml         # LBPH Classifier Weights
│   └── labels.json         # Label Index Metadata
│
├── models/
│   └── emotion-ferplus.onnx # Pretrained FERPlus ONNX Expression Model
│
├── utils/                  # Modular Architecture Utilities
│   ├── __init__.py
│   ├── face_detection.py   # Haar Cascade Classifier Loader
│   ├── face_capture.py     # Smart Capture & Quality Evaluation
│   ├── trainer.py          # LBPH Model Trainer
│   ├── recognizer.py       # Dual-Pipeline Real-Time Inference
│   └── expression_analyzer.py # Deep Neural Network FER Analyzer
│
└── data/
    └── haarcascade_frontalface_default.xml # Face Detector Model
```

---

## ⚙️ Installation & Execution (Windows)

### Fast Start:

Run in PowerShell:
```powershell
.\run.bat
```

Or run directly via python:
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Application URL: `http://localhost:8501`

---

## 🚀 How to Use & Demonstrate

1. **Dashboard:** View system overview and module health.
2. **Register Face:** Enter `Full Name` and `Person ID`. Follow live pose guidance to capture 30 high-quality face samples.
3. **Train Model:** Click **🚀 Train LBPH Classifier** to build model weights.
4. **Live Recognition:** Enable camera stream to observe real-time face recognition and facial expression predictions in parallel.
5. **Expression Analysis:** Test images or live snapshots to view probability bars for all 7 expression categories.
