# AI Deepfake Interaction System
> **CENG 384 · Digital Signal Processing · Course Project**  
> Group Project by Yusuf Yılmaz, Alperen Enes Yaman, Yiğit Burak Çetin, Hani Saleh Ali Saad Al-Shalal, & Mustafa Özkürkcü.

---

An advanced, interactive deepfake synthesis and evaluation platform. This system implements state-of-the-art AI and Digital Signal Processing (DSP) pipelines for real-time face swapping, high-fidelity voice cloning, text-to-speech prosody modification, and multi-participant virtual meeting simulations.

---

## 🌟 Key Module Features

### 🎙 1. High-Resolution Voice Cloning & Synthesis
*   **Coqui XTTS v2 Integration**: One-shot voice cloning from a 3–10 second speaker reference.
*   **Speech-Optimized Phase Vocoder**: High-time-resolution STFT window framing (`n_fft=512`, `hop_length=128`) optimized for human speech dynamics ($\approx21\text{ms}$ frames) to eliminate metallic echoes and bubbling artifacts.
*   **Pure DSP Emotional Prosody Engine**: Clean emotional shaping (Happy, Sad, Angry, Whispering) using custom speed-stretching, gain profiles, pre/de-emphasis filters, and hyperbolic tangent soft-clipping saturation.

### 🎭 2. Ultra-Realistic Face Swap & Post-Processing
*   **buffalo_l & Inswapper 128**: Dense 106-landmark topological analysis for face alignment.
*   **Interactive Delaunay Mesh Visualization**: Live triangulation overlay demonstrating face tracking coverage.
*   **Strict Anatomical Hulls**: Constrains face swaps strictly to the facial features, eliminating hair, ears, and background bleeding.
*   **Pyramid Blending**: Multi-band frequency-domain blending using a 5-level Laplacian pyramid to seamlessly merge skin textures.
*   **GFPGAN ONNX Face Restoration**: Integrated HD face restoration for crisp high-resolution outputs.

### 💬 3. Multi-Tier Conversational AI personality
*   **Dynamic API Cascade**: Automatic API fallback chain from Gemini 2.5 Flash -> Groq LLaMA3 -> Offline 100x Keyword personality matcher.

### 📊 4. Academic Quality & Safety Evaluation
*   **Quantitative Metrics**: Dynamic calculation of SSIM, PSNR (image quality), MCD (Mel-Cepstral Distortion), and SNR (signal-to-noise ratio).
*   **Visual & Audio Watermarking**: Mandatory visual disclosure stamping and near-ultrasonic sine watermarking ($95\%$ Nyquist frequency limit) for ethics compliance.
*   **Persistent Metrics Logger**: Auto-saves every run's quality metrics into `session_metrics.csv` for student download.

---

## 🚀 Quickstart Guide

### Prerequisites
*   Python 3.11
*   Microsoft Visual C++ Build Tools (required for some packages)

### 1. Install Dependencies
Initialize your virtual environment and install standard requirements:
```bash
python -m venv venv311
.\venv311\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Model Weights
Run the one-shot model downloader to fetch GFPGAN and facial models:
```bash
python download_models.py
```

### 3. Launch Server
Start the FastAPI server:
```bash
.\start_server.bat
```
Once active, visit **[http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/)** in your browser to experience the premium interactive UI.

---

## 🛠 Directory Layout
```text
deepfake_interaction_system/
├── backend/
│   ├── core/                  # DSP watermarking, evaluation, conversation logic
│   ├── dsp_models/            # Face swapper, XTTS voice synthesizer, model wrappers
│   └── main.py                # FastAPI Server routes
├── frontend/                  # Glassmorphic HTML5 / Vanilla JS UI client
├── test_assets/               # Auto-generated verification images & audio
├── requirements.txt           # Standard Python package dependencies
├── system_health_audit.md     # Full architectural validation log
└── README.md                  # Project overview and quickstart
```

---

## 🎓 Academic DSP Context & Formulas
This project acts as an educational showcase for time-domain and frequency-domain digital signal manipulation:
*   **Mel-Cepstral Distortion**: Uses Mel-frequency cepstral coefficients (MFCCs) to calculate acoustic spectral distance between original and synthesized envelopes:
    $$\text{MCD} = \frac{10}{\ln 10} \sqrt{2 \sum_{i=1}^{13} (c_i - c_i')^2}$$
*   **Laplacian Pyramids**: Implements frequency band division where low-frequency skin tones and high-frequency hair/pore detail are processed separately:
    $$L_i = G_i - \text{Expand}(G_{i+1})$$
