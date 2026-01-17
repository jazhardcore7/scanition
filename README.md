# Scanition - Nutrition Label Detection System

> **Deteksi Tabel Gizi Pada Kemasan Makanan Menggunakan YOLO dan OCR**

Web application untuk mendeteksi dan mengekstrak informasi nilai gizi dari foto kemasan makanan secara otomatis menggunakan teknologi Computer Vision dan OCR.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## 📚 About This Project

This is an undergraduate thesis project (Skripsi) from the Informatics Engineering Program, Computer Science Faculty, Sriwijaya University.

**Author:** Ahmad Bintara Mansur (NIM: 0901282227041)  
**Institution:** Universitas Sriwijaya  
**Program:** Teknik Informatika, Fakultas Ilmu Komputer  
**Year:** 2026

---

## 🌟 Features

- ✅ **Automatic Table Detection** - Detects nutrition facts table using YOLOv11
- ✅ **Text Detection** - Locates text regions using custom fine-tuned PaddleOCR
- ✅ **Text Recognition** - Reads Indonesian nutrition labels with fine-tuned TrOCR
- ✅ **Bounding Box Visualization** - Shows detected text regions with green boxes
- ✅ **Real-time Processing** - Interactive web interface built with Streamlit
- ✅ **GPU Acceleration** - CUDA support for faster inference

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Object Detection** | YOLOv11 (Ultralytics) |
| **Text Detection** | PaddleOCR (Custom Fine-tuned) |
| **Text Recognition** | TrOCR (Fine-tuned on Indonesian nutrition labels) |
| **Image Processing** | OpenCV, Pillow |
| **Deep Learning** | PyTorch (CUDA 11.8) |

---

## 📸 Screenshots

![Nutrition Detection Demo](assets/demo_screenshot.png, assets/demo_screenshot2.png, assets/demo_screenshot3.png)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) NVIDIA GPU with CUDA for faster inference

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jazhardcore7/scanition.git
   cd scanition
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download AI Models** (See [Model Setup](#-model-setup) below)

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

---

## 📦 Model Setup

**IMPORTANT:** AI models are too large for GitHub. You need to download/provide them separately.

### Required Models

Create a `models/` folder structure:

```
models/
├── yolo/
│   └── best_yolo.pt              # YOLOv11 model for nutrition table detection
├── paddleocr/
│   └── det_db_inference/
│       ├── inference.pdmodel      # PaddleOCR detection model
│       ├── inference.pdiparams    # PaddleOCR parameters
│       └── inference.yml          # PaddleOCR config
└── trocr/
    └── rec-tr/
        ├── config.json            # TrOCR configuration
        ├── model.safetensors      # Fine-tuned TrOCR weights (~246 MB)
        └── generation_config.json # Generation parameters
```

### Where to Get Models

1. **YOLOv11 Model**
   - Train your own using Ultralytics YOLO on nutrition table dataset
   - Or contact the author for the pre-trained model

2. **PaddleOCR Model**
   - Train using PaddleOCR framework on Indonesian text
   - Or use default PaddleOCR detection model

3. **TrOCR Model**
   - Fine-tuned model available from author
   - Base model: `microsoft/trocr-base-handwritten`
   - Fine-tuned on Indonesian nutrition label dataset

### Model Training (For Developers)

Refer to the thesis document for detailed training procedures:
- YOLOv11 training on nutrition table dataset
- PaddleOCR fine-tuning for Indonesian text
- TrOCR fine-tuning on nutrition label crops

---

## 📁 Project Structure

```
scanition/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── .streamlit/            # Streamlit configuration
│   └── config.toml
├── app/                   # Alternative app structure (optional)
│   ├── main.py
│   ├── config.py
│   └── utils/
│       └── paddle_detector.py
├── assets/                # Static files (images, logos)
│   └── images/
└── models/                # AI models (not in repo - see setup above)
    ├── yolo/
    ├── paddleocr/
    └── trocr/
```

---

## 🎯 How It Works

### Pipeline Architecture

```
Input Image (Nutrition Label Photo)
          ↓
[1] YOLOv11 Object Detection
    → Detects nutrition table location
    → Crops table region (93%+ confidence)
          ↓
[2] PaddleOCR Text Detection
    → Finds text bounding boxes
    → Returns 20-30 text regions
          ↓
[3] TrOCR Text Recognition
    → Reads each text box
    → Outputs Indonesian nutrition terms
          ↓
Final Output: JSON with detected nutrition information
```

### Example Output

```json
{
  "detected_texts": [
    "Energi Total",
    "120 kkal",
    "Protein",
    "5 g",
    "Lemak Total",
    "2 g",
    "Karbohidrat",
    "20 g",
    "Garam",
    "50 mg"
  ]
}
```

---

## 💻 Usage

### Web Interface

1. Navigate to **Detection** page
2. Upload a photo of food packaging with nutrition facts
3. Click **"Mulai Deteksi"** (Start Detection)
4. View results:
   - Original image with detected table
   - Image with green bounding boxes showing text locations
   - Statistics (text boxes count, processing time)
   - Extracted text tokens in JSON format

### Supported Image Formats

- JPG/JPEG
- PNG
- Maximum recommended size: 2000x2000 pixels

---

## 🔧 Configuration

### Streamlit Config

Edit `.streamlit/config.toml` to customize:

```toml
[theme]
primaryColor = "#1E4620"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"

[server]
maxUploadSize = 200
enableCORS = false
```

### Model Paths

Update paths in `main.py` if your model locations differ:

```python
YOLO_MODEL_PATH = "models/yolo/best_yolo.pt"
PADDLE_MODEL_PATH = "models/paddleocr/det_db_inference"
TROCR_MODEL_PATH = "models/trocr/rec-tr"
```

---

## 📊 Performance

### Model Metrics

| Model | Metric | Value |
|-------|--------|-------|
| **YOLOv11** | mAP@0.5 | 95%+ |
| **PaddleOCR** | Detection Accuracy | ~67% |
| **TrOCR** | CER (Character Error Rate) | 0.30 (30%) |

### Processing Time

- **Average**: 10-15 seconds per image (CPU)
- **With GPU**: 5-8 seconds per image

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><b>Model files not found</b></summary>

**Error:** `Model YOLO tidak ditemukan di: models/yolo/best_yolo.pt`

**Solution:** Download required models and place them in correct folders (see [Model Setup](#-model-setup))
</details>

<details>
<summary><b>CUDA not available</b></summary>

**Warning:** Models will run on CPU (slower but functional)

**Solution:** Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
</details>

<details>
<summary><b>PaddleOCR subprocess timeout</b></summary>

**Error:** `Subprocess failed with return code: -1`

**Solution:** Increase timeout in `main.py` line ~195:
```python
timeout=60  # Increase from 30 to 60 seconds
```
</details>

---

## 📄 License

This project is created for academic purposes as part of an undergraduate thesis.

**© 2026 Ahmad Bintara Mansur - Universitas Sriwijaya**

All rights reserved.

---

## 🙏 Acknowledgments

- **Pembimbing 1:** Hadipurnawan Satria, M.Sc., Ph.D.
- **Pembimbing 2:** Muhammad Naufal Rachmatullah, M.T.
- **Universitas Sriwijaya** - Fakultas Ilmu Komputer
- **Ultralytics** - YOLOv11 framework
- **PaddlePaddle** - PaddleOCR
- **Hugging Face** - TrOCR model

---

## 📧 Contact

**Ahmad Bintara Mansur**  
NIM: 0901282227041  
Email: [Your Email]  
Program Studi Teknik Informatika  
Fakultas Ilmu Komputer  
Universitas Sriwijaya

---

