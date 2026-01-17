"""
Configuration file untuk path dan settings aplikasi
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent  # d:/Skripsi/project/web/

# Model paths
MODEL_DIR = BASE_DIR / "models"
MODEL_YOLO_PATH = MODEL_DIR / "yolo" / "best_yolo.pt"
MODEL_PADDLE_DET_PATH = MODEL_DIR / "paddleocr" / "det_db_inference"
MODEL_TROCR_PATH = MODEL_DIR / "trocr" / "rec-tr"

# Asset paths
ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
PROFILE_IMAGE_PATH = IMAGES_DIR / "profile.jpg"
LOGO_IMAGE_PATH = IMAGES_DIR / "unsri_logo.png"

# Utils paths
UTILS_DIR = Path(__file__).parent / "utils"
PADDLE_DETECTOR_SCRIPT = UTILS_DIR / "paddle_detector.py"

# Convert to string for compatibility
MODEL_YOLO_PATH = str(MODEL_YOLO_PATH)
MODEL_PADDLE_DET_PATH = str(MODEL_PADDLE_DET_PATH)
MODEL_TROCR_PATH = str(MODEL_TROCR_PATH)
PROFILE_IMAGE_PATH = str(PROFILE_IMAGE_PATH)
LOGO_IMAGE_PATH = str(LOGO_IMAGE_PATH)
PADDLE_DETECTOR_SCRIPT = str(PADDLE_DETECTOR_SCRIPT)
