import os
import sys

# Environment variables to fix OpenMP and library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datetime import datetime
import json

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Skripsi - Deteksi Tabel Gizi",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# MODEL LOADING FUNCTIONS (CACHED)
# ========================================

@st.cache_resource
def load_yolo_model():
    """
    Load YOLOv11 model for nutrition table detection.
    Model path: models/best_yolo.pt
    """
    try:
        model_path = "models/best_yolo.pt"
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Model YOLO tidak ditemukan di: {model_path}")
            st.info("Model YOLO diperlukan untuk deteksi tabel. Silakan letakkan file model di folder 'models/'")
            return None
        
        model = YOLO(model_path)
        st.success("✅ YOLOv11 model loaded successfully")
        return model
    except AttributeError as e:
        if "C3k2" in str(e) or "module" in str(e):
            st.error(f"❌ YOLO Model Version Conflict!")
            st.info("""
            **Solusi**:
            1. Update ultralytics: `pip install --upgrade ultralytics`
            2. Atau re-export model dengan ultralytics versi saat ini
            3. Model saat ini mungkin di-train dengan versi ultralytics yang berbeda
            """)
        else:
            st.error(f"❌ Error loading YOLO model: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {str(e)}")
        return None

@st.cache_resource
def load_paddle_detector():
    """
    Load PaddleOCR for text detection only.
    Uses subprocess to avoid library conflicts with PyTorch/YOLO.
    This allows us to use the custom fine-tuned model in models/det_db_inference
    """
    st.info("📋 PaddleOCR akan dijalankan via subprocess (CPU mode) untuk menghindari library conflict")
    st.success("✅ PaddleOCR (Custom Fine-tuned Model) ready via subprocess")
    
    # Return a marker object that indicates we're using subprocess
    class SubprocessPaddleOCR:
        """Marker class to indicate subprocess mode"""
        def __init__(self):
            self.mode = "subprocess"
    
    return SubprocessPaddleOCR()

@st.cache_resource
def load_trocr_model():
    """
    Load Fine-tuned TrOCR model for text recognition.
    Model: Custom fine-tuned model from models/rec-tr
    Fine-tuned on nutrition label dataset with CER: 0.30
    """
    try:
        # Path to fine-tuned model
        model_path = "models/rec-tr"
        
        # Check if local fine-tuned model exists
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Fine-tuned TrOCR model not found at: {model_path}")
            st.info("Falling back to base model: microsoft/trocr-base-printed")
            model_name = "microsoft/trocr-base-printed"
        else:
            model_name = model_path
            st.info(f"📦 Loading fine-tuned TrOCR model from: {model_path}")
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor - Using trocr-small-printed as requested
        base_model_for_processor = "microsoft/trocr-small-printed"
        processor = TrOCRProcessor.from_pretrained(base_model_for_processor)
        
        # Load fine-tuned model
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        device_msg = "GPU" if device == "cuda" else "CPU"
        model_type = "Fine-tuned (CER: 0.30)" if model_name == model_path else "Base Model"
        st.success(f"✅ TrOCR {model_type} loaded successfully on {device_msg}")
        return processor, model, device
    except Exception as e:
        st.error(f"❌ Error loading TrOCR model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, "cpu"

# ========================================
# PROCESSING FUNCTIONS
# ========================================

def detect_table_yolo(image, yolo_model, confidence_threshold=0.25):
    """
    Step A: Detect nutrition table using YOLOv11
    
    Args:
        image: PIL Image or numpy array
        yolo_model: Loaded YOLO model
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        cropped_table: Cropped image of detected table (PIL Image)
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        confidence: Detection confidence score
    """
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Run YOLO detection
        results = yolo_model(img_array, conf=confidence_threshold)
        
        # Get detection with highest confidence
        if len(results[0].boxes) == 0:
            return None, None, 0.0
        
        # Get the box with highest confidence
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        
        bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        confidence = float(confidences[best_idx])
        
        # Crop the table region
        x1, y1, x2, y2 = bbox
        cropped_table = img_array[y1:y2, x1:x2]
        cropped_table_pil = Image.fromarray(cropped_table)
        
        return cropped_table_pil, bbox, confidence
    
    except Exception as e:
        st.error(f"❌ Error in YOLO detection: {str(e)}")
        return None, None, 0.0

def detect_text_boxes_paddle(cropped_table, paddle_ocr):
    """
    Step B: Detect text bounding boxes using PaddleOCR
    
    Args:
        cropped_table: Cropped table image (PIL Image)
        paddle_ocr: Loaded PaddleOCR detector or SubprocessPaddleOCR marker
    
    Returns:
        text_boxes: List of bounding box coordinates
    """
    try:
        # Check if we're using subprocess mode
        if hasattr(paddle_ocr, 'mode') and paddle_ocr.mode == 'subprocess':
            # Use subprocess to call paddle_detector.py
            import subprocess
            import tempfile
            
            # Save cropped table to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                if isinstance(cropped_table, Image.Image):
                    cropped_table.save(f.name)
                else:
                    Image.fromarray(cropped_table).save(f.name)
                temp_path = f.name
            
            try:
                # Call paddle_detector.py subprocess
                result = subprocess.run(
                    [sys.executable, 'paddle_detector.py', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Debug: Show subprocess output
                if result.returncode != 0:
                    st.error(f"Subprocess failed with return code: {result.returncode}")
                    if result.stderr:
                        st.error(f"Stderr: {result.stderr[:1000]}")
                    return []
                
                # Check if stdout is empty
                if not result.stdout or result.stdout.strip() == "":
                    st.error("PaddleOCR subprocess returned empty output")
                    if result.stderr:
                        st.warning(f"Stderr: {result.stderr[:1000]}")
                    return []
                
                # Parse JSON result
                try:
                    data = json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse JSON from subprocess: {str(e)}")
                    st.error(f"Stdout (first 500 chars): {result.stdout[:500]}")
                    if result.stderr:
                        st.error(f"Stderr: {result.stderr[:1000]}")
                    return []
                
                if data['success']:
                    # Convert boxes to numpy arrays
                    text_boxes = []
                    for bbox in data['boxes']:
                        text_boxes.append(np.array(bbox))
                    return text_boxes
                else:
                    st.error(f"PaddleOCR subprocess error: {data.get('error', 'Unknown error')}")
                    # Show full traceback if available
                    if 'traceback' in data:
                        with st.expander("📋 Full Error Traceback (Click to expand)"):
                            st.code(data['traceback'], language='python')
                    return []
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        else:
            # Original in-process PaddleOCR (if it works)
            # Convert PIL to numpy array
            if isinstance(cropped_table, Image.Image):
                img_array = np.array(cropped_table)
            else:
                img_array = cropped_table
            
            # Run PaddleOCR detection
            result = paddle_ocr.ocr(img_array, cls=False, rec=False)
            
            if result is None or len(result) == 0 or result[0] is None:
                return []
            
            # Extract bounding boxes
            text_boxes = []
            for line in result[0]:
                # line is the bounding box coordinates
                bbox = np.array(line).astype(int)
                text_boxes.append(bbox)
            
            return text_boxes
    
    except Exception as e:
        st.error(f"❌ Error in PaddleOCR detection: {str(e)}")
        return []

def recognize_text_trocr(cropped_text, processor, model, device):
    """
    Step C: Recognize text using TrOCR
    
    Args:
        cropped_text: Cropped text region (PIL Image)
        processor: TrOCR processor
        model: TrOCR model
        device: Device (cpu or cuda)
    
    Returns:
        recognized_text: Recognized text string
    """
    try:
        # Prepare image for TrOCR
        pixel_values = processor(cropped_text, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        
        # Decode the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()
    
    except Exception as e:
        return f"[Error: {str(e)}]"

def draw_bounding_boxes(image, detected_texts):
    """
    Draw bounding boxes with text labels on the image
    
    Args:
        image: PIL Image or numpy array
        detected_texts: List of dictionaries with 'bbox' and 'text' keys
    
    Returns:
        annotated_image: PIL Image with bounding boxes drawn
    """
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Convert to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Draw each bounding box
        for idx, item in enumerate(detected_texts):
            bbox = item['bbox']  # [x1, y1, x2, y2]
            text = item['text']
            
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            color = (0, 255, 0)  # Bright green in BGR
            thickness = 3  # Thicker lines for better visibility
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare text label
            label = f"{idx+1}: {text[:20]}..." if len(text) > 20 else f"{idx+1}: {text}"
            
            # Draw text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw filled rectangle for text background
            cv2.rectangle(img_bgr, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(img_bgr, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        return Image.fromarray(img_rgb)
    
    except Exception as e:
        st.error(f"❌ Error drawing bounding boxes: {str(e)}")
        return image if isinstance(image, Image.Image) else Image.fromarray(image)

def process_image(image_input, yolo_model, paddle_ocr, trocr_processor, trocr_model, device):
    """
    Complete Hybrid Processing Pipeline
    
    Args:
        image_input: Input image (PIL Image)
        yolo_model: Loaded YOLO model
        paddle_ocr: Loaded PaddleOCR detector
        trocr_processor: TrOCR processor
        trocr_model: TrOCR model
        device: Device for TrOCR
    
    Returns:
        results: Dictionary containing:
            - cropped_table: Cropped nutrition table image
            - yolo_confidence: YOLO detection confidence
            - detected_texts: List of detected text boxes and recognized text
            - total_text_boxes: Number of text boxes detected
    """
    results = {
        "cropped_table": None,
        "annotated_table": None,  # New: Image with bounding boxes drawn
        "yolo_confidence": 0.0,
        "detected_texts": [],
        "total_text_boxes": 0,
        "processing_time": 0.0
    }
    
    try:
        start_time = datetime.now()
        
        # Step A: YOLO Detection
        cropped_table, bbox, confidence = detect_table_yolo(image_input, yolo_model)
        
        if cropped_table is None:
            return results
        
        results["cropped_table"] = cropped_table
        results["yolo_confidence"] = confidence
        
        # Step B: PaddleOCR Text Detection
        text_boxes = detect_text_boxes_paddle(cropped_table, paddle_ocr)
        results["total_text_boxes"] = len(text_boxes)
        
        if len(text_boxes) == 0:
            return results
        
        # Step C: TrOCR Text Recognition
        cropped_table_array = np.array(cropped_table)
        detected_texts = []
        
        # Progress bar for text recognition
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        for idx, text_box in enumerate(text_boxes):
            # Update progress text
            progress_text.text(f"Membaca teks {idx + 1}/{len(text_boxes)}...")
            
            # Get bounding box coordinates
            x_coords = text_box[:, 0]
            y_coords = text_box[:, 1]
            x1, y1 = int(x_coords.min()), int(y_coords.min())
            x2, y2 = int(x_coords.max()), int(y_coords.max())
            
            # Crop text region
            text_crop = cropped_table_array[y1:y2, x1:x2]
            
            # Skip if crop is too small
            if text_crop.shape[0] < 5 or text_crop.shape[1] < 5:
                continue
            
            text_crop_pil = Image.fromarray(text_crop)
            
            # Recognize text with TrOCR
            recognized_text = recognize_text_trocr(text_crop_pil, trocr_processor, trocr_model, device)
            
            detected_texts.append({
                "bbox": [x1, y1, x2, y2],
                "text": recognized_text
            })
            
            # Update progress
            progress_bar.progress((idx + 1) / len(text_boxes))
        
        progress_bar.empty()
        progress_text.empty()
        results["detected_texts"] = detected_texts
        
        # Draw bounding boxes on the cropped table
        if len(detected_texts) > 0:
            results["annotated_table"] = draw_bounding_boxes(cropped_table, detected_texts)
        else:
            results["annotated_table"] = cropped_table
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        results["processing_time"] = processing_time
        
        return results
    
    except Exception as e:
        st.error(f"❌ Error in processing pipeline: {str(e)}")
        return results

# ========================================
# CUSTOM CSS - Hide Streamlit Elements
# ========================================
hide_streamlit_style = """
    <style>
    /* Hide Hamburger Menu */
    #MainMenu {visibility: hidden;}
    
    /* Hide Footer */
    footer {visibility: hidden;}
    
    /* Hide Header */
    header {visibility: hidden;}
    
    /* Center align title */
    .main-title {
        text-align: center;
        color: #1E4620;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 3px solid #1E4620;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F0F2F6;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1E4620;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #2D6930;
        border-color: #2D6930;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ========================================
# SIDEBAR NAVIGATION
# ========================================
with st.sidebar:
    # Logo/Header Sidebar
    st.markdown("### 🍎 Deteksi Tabel Gizi")
    st.markdown("---")
    
    # Navigation Menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Detection", "History", "About"],
        icons=["house-fill", "search", "clock-history", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#F0F2F6"},
            "icon": {"color": "#1E4620", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "padding": "10px 15px",
                "--hover-color": "#E8F5E9",
            },
            "nav-link-selected": {
                "background-color": "#1E4620",
                "color": "white",
                "font-weight": "bold",
            },
        },
    )
    
    st.markdown("---")
    st.markdown("**Universitas Sriwijaya**")
    st.markdown("Fakultas Ilmu Komputer")

# ========================================
# MAIN TITLE
# ========================================
st.markdown(
    '<h1 class="main-title">Deteksi Tabel Gizi Pada Kemasan Makanan Menggunakan YOLO dan OCR</h1>',
    unsafe_allow_html=True
)

# ========================================
# PAGE ROUTING
# ========================================

if selected == "Home":
    # HOME PAGE
    st.markdown("## 🏠 Selamat Datang")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Tentang Aplikasi
        
        Aplikasi ini merupakan sistem deteksi dan ekstraksi informasi **Tabel Gizi** 
        pada kemasan makanan menggunakan teknologi:
        
        - **YOLOv11**: Untuk mendeteksi lokasi tabel gizi
        - **PaddleOCR**: Untuk deteksi teks (Text Detection)
        - **TrOCR**: Untuk pengenalan teks (Text Recognition)
        
        ### Fitur Utama
        
        ✅ **Deteksi Otomatis**: Mendeteksi tabel gizi dari foto kemasan makanan  
        ✅ **Ekstraksi Informasi**: Mengekstrak nilai gizi (kalori, protein, lemak, dll)  
        ✅ **Riwayat Deteksi**: Menyimpan hasil deteksi sebelumnya  
        ✅ **User-Friendly**: Interface sederhana dan mudah digunakan  
        """)
    
    with col2:
        st.info("""
        **📌 Cara Penggunaan:**
        
        1. Pilih menu **Detection**
        2. Upload foto kemasan makanan
        3. Klik **Proses Deteksi**
        4. Lihat hasil ekstraksi informasi gizi
        """)
        
        st.success("✨ **Status Sistem**: Ready")

elif selected == "Detection":
    # DETECTION PAGE
    st.markdown("## 🔍 Deteksi Tabel Gizi")
    
    # Load models (cached)
    with st.spinner("Loading AI models..."):
        yolo_model = load_yolo_model()
        paddle_ocr = load_paddle_detector()
        trocr_processor, trocr_model, device = load_trocr_model()
    
    # Check if all models are loaded
    models_ready = all([yolo_model is not None, paddle_ocr is not None, 
                       trocr_processor is not None, trocr_model is not None])
    
    if not models_ready:
        st.error("❌ Tidak semua model berhasil dimuat. Pastikan folder 'models' ada dan berisi file model yang dibutuhkan.")
        st.stop()
    
    st.markdown("---")
    
    # ========================================
    # 2-COLUMN LAYOUT
    # ========================================
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # ========================================
    # KOLOM KIRI - INPUT
    # ========================================
    with col_left:
        st.markdown("### 📤 Input Gambar")
        
        # File Uploader
        uploaded_file = st.file_uploader(
            "Upload Foto Kemasan Makanan",
            type=["jpg", "jpeg", "png"],
            help="Upload gambar kemasan makanan yang memiliki tabel informasi nilai gizi"
        )
        
        # Preview Image
        if uploaded_file is not None:
            st.markdown("#### 🖼️ Preview Gambar")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            # Image Info
            st.info(f"""
            **Nama**: {uploaded_file.name}  
            **Ukuran**: {uploaded_file.size / 1024:.2f} KB  
            **Dimensi**: {image.size[0]} x {image.size[1]} px
            """)
            
            # Detection Button
            st.markdown("---")
            detect_button = st.button("🚀 Mulai Deteksi", type="primary")
        else:
            st.info("👆 Silakan upload gambar untuk memulai deteksi")
            detect_button = False
    
    # ========================================
    # KOLOM KANAN - OUTPUT
    # ========================================
    with col_right:
        st.markdown("### 📊 Hasil Deteksi")
        
        if uploaded_file is None:
            # Placeholder when no image uploaded
            st.info("""
            **Menunggu input gambar...**
            
            Upload gambar di kolom kiri untuk memulai proses deteksi.
            """)
        
        elif detect_button:
            # Process the image with spinner
            with st.spinner("⏳ Sedang memproses: Deteksi Tabel → Deteksi Teks → TrOCR Reading..."):
                results = process_image(
                    image_input=image,
                    yolo_model=yolo_model,
                    paddle_ocr=paddle_ocr,
                    trocr_processor=trocr_processor,
                    trocr_model=trocr_model,
                    device=device
                )
            
            # ========================================
            # DISPLAY RESULTS
            # ========================================
            
            if results["cropped_table"] is not None:
                # 1. Cropped Table Images - Original and Annotated
                st.markdown("#### 🍽️ Tabel Gizi Terdeteksi")
                
                # Two columns: Original and with Bounding Boxes
                img_col1, img_col2 = st.columns(2)
                
                with img_col1:
                    st.markdown("**Original**")
                    st.image(results["cropped_table"], use_column_width=True)
                
                with img_col2:
                    st.markdown("**Dengan Bounding Boxes PaddleOCR**")
                    if results["annotated_table"] is not None:
                        st.image(results["annotated_table"], use_column_width=True)
                    else:
                        st.image(results["cropped_table"], use_column_width=True)
                
                st.caption(f"✅ YOLO Confidence: **{results['yolo_confidence']:.2%}**")
                
                st.markdown("---")
                
                # 2. Statistics
                st.markdown("#### 📈 Statistik Deteksi")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Kotak Teks", results["total_text_boxes"])
                with metric_col2:
                    st.metric("Teks Terbaca", len(results["detected_texts"]))
                with metric_col3:
                    st.metric("Waktu Proses", f"{results['processing_time']:.2f}s")
                
                st.markdown("---")
                
                # 3. Tokens Output - Text detected by TrOCR
                if len(results["detected_texts"]) > 0:
                    st.markdown("#### 📝 Hasil Deteksi Teks (Tokens)")
                    
                    # Extract tokens from detected texts
                    tokens = []
                    for item in results["detected_texts"]:
                        tokens.append(item["text"])
                    
                    # Display tokens
                    st.markdown(f"**List of detected texts** • *{len(tokens)} items*")
                    st.code(json.dumps(tokens, indent=2, ensure_ascii=False), language="json")
                else:
                    st.warning("⚠️ Tidak ada teks yang berhasil dikenali dari tabel")
            else:
                st.error("❌ Gagal mendeteksi tabel gizi pada gambar ini. Pastikan tabel gizi terlihat jelas.")
        
        else:
            # Before button clicked
            st.info("""
            **Siap untuk diproses!**
            
            Klik tombol **"Mulai Deteksi"** di kolom kiri untuk memulai.
            """)


elif selected == "History":
    # HISTORY PAGE
    st.markdown("## 🕐 Riwayat Deteksi")
    
    st.info("📋 Riwayat deteksi akan ditampilkan di sini (Coming Soon)")
    
    # Example table structure
    st.markdown("### Contoh Tabel Riwayat")
    sample_data = {
        "Tanggal": ["2026-01-13", "2026-01-12", "2026-01-11"],
        "Nama Produk": ["Indomie Goreng", "Chitato Rasa Sapi", "Oreo Original"],
        "Kalori": ["370 kkal", "520 kkal", "480 kkal"],
        "Status": ["✅ Sukses", "✅ Sukses", "✅ Sukses"]
    }
    df = pd.DataFrame(sample_data)
    st.dataframe(df, use_column_width=True)

elif selected == "About":
    # ABOUT PAGE
    st.markdown("## ℹ️ Tentang Aplikasi")
    
    st.markdown("---")
    
    # ========================================
    # STUDENT PROFILE CARD
    # ========================================
    st.markdown("### 👨‍🎓 Profil Mahasiswa")
    
    # Profile Card Layout
    profile_col1, profile_col2 = st.columns([1, 2], gap="large")
    
    with profile_col1:
        # Profile Photo
        profile_path = "assets/images/profile.jpg"
        if os.path.exists(profile_path):
            st.image(profile_path, use_column_width=True)
        else:
            # Placeholder if no photo
            st.image("https://via.placeholder.com/400x400/1E4620/FFFFFF?text=Foto+Profil", 
                    use_column_width=True)
            st.caption("⚠️ Upload foto ke: `assets/images/profile.jpg`")
    
    with profile_col2:
        # Custom CSS for profile card
        st.markdown("""
        <style>
        .profile-card {
            background-color: #F8F9FA;
            padding: 2rem;
            border-radius: 10px;
            border-left: 5px solid #1E4620;
            margin-top: 1rem;
        }
        .profile-card h3 {
            color: #1E4620;
            margin-bottom: 0.5rem;
        }
        .profile-card p {
            margin: 0.5rem 0;
            font-size: 1.1rem;
        }
        .profile-label {
            font-weight: bold;
            color: #1E4620;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Profile Details Card
        st.markdown("""
        <div class="profile-card">
            <h3>📋 Data Mahasiswa</h3>
            <p><span class="profile-label">Nama:</span> Ahmad Bintara Mansur</p>
            <p><span class="profile-label">NIM:</span> 0901282227041</p>
            <p><span class="profile-label">Program Studi:</span> TEKNIK INFORMATIKA</p>
            <p><span class="profile-label">Fakultas:</span> Ilmu Komputer</p>
            <p><span class="profile-label">Universitas:</span> Universitas Sriwijaya</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Thesis Title Card
        st.markdown("""
        <div class="profile-card">
            <h2>📖 Judul Skripsi</h2>
            <p style="text-align: justify;">
            <strong>Deteksi Tabel Gizi Pada Kemasan Makanan Menggunakan YOLO dan OCR</strong>
            </p>
            <h3> Dosen Pembimbing 1:</h3>
            <p>Hadipurnawan Satria, M.Sc., Ph.D.</p>
            <h3> Dosen Pembimbing 2:</h3>
            <p>Muhammad Naufal Rachmatullah, M.T.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================
    # APPLICATION INFO
    # ========================================
    st.markdown("### 📱 Tentang Aplikasi")
    
    app_col1, app_col2 = st.columns([2, 1])
    
    with app_col1:
        st.markdown("""
        **Versi**: 1.0.0  
        **Tahun**: 2026
        
        #### 🎯 Tujuan Penelitian
        
        Penelitian ini bertujuan untuk mengembangkan sistem otomatis yang dapat:
        
        1. Mendeteksi lokasi tabel informasi nilai gizi pada kemasan makanan
        2. Mengekstrak informasi gizi secara akurat menggunakan OCR
        3. Membantu konsumen mendapatkan informasi gizi dengan cepat
        
        #### 🛠️ Teknologi yang Digunakan
        
        - **Frontend**: Streamlit
        - **Object Detection**: YOLOv11 (Ultralytics)
        - **Text Detection**: PaddleOCR (Custom Trained)
        - **Text Recognition**: TrOCR (Hugging Face Transformers)
        - **Image Processing**: OpenCV, Pillow
        - **Deep Learning**: PyTorch (CUDA 11.8)
        
        #### 📊 Pendekatan Hybrid Model
        
        Aplikasi ini menggunakan kombinasi 3 model AI:
        1. **YOLOv11** → Deteksi lokasi tabel gizi
        2. **PaddleOCR** → Deteksi kotak-kotak teks
        3. **TrOCR** → Membaca teks dari setiap kotak
        """)
    
    with app_col2:
        # University Logo
        logo_path = "assets/images/unsri_logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)
        else:
            st.info("""
            **🎓 Logo Unsri**
            
            Upload logo ke:
            `assets/images/unsri_logo.png`
            """)
        
        st.markdown("""
        ### 🏛️ Universitas Sriwijaya
        
        **Fakultas Ilmu Komputer**
        
        📍 Jl. Raya Palembang - Prabumulih  
        Km. 32 Indralaya, Ogan Ilir  
        Sumatera Selatan 30662
        
        🌐 [www.unsri.ac.id](https://www.unsri.ac.id)
        """)
        
        st.success("""
        **💡 Tips Penggunaan:**
        
        - Gunakan foto dengan pencahayaan baik
        - Pastikan tabel gizi terlihat jelas
        - Hindari foto blur atau berbayang
        - Gunakan gambar resolusi tinggi
        """)
    
    st.markdown("---")
    
    # ========================================
    # CONTACT & ACKNOWLEDGMENT
    # ========================================
    
    contact_col1, contact_col2 = st.columns(2)
    
    with contact_col1:
        st.markdown("""
        ### 📧 Kontak
        
        Untuk pertanyaan atau saran terkait aplikasi:
        
        - **Email**: [bintaramansur9@gmail.com]
        """)
    
    with contact_col2:
        st.markdown("""
        ### 🙏 Acknowledgment
        
        Terima kasih kepada:
        - Dosen pembimbing atas bimbingannya
        - Fakultas Ilmu Komputer UNSRI
        - Keluarga dan teman-teman
        - OpenAI, Ultralytics, PaddleOCR, Hugging Face
        """)

# ========================================
# FOOTER (Custom)
# ========================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>© 2026 Universitas Sriwijaya - Fakultas Ilmu Komputer</p>",
    unsafe_allow_html=True
)

