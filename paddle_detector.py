"""
Standalone PaddleOCR Detection Script - SIMPLIFIED VERSION

This script runs PaddleOCR separately to avoid conflicts with PyTorch.
Uses new fine-tuned detection model from models/paddleocr/det-new/best_model
"""

import sys
import json
import os

# Suppress ALL warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
from PIL import Image

def detect_text_boxes(image_path):
    """
    Detect text boxes using PaddleOCR
    
    Args:
        image_path: Path to the image file
        
    Returns:
        JSON string with detected text boxes
    """
    try:
        from paddleocr import PaddleOCR
        
        # Force CPU mode to avoid CUDA DLL issues
        use_gpu = False
        
        # Initialize PaddleOCR with new fine-tuned model
        # Priority: det-new/best_model (new fine-tuned) > det_db_inference (old)
        det_model_path_new = "models/paddleocr/det-new/best_model"
        det_model_path_old = "models/paddleocr/det_db_inference"
        
        # Check which model exists
        if os.path.exists(det_model_path_new):
            det_model_path = det_model_path_new
        elif os.path.exists(det_model_path_old):
            det_model_path = det_model_path_old
        else:
            det_model_path = None
        
        ocr = PaddleOCR(
            use_angle_cls=False,
            rec=False,  # Only detection
            det_model_dir=det_model_path,
            use_gpu=use_gpu,
            lang='en',
            show_log=False
        )
        
        # Load image
        img_array = np.array(Image.open(image_path))
        
        # Run detection
        result = ocr.ocr(img_array, cls=False, rec=False)
        
        # Extract bounding boxes - SIMPLIFIED, no validation
        text_boxes = []
        
        # If result exists and has content, extract boxes
        if result:
            try:
                for item in result:
                    if item:
                        for box in item:
                            # Convert to list safely
                            box_list = []
                            for point in box:
                                if hasattr(point, 'tolist'):
                                    box_list.append(point.tolist())
                                elif isinstance(point, (list, tuple)):
                                    box_list.append(list(point))
                                else:
                                    box_list.append([int(point[0]), int(point[1])])
                            text_boxes.append(box_list)
            except:
                pass  # If extraction fails, return empty
        
        return json.dumps({"success": True, "boxes": text_boxes})
    
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No image path provided"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = detect_text_boxes(image_path)
    print(result)
