"""
UTILS.PY - COMPLETE WORKING VERSION
Fixed: Added missing validate_image function
"""

import logging
from typing import Optional, Tuple
import numpy as np
import cv2
import io
from PIL import Image

logger = logging.getLogger(__name__)


def setup_logger(name: str = __name__, level: int = logging.INFO):
    """Setup logger"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)


def validate_image_for_frontend(file_contents: bytes) -> Tuple[Optional[np.ndarray], str]:
    """
    SMART image validator that detects format and converts to BGR if needed
    
    Returns:
        (image_array, source_format) where source_format is 'bgr' or 'rgb'
    """
    try:
        # Try 1: Check if it's a PNG/JPEG from frontend
        try:
            # Frontend images are usually RGB
            pil_image = Image.open(io.BytesIO(file_contents))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL RGB to numpy RGB
            image_rgb = np.array(pil_image)
            
            # Convert RGB to BGR (for model compatibility)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            logger.info(f"✓ Frontend image: {image_bgr.shape}, converted RGB→BGR")
            return image_bgr, 'rgb'
            
        except:
            # Try 2: Direct cv2 decode (for BGR images)
            nparr = np.frombuffer(file_contents, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image_bgr is not None:
                logger.info(f"✓ Direct BGR image: {image_bgr.shape}")
                return image_bgr, 'bgr'
            
            # Try 3: Fallback - any image format
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if image is not None:
                # Handle different channel counts
                if len(image.shape) == 2:  # Grayscale
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:  # RGBA
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                elif image.shape[2] == 3:  # Assume RGB
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image
                
                logger.info(f"✓ Converted image to BGR: {image_bgr.shape}")
                return image_bgr, 'converted'
            
            raise ValueError("Could not decode image")
            
    except Exception as e:
        logger.error(f"❌ Image validation failed: {str(e)}")
        return None, 'error'


def validate_image(file_contents: bytes) -> Optional[np.ndarray]:
    """
    Main validate_image function - wrapper for validate_image_for_frontend
    This is the function that main.py expects
    """
    image, format_type = validate_image_for_frontend(file_contents)
    if image is not None:
        logger.info(f"Image validated: {image.shape}, format: {format_type}")
    return image


def force_bgr_conversion(file_contents: bytes) -> Optional[np.ndarray]:
    """Force all images to BGR format - for Colab consistency"""
    nparr = np.frombuffer(file_contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is not None:
        # Force consistent BGR format
        # First ensure it's 3 channels
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        logger.info(f"Forced BGR conversion: {img.shape}")
        return img
    
    return None


def create_heatmap(probs: np.ndarray, shape: tuple) -> np.ndarray:
    """Create heatmap visualization"""
    heatmap = cv2.resize(probs, (shape[1], shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap