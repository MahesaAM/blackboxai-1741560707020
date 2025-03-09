import logging
from typing import Optional, Tuple, Union
import cv2
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Preprocess an image for face detection and recognition.
    
    Args:
        image (np.ndarray): Input image
        target_size (Optional[Tuple[int, int]]): Target size for resizing (width, height)
        normalize (bool): Whether to normalize pixel values to [0,1]
    
    Returns:
        np.ndarray: Preprocessed image
    """
    try:
        # Input validation
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        # Make a copy to avoid modifying the original
        processed = image.copy()
        
        # Resize if target size is specified
        if target_size is not None:
            if not isinstance(target_size, tuple) or len(target_size) != 2:
                raise ValueError("target_size must be a tuple of (width, height)")
            processed = cv2.resize(processed, target_size)
            
        # Normalize if requested
        if normalize:
            processed = processed.astype(np.float32) / 255.0
            
        return processed

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale if it's not already.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Grayscale image
    """
    try:
        if image is None:
            raise ValueError("Invalid input image")
            
        if len(image.shape) == 2:
            return image
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    except Exception as e:
        logger.error(f"Error converting to grayscale: {str(e)}")
        raise

def normalize_image(
    image: np.ndarray,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> np.ndarray:
    """
    Normalize image pixel values to a specified range.
    
    Args:
        image (np.ndarray): Input image
        min_value (float): Minimum value after normalization
        max_value (float): Maximum value after normalization
        
    Returns:
        np.ndarray: Normalized image
    """
    try:
        if image is None:
            raise ValueError("Invalid input image")
            
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max - image_min == 0:
            return np.full_like(image, min_value)
            
        normalized = (image - image_min) / (image_max - image_min)
        normalized = normalized * (max_value - min_value) + min_value
        
        return normalized

    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        raise

def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve image contrast.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Histogram equalized image
    """
    try:
        if image is None:
            raise ValueError("Invalid input image")
            
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        return cv2.equalizeHist(gray)

    except Exception as e:
        logger.error(f"Error equalizing histogram: {str(e)}")
        raise

def detect_and_align_face(
    image: np.ndarray,
    landmarks: np.ndarray
) -> Optional[np.ndarray]:
    """
    Align face to a standard position using facial landmarks.
    
    Args:
        image (np.ndarray): Input image
        landmarks (np.ndarray): Array of facial landmarks
        
    Returns:
        Optional[np.ndarray]: Aligned face image or None if alignment fails
    """
    try:
        if image is None or landmarks is None:
            raise ValueError("Invalid input image or landmarks")
            
        # Get eye centers
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        
        # Calculate angle to align eyes horizontally
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get center of rotation (midpoint between eyes)
        eye_center = ((left_eye_center + right_eye_center) * 0.5).astype(np.int32)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            tuple(eye_center),
            angle,
            1.0
        )
        
        # Perform the rotation
        aligned = cv2.warpAffine(
            image,
            rotation_matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        return aligned

    except Exception as e:
        logger.error(f"Error aligning face: {str(e)}")
        return None

def enhance_image_quality(
    image: np.ndarray,
    denoise: bool = True,
    sharpen: bool = True
) -> np.ndarray:
    """
    Enhance image quality for better face detection and recognition.
    
    Args:
        image (np.ndarray): Input image
        denoise (bool): Whether to apply denoising
        sharpen (bool): Whether to apply sharpening
        
    Returns:
        np.ndarray: Enhanced image
    """
    try:
        if image is None:
            raise ValueError("Invalid input image")
            
        enhanced = image.copy()
        
        # Apply denoising if requested
        if denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced,
                None,
                10,
                10,
                7,
                21
            )
            
        # Apply sharpening if requested
        if sharpen:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
        return enhanced

    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        raise

def crop_face(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.2
) -> Optional[np.ndarray]:
    """
    Crop a face region from an image with optional margin.
    
    Args:
        image (np.ndarray): Input image
        bbox (Tuple[int, int, int, int]): Bounding box coordinates (x, y, w, h)
        margin (float): Margin to add around the face as a fraction of face size
        
    Returns:
        Optional[np.ndarray]: Cropped face image or None if crop fails
    """
    try:
        if image is None or not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError("Invalid input image or bounding box")
            
        x, y, w, h = bbox
        
        # Calculate margins
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Calculate crop coordinates with margins
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # Crop the face region
        face_crop = image[y1:y2, x1:x2]
        
        return face_crop

    except Exception as e:
        logger.error(f"Error cropping face: {str(e)}")
        return None

def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0
) -> np.ndarray:
    """
    Adjust image brightness and contrast.
    
    Args:
        image (np.ndarray): Input image
        brightness (float): Brightness factor (>1 increases, <1 decreases)
        contrast (float): Contrast factor (>1 increases, <1 decreases)
        
    Returns:
        np.ndarray: Adjusted image
    """
    try:
        if image is None:
            raise ValueError("Invalid input image")
            
        # Convert to float for calculations
        adjusted = image.astype(np.float32)
        
        # Apply brightness
        adjusted *= brightness
        
        # Apply contrast
        adjusted = (adjusted - 128) * contrast + 128
        
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)

    except Exception as e:
        logger.error(f"Error adjusting brightness/contrast: {str(e)}")
        raise
