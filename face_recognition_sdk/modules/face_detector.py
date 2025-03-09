import os
import logging
from typing import List, Tuple, Optional
import cv2
import dlib
import numpy as np
from ..utils.image_processing import preprocess_image

# Configure logging
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Class for detecting faces in images using both OpenCV and dlib.
    Provides options for using either Haar Cascades (OpenCV) or HOG+SVM (dlib).
    """

    def __init__(self, model_path: str, detection_method: str = 'dlib'):
        """
        Initialize the face detector.

        Args:
            model_path (str): Path to the face detection model file
            detection_method (str): Method to use for face detection ('dlib' or 'opencv')
        """
        self.detection_method = detection_method.lower()
        
        try:
            if self.detection_method == 'dlib':
                # Initialize dlib's HOG face detector
                self.detector = dlib.get_frontal_face_detector()
                logger.info("Initialized dlib HOG face detector")
                
            elif self.detection_method == 'opencv':
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"OpenCV cascade file not found at {model_path}")
                    
                # Initialize OpenCV's Haar Cascade classifier
                self.detector = cv2.CascadeClassifier(model_path)
                if self.detector.empty():
                    raise RuntimeError("Error loading OpenCV cascade classifier")
                logger.info("Initialized OpenCV Haar Cascade face detector")
                
            else:
                raise ValueError("Detection method must be either 'dlib' or 'opencv'")

        except Exception as e:
            logger.error(f"Error initializing face detector: {str(e)}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the input image.

        Args:
            image (np.ndarray): Input image (BGR format for OpenCV)

        Returns:
            List[Tuple[int, int, int, int]]: List of face coordinates (x, y, width, height)
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            # Preprocess image
            processed_image = preprocess_image(image)
            
            if self.detection_method == 'dlib':
                return self._detect_faces_dlib(processed_image)
            else:
                return self._detect_faces_opencv(processed_image)

        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []

    def _detect_faces_dlib(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using dlib's HOG face detector.

        Args:
            image (np.ndarray): Preprocessed input image

        Returns:
            List[Tuple[int, int, int, int]]: List of face coordinates
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Detect faces
            faces = self.detector(gray)
            
            # Convert dlib rectangles to (x, y, w, h) format
            face_coords = []
            for face in faces:
                x = face.left()
                y = face.top()
                w = face.right() - face.left()
                h = face.bottom() - face.top()
                face_coords.append((x, y, w, h))

            logger.debug(f"Detected {len(face_coords)} faces using dlib")
            return face_coords

        except Exception as e:
            logger.error(f"Error in dlib face detection: {str(e)}")
            return []

    def _detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV's Haar Cascade classifier.

        Args:
            image (np.ndarray): Preprocessed input image

        Returns:
            List[Tuple[int, int, int, int]]: List of face coordinates
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Convert to list of tuples
            face_coords = [(x, y, w, h) for (x, y, w, h) in faces]
            
            logger.debug(f"Detected {len(face_coords)} faces using OpenCV")
            return face_coords

        except Exception as e:
            logger.error(f"Error in OpenCV face detection: {str(e)}")
            return []

    def get_face_crop(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop a detected face from the image.

        Args:
            image (np.ndarray): Input image
            face_coords (Tuple[int, int, int, int]): Face coordinates (x, y, w, h)

        Returns:
            Optional[np.ndarray]: Cropped face image or None if invalid
        """
        try:
            x, y, w, h = face_coords
            
            # Validate coordinates
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError("Invalid face coordinates")
            if x + w > image.shape[1] or y + h > image.shape[0]:
                raise ValueError("Face coordinates outside image boundaries")

            # Crop and return face region
            return image[y:y+h, x:x+w]

        except Exception as e:
            logger.error(f"Error cropping face: {str(e)}")
            return None

    def detect_and_crop_largest_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop the largest face in the image.

        Args:
            image (np.ndarray): Input image

        Returns:
            Optional[np.ndarray]: Cropped image of the largest face, or None if no face detected
        """
        try:
            # Detect all faces
            faces = self.detect_faces(image)
            
            if not faces:
                return None

            # Find the largest face by area
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Crop and return the largest face
            return self.get_face_crop(image, largest_face)

        except Exception as e:
            logger.error(f"Error detecting and cropping largest face: {str(e)}")
            return None
