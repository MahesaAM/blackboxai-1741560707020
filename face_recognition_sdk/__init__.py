import os
import logging
from typing import Optional, List, Tuple, Union
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRecognitionSDK:
    """
    Main SDK class that integrates face detection, recognition, and liveness detection.
    """
    
    def __init__(self):
        """Initialize the SDK components."""
        self.face_detector = None
        self.face_recognizer = None
        self.liveness_detector = None
        self.models_loaded = False
        self.database_loaded = False
        logger.info("FaceRecognitionSDK initialized")

    def load_models(self, model_path: str) -> bool:
        """
        Load all required models from the specified path.
        
        Args:
            model_path (str): Path to the directory containing model files
            
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path {model_path} does not exist")

            # Import modules here to avoid circular imports
            from .modules.face_detector import FaceDetector
            from .modules.face_recognizer import FaceRecognizer
            from .modules.liveness_detector import LivenessDetector

            # Initialize components with their respective models
            self.face_detector = FaceDetector(
                model_path=os.path.join(model_path, "face_detection_model.xml")
            )
            
            self.face_recognizer = FaceRecognizer(
                model_path=os.path.join(model_path, "shape_predictor_68_face_landmarks.dat")
            )
            
            self.liveness_detector = LivenessDetector(
                landmark_model_path=os.path.join(model_path, "shape_predictor_68_face_landmarks.dat")
            )

            self.models_loaded = True
            logger.info("All models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def load_face_database(self, database_path: str) -> bool:
        """
        Load the face database for recognition.
        
        Args:
            database_path (str): Path to the directory containing known face images
            
        Returns:
            bool: True if database loaded successfully, False otherwise
        """
        try:
            if not self.models_loaded:
                raise RuntimeError("Models must be loaded before loading face database")
                
            if not os.path.exists(database_path):
                raise FileNotFoundError(f"Database path {database_path} does not exist")

            success = self.face_recognizer.load_face_database(database_path)
            if success:
                self.database_loaded = True
                logger.info("Face database loaded successfully")
            return success

        except Exception as e:
            logger.error(f"Error loading face database: {str(e)}")
            return False

    def start_face_detection(self, camera_id: int = 0) -> None:
        """
        Start real-time face detection from webcam.
        
        Args:
            camera_id (int): ID of the camera device to use
        """
        try:
            if not self.models_loaded:
                raise RuntimeError("Models must be loaded before starting detection")

            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera with ID {camera_id}")

            logger.info("Starting real-time face detection")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                # Process each detected face
                for face_coords in faces:
                    # Draw bounding box
                    x, y, w, h = face_coords
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Check liveness if requested
                    if self.check_liveness(frame[y:y+h, x:x+w]):
                        cv2.putText(frame, "Live Face", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Perform recognition if database is loaded
                    if self.database_loaded:
                        name = self.recognize_face(frame[y:y+h, x:x+w])
                        if name:
                            cv2.putText(frame, name, (x, y+h+25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow('Face Recognition SDK Demo', frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def check_liveness(self, face_image: np.ndarray) -> bool:
        """
        Check if a detected face is live.
        
        Args:
            face_image (np.ndarray): Image containing the face to check
            
        Returns:
            bool: True if the face is determined to be live, False otherwise
        """
        try:
            if not self.models_loaded:
                raise RuntimeError("Models must be loaded before checking liveness")

            return self.liveness_detector.check_liveness(face_image)

        except Exception as e:
            logger.error(f"Error checking liveness: {str(e)}")
            return False

    def recognize_face(self, face_image: np.ndarray) -> Optional[str]:
        """
        Recognize a face from the loaded database.
        
        Args:
            face_image (np.ndarray): Image containing the face to recognize
            
        Returns:
            Optional[str]: Name of the recognized person, or None if not recognized
        """
        try:
            if not self.models_loaded:
                raise RuntimeError("Models must be loaded before recognition")
            if not self.database_loaded:
                raise RuntimeError("Face database must be loaded before recognition")

            return self.face_recognizer.recognize_face(face_image)

        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return None

    def __del__(self):
        """Cleanup resources when the SDK instance is destroyed."""
        cv2.destroyAllWindows()
        logger.info("FaceRecognitionSDK resources cleaned up")
