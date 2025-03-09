import logging
from typing import List, Tuple, Optional, Dict
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from ..utils.image_processing import preprocess_image

# Configure logging
logger = logging.getLogger(__name__)

class LivenessDetector:
    """
    Class for detecting liveness using multiple methods:
    1. Eye blink detection using Eye Aspect Ratio (EAR)
    2. Head pose estimation
    3. Texture analysis for anti-spoofing
    """

    def __init__(
        self,
        landmark_model_path: str,
        ear_threshold: float = 0.3,
        ear_consec_frames: int = 2,
        pose_threshold: float = 30.0
    ):
        """
        Initialize the liveness detector.

        Args:
            landmark_model_path (str): Path to dlib's facial landmark predictor
            ear_threshold (float): Threshold for eye aspect ratio to indicate blink
            ear_consec_frames (int): Number of consecutive frames for valid blink
            pose_threshold (float): Maximum allowed head pose angle change
        """
        self.ear_threshold = ear_threshold
        self.ear_consec_frames = ear_consec_frames
        self.pose_threshold = pose_threshold
        
        # Initialize counters and flags
        self.blink_counter = 0
        self.blink_total = 0
        self.previous_pose = None
        
        try:
            # Initialize facial landmark predictor
            self.predictor = dlib.shape_predictor(landmark_model_path)
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Define facial landmarks indices
            self.LEFT_EYE_START = 36
            self.LEFT_EYE_END = 42
            self.RIGHT_EYE_START = 42
            self.RIGHT_EYE_END = 48
            
            # 3D model points for head pose estimation
            self.model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])
            
            logger.info("Liveness detector initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing liveness detector: {str(e)}")
            raise

    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect facial landmarks in the image.

        Args:
            image (np.ndarray): Input image

        Returns:
            Optional[np.ndarray]: Array of facial landmarks or None if detection fails
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector(gray)
            if not faces:
                return None
                
            # Get facial landmarks for the first face
            landmarks = self.predictor(gray, faces[0])
            
            # Convert landmarks to numpy array
            landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            return landmarks_array

        except Exception as e:
            logger.error(f"Error detecting facial landmarks: {str(e)}")
            return None

    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate the eye aspect ratio (EAR).

        Args:
            eye_landmarks (np.ndarray): Array of eye landmark coordinates

        Returns:
            float: Calculated EAR value
        """
        # Compute vertical eye distances
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute horizontal eye distance
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def check_blink(self, landmarks: np.ndarray) -> bool:
        """
        Detect eye blinks using the eye aspect ratio.

        Args:
            landmarks (np.ndarray): Array of facial landmarks

        Returns:
            bool: True if blink detected, False otherwise
        """
        try:
            # Extract eye landmarks
            left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Check for blink
            if ear < self.ear_threshold:
                self.blink_counter += 1
            else:
                if self.blink_counter >= self.ear_consec_frames:
                    self.blink_total += 1
                self.blink_counter = 0
                
            return self.blink_total > 0

        except Exception as e:
            logger.error(f"Error checking for blinks: {str(e)}")
            return False

    def estimate_head_pose(self, landmarks: np.ndarray, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Estimate head pose using facial landmarks.

        Args:
            landmarks (np.ndarray): Array of facial landmarks
            image_size (Tuple[int, int]): Image dimensions (height, width)

        Returns:
            Optional[np.ndarray]: Rotation vector if successful, None otherwise
        """
        try:
            # Camera internals
            focal_length = image_size[1]
            center = (image_size[1]/2, image_size[0]/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            
            # Assume no lens distortion
            dist_coeffs = np.zeros((4,1))
            
            # Get key facial landmarks for pose estimation
            image_points = np.array([
                landmarks[30],    # Nose tip
                landmarks[8],     # Chin
                landmarks[36],    # Left eye left corner
                landmarks[45],    # Right eye right corner
                landmarks[48],    # Left mouth corner
                landmarks[54]     # Right mouth corner
            ], dtype="double")
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            return rotation_vec if success else None

        except Exception as e:
            logger.error(f"Error estimating head pose: {str(e)}")
            return None

    def check_head_movement(self, landmarks: np.ndarray, image_size: Tuple[int, int]) -> bool:
        """
        Detect significant head movement between frames.

        Args:
            landmarks (np.ndarray): Array of facial landmarks
            image_size (Tuple[int, int]): Image dimensions (height, width)

        Returns:
            bool: True if significant head movement detected, False otherwise
        """
        try:
            # Get current head pose
            current_pose = self.estimate_head_pose(landmarks, image_size)
            
            if current_pose is None:
                return False
                
            if self.previous_pose is None:
                self.previous_pose = current_pose
                return False
                
            # Calculate pose difference
            pose_diff = np.abs(current_pose - self.previous_pose)
            self.previous_pose = current_pose
            
            # Check if pose change exceeds threshold
            return np.any(pose_diff > np.radians(self.pose_threshold))

        except Exception as e:
            logger.error(f"Error checking head movement: {str(e)}")
            return False

    def analyze_texture(self, face_region: np.ndarray) -> bool:
        """
        Analyze face texture for detecting printed photos or screen displays.

        Args:
            face_region (np.ndarray): Region of the image containing the face

        Returns:
            bool: True if texture appears to be from a real face, False otherwise
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply LBP or other texture analysis
            # Here we use a simple gradient-based approach
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Calculate texture score (higher for real faces)
            texture_score = np.mean(gradient_magnitude)
            
            # Threshold for texture score (may need adjustment)
            return texture_score > 20.0

        except Exception as e:
            logger.error(f"Error analyzing texture: {str(e)}")
            return False

    def check_liveness(self, image: np.ndarray) -> bool:
        """
        Perform complete liveness detection using multiple methods.

        Args:
            image (np.ndarray): Input image to check for liveness

        Returns:
            bool: True if the face is determined to be live, False otherwise
        """
        try:
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Get facial landmarks
            landmarks = self.get_landmarks(processed_image)
            if landmarks is None:
                return False
                
            # Perform all liveness checks
            blink_detected = self.check_blink(landmarks)
            head_movement = self.check_head_movement(landmarks, processed_image.shape[:2])
            texture_check = self.analyze_texture(processed_image)
            
            # Combine results (can be weighted if needed)
            liveness_score = sum([
                blink_detected,
                head_movement,
                texture_check
            ])
            
            # Require at least 2 positive checks for liveness
            return liveness_score >= 2

        except Exception as e:
            logger.error(f"Error checking liveness: {str(e)}")
            return False

    def reset(self):
        """Reset all counters and stored states."""
        self.blink_counter = 0
        self.blink_total = 0
        self.previous_pose = None
