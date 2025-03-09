import os
import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import face_recognition
from ..utils.image_processing import preprocess_image

# Configure logging
logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Class for face recognition using the face_recognition library with dlib backend.
    Handles loading known faces, computing face encodings, and performing recognition.
    """

    def __init__(self, model_path: str, tolerance: float = 0.6):
        """
        Initialize the face recognizer.

        Args:
            model_path (str): Path to the shape predictor model file
            tolerance (float): Tolerance for face recognition (lower is stricter)
        """
        self.model_path = model_path
        self.tolerance = tolerance
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.database_loaded = False

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Shape predictor model not found at {model_path}")
            
            logger.info("Face recognizer initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing face recognizer: {str(e)}")
            raise

    def load_face_database(self, database_path: str) -> bool:
        """
        Load known faces from the database directory.
        Expected structure:
        database_path/
            person1_name/
                image1.jpg
                image2.jpg
            person2_name/
                image1.jpg
                ...

        Args:
            database_path (str): Path to the directory containing known face images

        Returns:
            bool: True if database loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(database_path):
                raise FileNotFoundError(f"Database path {database_path} does not exist")

            # Clear existing database
            self.known_face_encodings = []
            self.known_face_names = []

            # Supported image extensions
            valid_extensions = ('.jpg', '.jpeg', '.png')

            # Iterate through each person's directory
            for person_name in os.listdir(database_path):
                person_dir = os.path.join(database_path, person_name)
                
                if not os.path.isdir(person_dir):
                    continue

                # Process each image in the person's directory
                for image_name in os.listdir(person_dir):
                    if not image_name.lower().endswith(valid_extensions):
                        continue

                    image_path = os.path.join(person_dir, image_name)
                    
                    # Load and process image
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)

                    if len(face_encodings) != 1:
                        logger.warning(
                            f"Skipping {image_path}: Found {len(face_encodings)} faces "
                            f"(expected exactly 1 face)"
                        )
                        continue

                    # Add face encoding and name to database
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(person_name)

            if not self.known_face_encodings:
                raise ValueError("No valid faces found in the database")

            self.database_loaded = True
            logger.info(f"Loaded {len(self.known_face_encodings)} faces from database")
            return True

        except Exception as e:
            logger.error(f"Error loading face database: {str(e)}")
            self.database_loaded = False
            return False

    def get_face_encoding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the face encoding for a given face image.

        Args:
            face_image (np.ndarray): Image containing a face

        Returns:
            Optional[np.ndarray]: Face encoding if successful, None otherwise
        """
        try:
            # Preprocess image
            processed_image = preprocess_image(face_image)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(processed_image)
            
            if not face_encodings:
                logger.warning("No face found in the image")
                return None
                
            if len(face_encodings) > 1:
                logger.warning("Multiple faces found in the image, using the first one")
                
            return face_encodings[0]

        except Exception as e:
            logger.error(f"Error computing face encoding: {str(e)}")
            return None

    def recognize_face(self, face_image: np.ndarray) -> Optional[str]:
        """
        Recognize a face by comparing it with the known face database.

        Args:
            face_image (np.ndarray): Image containing the face to recognize

        Returns:
            Optional[str]: Name of the recognized person, or None if not recognized
        """
        try:
            if not self.database_loaded:
                raise RuntimeError("Face database not loaded")

            # Get face encoding
            face_encoding = self.get_face_encoding(face_image)
            if face_encoding is None:
                return None

            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=self.tolerance
            )
            
            if not any(matches):
                logger.debug("Face not recognized in database")
                return None

            # Get the best match
            face_distances = face_recognition.face_distance(
                self.known_face_encodings,
                face_encoding
            )
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                recognized_name = self.known_face_names[best_match_index]
                logger.debug(f"Face recognized as: {recognized_name}")
                return recognized_name

            return None

        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return None

    def add_face_to_database(self, name: str, face_image: np.ndarray) -> bool:
        """
        Add a new face to the in-memory database.

        Args:
            name (str): Name of the person
            face_image (np.ndarray): Image containing the person's face

        Returns:
            bool: True if face was added successfully, False otherwise
        """
        try:
            # Get face encoding
            face_encoding = self.get_face_encoding(face_image)
            if face_encoding is None:
                return False

            # Add to database
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            logger.info(f"Added new face to database: {name}")
            return True

        except Exception as e:
            logger.error(f"Error adding face to database: {str(e)}")
            return False

    def get_face_similarity(self, face1: np.ndarray, face2: np.ndarray) -> Optional[float]:
        """
        Compute similarity between two face images.

        Args:
            face1 (np.ndarray): First face image
            face2 (np.ndarray): Second face image

        Returns:
            Optional[float]: Similarity score (0-1) where 1 is perfect match,
                           or None if computation fails
        """
        try:
            # Get face encodings
            encoding1 = self.get_face_encoding(face1)
            encoding2 = self.get_face_encoding(face2)
            
            if encoding1 is None or encoding2 is None:
                return None

            # Compute face distance
            distance = face_recognition.face_distance([encoding1], encoding2)[0]
            
            # Convert distance to similarity score (0-1)
            similarity = 1 - min(distance, 1.0)
            
            return similarity

        except Exception as e:
            logger.error(f"Error computing face similarity: {str(e)}")
            return None
