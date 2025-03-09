import os
import sys
import logging
import cv2
import numpy as np
from typing import Optional

# Add parent directory to path to import SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition_sdk import FaceRecognitionSDK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRecognitionDemo:
    """Demo application for the Face Recognition SDK."""

    def __init__(self):
        """Initialize the demo application."""
        self.sdk = FaceRecognitionSDK()
        self.camera_id = 0
        self.display_scale = 1.0
        self.show_landmarks = True
        self.show_fps = True
        self.running = False

    def initialize_sdk(self, model_path: str, database_path: Optional[str] = None) -> bool:
        """
        Initialize the SDK with models and face database.

        Args:
            model_path (str): Path to the directory containing model files
            database_path (Optional[str]): Path to the directory containing known faces

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load models
            if not self.sdk.load_models(model_path):
                logger.error("Failed to load models")
                return False

            # Load face database if provided
            if database_path and not self.sdk.load_face_database(database_path):
                logger.error("Failed to load face database")
                return False

            return True

        except Exception as e:
            logger.error(f"Error initializing SDK: {str(e)}")
            return False

    def start_demo(self):
        """Start the demo application."""
        try:
            # Load test image from URL
            import urllib.request
            test_image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
            resp = urllib.request.urlopen(test_image_url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not load test image from {test_image_url}")

            self.running = True
            logger.info("Starting demo application with test image...")

            # Process single frame and save to file
            processed_frame = self.process_frame(frame)
            output_path = "processed_image.jpg"
            cv2.imwrite(output_path, processed_frame)
            logger.info(f"Processed image saved to {output_path}")

        except Exception as e:
            logger.error(f"Error in demo application: {str(e)}")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Args:
            frame (np.ndarray): Input frame from camera

        Returns:
            np.ndarray: Processed frame with visualizations
        """
        try:
            # Create a copy for visualization
            display = frame.copy()

            # Detect faces
            faces = self.sdk.face_detector.detect_faces(frame)

            # Process each detected face
            for face_coords in faces:
                x, y, w, h = face_coords
                face_roi = frame[y:y+h, x:x+w]

                # Check liveness
                is_live = self.sdk.check_liveness(face_roi)
                
                # Get landmarks if enabled
                landmarks = None
                if self.show_landmarks:
                    landmarks = self.sdk.liveness_detector.get_landmarks(face_roi)

                # Perform recognition if face is live
                name = None
                if is_live:
                    name = self.sdk.recognize_face(face_roi)

                # Draw visualizations
                self.draw_results(
                    display,
                    face_coords,
                    is_live,
                    name,
                    landmarks
                )

            return display

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame

    def draw_results(
        self,
        frame: np.ndarray,
        face_coords: tuple,
        is_live: bool,
        name: Optional[str],
        landmarks: Optional[np.ndarray]
    ) -> None:
        """
        Draw detection results on the frame.

        Args:
            frame (np.ndarray): Frame to draw on
            face_coords (tuple): Face bounding box coordinates
            is_live (bool): Whether the face is determined to be live
            name (Optional[str]): Recognized person's name
            landmarks (Optional[np.ndarray]): Facial landmarks if available
        """
        try:
            x, y, w, h = face_coords

            # Draw face bounding box
            color = (0, 255, 0) if is_live else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Draw liveness status
            status = "Live" if is_live else "Fake"
            cv2.putText(
                frame,
                status,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # Draw name if recognized
            if name:
                cv2.putText(
                    frame,
                    name,
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            # Draw landmarks if available
            if landmarks is not None and self.show_landmarks:
                for (x, y) in landmarks:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        except Exception as e:
            logger.error(f"Error drawing results: {str(e)}")

def main():
    """Main function to run the demo."""
    try:
        # Initialize demo
        demo = FaceRecognitionDemo()

        # Set paths
        model_path = os.path.join(os.path.dirname(__file__), "models")
        database_path = os.path.join(os.path.dirname(__file__), "data", "known_faces")

        # Initialize SDK
        if not demo.initialize_sdk(model_path, database_path):
            logger.error("Failed to initialize SDK")
            return

        # Start demo
        demo.start_demo()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
