import os
import pickle
import cv2
import face_recognition
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceService:
    def __init__(self, encodings_file: str, detection_model: str = 'hog', 
                 tolerance: float = 0.6, confidence_threshold: float = 0.5):
        self.encodings_file = encodings_file
        self.detection_model = detection_model
        self.tolerance = tolerance
        self.confidence_threshold = confidence_threshold
        self.known_face_encodings = []
        self.known_face_ids = []
        self.model_loaded = False

    def load_model(self) -> bool:
        """Loads the pre-trained face encodings from the pickle file."""
        if not os.path.exists(self.encodings_file):
            logger.error(f"Model file '{self.encodings_file}' not found.")
            return False
        
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data.get("encodings", [])
                self.known_face_ids = data.get("names", [])
            
            self.model_loaded = True
            logger.info(f"ML Model loaded successfully. {len(self.known_face_ids)} students registered.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def process_frame(self, frame: np.ndarray, resize_scale: float = 0.25) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Processes a video frame to detect and recognize faces.
        Returns the annotated frame and a list of detected results.
        """
        if not self.model_loaded:
            return frame, []

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model=self.detection_model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)

        detected_results = []
        scale_factor = int(1 / resize_scale)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Scale back up face locations
            top *= scale_factor
            right *= scale_factor
            bottom *= scale_factor
            left *= scale_factor
            
            name = "Unknown"
            confidence = 0.0
            roll_no = None
            is_match = False

            if self.known_face_encodings:
                # Calculate distances to all known faces
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                
                confidence = 1 - best_distance
                
                # Check if match meets criteria
                if confidence >= self.confidence_threshold and best_distance <= self.tolerance:
                    full_id = self.known_face_ids[best_match_index]
                    try:
                        roll_no, student_name = full_id.split('_', 1)
                        name = student_name.replace('_', ' ')
                        is_match = True
                    except ValueError:
                        logger.warning(f"Invalid ID format: {full_id}")
                        name = full_id

            detected_results.append({
                'name': name,
                'roll_no': roll_no,
                'confidence': confidence,
                'location': (top, right, bottom, left),
                'is_match': is_match
            })

        return frame, detected_results

    @staticmethod
    def draw_results(frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Draws bounding boxes and labels on the frame."""
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            is_match = result['is_match']

            color = (0, 255, 0) if is_match else (0, 0, 255)
            display_name = f"{name} ({confidence:.0%})"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, display_name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
