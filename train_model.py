import face_recognition
import mediapipe as mp
import numpy as np
import pickle
import os
import re
import cv2

# --- Configuration ---
# NOTE: Update this path if your 'public' folder is elsewhere
DATASET_PATH = os.path.join(os.getcwd(), 'public') 
ENCODINGS_FILE = "face_encodings.pkl"

# Improved accuracy settings
MIN_IMAGE_SIZE = 100  # Minimum face size to accept
NUM_JITTERS = 15  # Increased for better encoding quality (default is 1)
USE_ALL_ENCODINGS = True  # Store all encodings per person (better accuracy)
USE_CNN_FOR_TRAINING = True  # Use CNN model for more accurate face detection during training

# Try MediaPipe, fallback to dlib
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        USE_MEDIAPIPE = True
        print("Using MediaPipe for face detection")
    else:
        USE_MEDIAPIPE = False
        print("MediaPipe solutions not available, using dlib")
except Exception as e:
    USE_MEDIAPIPE = False
    print(f"MediaPipe not available ({e}), using dlib")

def train_and_save_encodings(dataset_path, output_file):
    """
    Scans the dataset folder, generates face encodings, and saves them to a file.
    Also returns the roster details for database synchronization.
    Uses improved accuracy settings with CNN model and multiple jitters.
    """
    known_face_encodings = []
    known_face_names = []
    student_roster = []

    print(f"Scanning directory: {dataset_path}")
    print(f"Scanning directory: {dataset_path}")
    print(f"Using face detection model: MediaPipe (High Performance)")
    print(f"Number of jitters for encoding: {NUM_JITTERS}")
    
    # Iterate through all student folders (e.g., '07_Samiksha_Pawar')
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue

        try:
            # 1. Extract RollNo and Name from folder name
            match = re.match(r'(\d+)_([A-Za-z_]+)', folder_name)
            if not match:
                print(f"Warning: Skipping folder {folder_name}. Name format incorrect.")
                continue
            
            roll_no = int(match.group(1))
            name = match.group(2).replace('_', ' ') # Replace underscores with spaces

            # 2. Process images within the student folder
            print(f"\nProcessing student: {roll_no} - {name}")
            
            # Placeholder for student data (for DB sync)
            student_roster.append({'RollNo': roll_no, 'Name': name})

            # Store multiple encodings per student for better accuracy
            student_encodings = []
            
            # Iterate through all images in the student's folder
            images_processed = 0
            for image_file in os.listdir(folder_path):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, image_file)
                    
                    try:
                        # Load and preprocess image
                        image = face_recognition.load_image_file(image_path)
                        
                        # Check image quality
                        if image.shape[0] < MIN_IMAGE_SIZE or image.shape[1] < MIN_IMAGE_SIZE:
                            print(f"  Warning: Image {image_file} too small. Skipping.")
                            continue
                        
                        # Find faces using MediaPipe or dlib
                        height, width, _ = image.shape
                        
                        if USE_MEDIAPIPE:
                            # Use MediaPipe
                            results = face_detection.process(image)
                            face_locations = []
                            
                            if results.detections:
                                for detection in results.detections:
                                    bboxC = detection.location_data.relative_bounding_box
                                    x = int(bboxC.xmin * width)
                                    y = int(bboxC.ymin * height)
                                    w = int(bboxC.width * width)
                                    h = int(bboxC.height * height)
                                    top = max(0, y)
                                    right = min(width, x + w)
                                    bottom = min(height, y + h)
                                    left = max(0, x)
                                    face_locations.append((top, right, bottom, left))
                        else:
                            # Use dlib (CNN for better accuracy during training)
                            model = 'cnn' if USE_CNN_FOR_TRAINING else 'hog'
                            face_locations = face_recognition.face_locations(image, model=model)

                        if len(face_locations) == 1:
                            # Generate the 128-dimension face encoding with multiple jitters
                            encodings = face_recognition.face_encodings(
                                image, 
                                face_locations,
                                num_jitters=NUM_JITTERS
                            )
                            
                            if len(encodings) > 0:
                                encoding = encodings[0]
                                student_encodings.append(encoding)
                                images_processed += 1
                                print(f"  ✓ Encoded {image_file}")
                        elif len(face_locations) > 1:
                            print(f"  ⚠ Multiple faces found in {image_file}. Skipping.")
                        else:
                            print(f"  ⚠ No face found in {image_file}. Skipping.")
                    except Exception as e:
                        print(f"  ✗ Error processing {image_file}: {e}")

            # Store ALL encodings for this student (better accuracy than averaging)
            if student_encodings:
                if USE_ALL_ENCODINGS:
                    # Store each encoding separately for better matching
                    for enc in student_encodings:
                        known_face_encodings.append(enc)
                        known_face_names.append(f"{roll_no}_{name}")
                    print(f"  ✓ Stored {len(student_encodings)} separate encodings for better accuracy")
                elif len(student_encodings) > 1:
                    # Create averaged encoding from all images
                    avg_encoding = np.mean(student_encodings, axis=0)
                    known_face_encodings.append(avg_encoding)
                    known_face_names.append(f"{roll_no}_{name}")
                    print(f"  ✓ Created averaged encoding from {len(student_encodings)} images")
                else:
                    # Use single encoding
                    known_face_encodings.append(student_encodings[0])
                    known_face_names.append(f"{roll_no}_{name}")
                    print(f"  ✓ Created encoding from 1 image")
                
                print(f"  Successfully encoded {images_processed} image(s) for {name}.")
            else:
                print(f"  ✗ No valid encodings generated for {name}")
            
        except Exception as e:
            print(f"General error processing folder {folder_name}: {e}")

    # 3. Save the encodings
    if known_face_encodings:
        data = {
            "encodings": known_face_encodings,
            "names": known_face_names
        }
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print("\n--- Training Complete ---")
        print(f"Total students encoded: {len(student_roster)}")
        print(f"Encodings saved successfully to {output_file}")
        return student_roster
    else:
        print("\n--- Training Failed ---")
        print("No faces were successfully encoded.")
        return []

if __name__ == "__main__":
    # Ensure the 'public' folder path is correct relative to where you run this script
    train_and_save_encodings(DATASET_PATH, ENCODINGS_FILE)
