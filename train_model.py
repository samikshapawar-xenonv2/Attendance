import face_recognition
import numpy as np
import pickle
import os
import re

# --- Configuration ---
# NOTE: Update this path if your 'public' folder is elsewhere
DATASET_PATH = os.path.join(os.getcwd(), 'public') 
ENCODINGS_FILE = "face_encodings.pkl"

def train_and_save_encodings(dataset_path, output_file):
    """
    Scans the dataset folder, generates face encodings, and saves them to a file.
    Also returns the roster details for database synchronization.
    """
    known_face_encodings = []
    known_face_names = []
    student_roster = []

    print(f"Scanning directory: {dataset_path}")
    
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
            print(f"Processing student: {roll_no} - {name}")
            
            # Placeholder for student data (for DB sync)
            student_roster.append({'RollNo': roll_no, 'Name': name})

            # Iterate through all images in the student's folder
            images_processed = 0
            for image_file in os.listdir(folder_path):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, image_file)
                    
                    try:
                        image = face_recognition.load_image_file(image_path)
                        # Find faces in the image
                        face_locations = face_recognition.face_locations(image)

                        if len(face_locations) == 1:
                            # Generate the 128-dimension face encoding
                            encoding = face_recognition.face_encodings(image, face_locations)[0]
                            known_face_encodings.append(encoding)
                            # Use the combined ID for matching later
                            known_face_names.append(f"{roll_no}_{name}") 
                            images_processed += 1
                        elif len(face_locations) > 1:
                            print(f"  Warning: Multiple faces found in {image_file}. Skipping.")
                        else:
                            print(f"  Warning: No face found in {image_file}. Skipping.")
                    except Exception as e:
                        print(f"  Error processing {image_file}: {e}")

            print(f"  Successfully encoded {images_processed} image(s).")
            
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
