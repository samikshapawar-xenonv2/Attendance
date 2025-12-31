"""
Face Recognition Accuracy Configuration
Adjust these settings to fine-tune recognition accuracy vs speed
"""

# ===== TRAINING CONFIGURATION =====

# Face detection model for training
# 'hog' = faster but less accurate (good for CPU)
# 'cnn' = slower but more accurate (requires more processing power)
TRAIN_FACE_DETECTION_MODEL = 'cnn'

# Number of times to re-sample face during encoding (default: 1)
# Higher = more accurate but slower training
# Recommended: 5-15 for best accuracy
TRAIN_NUM_JITTERS = 12

# Minimum image dimensions (in pixels) to accept
TRAIN_MIN_IMAGE_SIZE = 100


# ===== RECOGNITION CONFIGURATION =====

# Face detection model for real-time recognition
# 'hog' = ~5-10 FPS, good for basic accuracy
# 'cnn' = ~1-3 FPS, excellent accuracy
RECOGNITION_FACE_DETECTION_MODEL = 'cnn'

# Recognition tolerance (0.0 - 1.0)
# Lower = more strict (fewer false positives, might miss some faces)
# Higher = more lenient (more detections, but might have false positives)
# Default: 0.6, Recommended: 0.4-0.5 for high accuracy
RECOGNITION_TOLERANCE = 0.45

# Confidence threshold (0.0 - 1.0)
# Minimum confidence score to accept a match
# Confidence = 1 - distance
# Recommended: 0.5-0.6
CONFIDENCE_THRESHOLD = 0.55

# Minimum consecutive detections before marking attendance
# Prevents false positives from single frame misdetections
# Recommended: 3-5 detections
MIN_DETECTION_COUNT = 3

# Process every Nth frame (balance between speed and accuracy)
# Lower = more frequent processing (slower but more responsive)
# Higher = less frequent processing (faster but might miss faces)
# Recommended: 2-3
PROCESS_EVERY_N_FRAMES = 2


# ===== CAMERA CONFIGURATION =====

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Frame scaling factor for processing
# Smaller = faster processing but lower accuracy
# 1.0 = full resolution, 0.5 = half resolution
FRAME_SCALE_FACTOR = 0.5


# ===== PERFORMANCE PRESETS =====

def apply_preset(preset='balanced'):
    """
    Apply predefined accuracy/speed presets
    
    Presets:
    - 'fast': Prioritize speed (good for older computers)
    - 'balanced': Balance between speed and accuracy (recommended)
    - 'accurate': Maximum accuracy (requires good hardware)
    """
    global RECOGNITION_FACE_DETECTION_MODEL, RECOGNITION_TOLERANCE
    global CONFIDENCE_THRESHOLD, MIN_DETECTION_COUNT, PROCESS_EVERY_N_FRAMES
    
    if preset == 'fast':
        RECOGNITION_FACE_DETECTION_MODEL = 'hog'
        RECOGNITION_TOLERANCE = 0.5
        CONFIDENCE_THRESHOLD = 0.5
        MIN_DETECTION_COUNT = 2
        PROCESS_EVERY_N_FRAMES = 3
        print("✓ Applied 'FAST' preset")
        
    elif preset == 'balanced':
        RECOGNITION_FACE_DETECTION_MODEL = 'cnn'
        RECOGNITION_TOLERANCE = 0.45
        CONFIDENCE_THRESHOLD = 0.55
        MIN_DETECTION_COUNT = 3
        PROCESS_EVERY_N_FRAMES = 2
        print("✓ Applied 'BALANCED' preset (recommended)")
        
    elif preset == 'accurate':
        RECOGNITION_FACE_DETECTION_MODEL = 'cnn'
        RECOGNITION_TOLERANCE = 0.4
        CONFIDENCE_THRESHOLD = 0.6
        MIN_DETECTION_COUNT = 5
        PROCESS_EVERY_N_FRAMES = 1
        print("✓ Applied 'ACCURATE' preset")
        
    else:
        print(f"⚠ Unknown preset '{preset}'. Using default 'balanced' settings.")


# ===== TIPS FOR IMPROVING ACCURACY =====
"""
1. IMAGE QUALITY:
   - Use high-quality images (at least 200x200 pixels per face)
   - Ensure good lighting (avoid shadows on faces)
   - Take photos from multiple angles
   - Use 5-10 images per student

2. TRAINING:
   - Use NUM_JITTERS = 10-15 for best results
   - Use 'cnn' model for training
   - Ensure only one face per training image

3. RECOGNITION:
   - Lower RECOGNITION_TOLERANCE for fewer false positives
   - Increase MIN_DETECTION_COUNT to prevent accidental detections
   - Use good lighting during attendance
   - Position camera 1-2 meters from students

4. HARDWARE:
   - Use 'cnn' model if you have good CPU/GPU
   - Use 'hog' model on older computers
   - Consider using GPU acceleration (requires dlib compiled with CUDA)
"""
