# Human-Detection

Repository for Human-Robot Interaction (Experimental)

## Overview

This project uses computer vision to detect humans and recognize faces in real-time using a webcam. It leverages YOLOv5 for human detection and DeepFace for face recognition.

## Directory Structure

- `human_detector.py`: Main script for human detection and face recognition.
- `live.py`: Script to capture face images for dataset creation.
- `enode.py`: Generates face encodings from the dataset.
- `face_dataset/`: Stores face images organized by person name.
- `yolov5s.pt`: YOLOv5 model weights.
- `requirements.txt`: Python dependencies.
- `face_encodings.npy`, `face_names.npy`: Saved face encodings and names.
- `venv311/`: Python virtual environment.

## Setup

1. **Install dependencies:**
   ```
   
   pip install -r requirements.txt

2. Prepare the face dataset:
    Run live.py to capture images
   ```python live.py```
3. Encode faces:
    Run enode.py to generate encodings:
    ```python enode.py```
4. Run human detection and face recognition:
    ```python human_detector.py```

   
## Requrirements

- Python 3.11+
- OpenCV
- torch
- deepface
- face_recognition
- numpy

See requirements.txt for details.

