from deepface import DeepFace
import cv2
import torch
import numpy as np
import os

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device).eval()

face_database_path = 'face_dataset'

def detect_humans(frame):
    results = model(frame)
    human_locations = []
    for *xyxy, conf, cls in reversed(results.xyxy[0].tolist()):
        if int(cls) == 0:  # Class ID 0 is for 'person'
            x1, y1, x2, y2 = map(int, xyxy)
            human_locations.append(((y1, x2, y2, x1), conf)) # Convert to top, right, bottom, left format
    return human_locations, results

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    human_locations_with_conf, _ = detect_humans(rgb_frame)

    for loc, conf in human_locations_with_conf:
        top, right, bottom, left = loc
        face_image = rgb_frame[top:bottom, left:right]
        face_image_bgr = frame[top:bottom, left:right] # Keep a BGR version for drawing

        try:
            # DeepFace.find() needs the image and the database path
            df = DeepFace.find(face_image, db_path=face_database_path, enforce_detection=False, silent=True)

            name = "Unknown"
            # Handle both list and DataFrame return types
            if isinstance(df, list):
                if len(df) > 0 and hasattr(df[0], 'empty') and not df[0].empty:
                    identity_path = df[0]['identity'].iloc[0]
                    name = os.path.basename(os.path.dirname(identity_path))
            elif hasattr(df, 'empty') and not df.empty:
                identity_path = df['identity'].iloc[0]
                name = os.path.basename(os.path.dirname(identity_path))

            # Draw bounding box and name for the recognized face (within the human box)
            cv2.rectangle(face_image_bgr, (0, 0), (face_image_bgr.shape[1], face_image_bgr.shape[0]), (255, 0, 0), 2)
            cv2.putText(face_image_bgr, name, (5, face_image_bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            # Place the processed face region back into the frame (optional, for visualization)
            frame[top:bottom, left:right] = face_image_bgr

            # Draw human bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'Human ({conf:.2f})', (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error during DeepFace analysis: {e}")
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'Human ({conf:.2f})', (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Human and Face Recognition (DeepFace)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()