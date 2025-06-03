import face_recognition
import os
import numpy as np

def encode_faces(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, name)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                filepath = os.path.join(person_path, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    face_locations = face_recognition.face_locations(image)
                    if len(face_locations) > 0:
                        face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(name)
                    else:
                        print(f"No face found in {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    return known_face_encodings, known_face_names

if __name__ == '__main__':
    dataset_folder = 'face_dataset'
    known_encodings, known_names = encode_faces(dataset_folder)

    # You can save these encodings and names to a file for later use
    np.save('face_encodings.npy', known_encodings)
    np.save('face_names.npy', known_names)

    print("Face encodings generated and saved!")
    print(f"Known faces: {known_names}")