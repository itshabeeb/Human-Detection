import cv2
import os

def capture_images(name, num_samples=10):
    path = f'face_dataset/{name}'
    os.makedirs(path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print(f"Capturing {num_samples} images for {name}...")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('Capturing...', frame)

        k = cv2.waitKey(100)
        if k == 27:  # ESC to exit
            break
        elif count < num_samples and k == ord('s'):  # Press 's' to save image
            filename = os.path.join(path, f'{count + 1}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            count += 1
            if count >= num_samples:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing images for {name}.")

if __name__ == '__main__':
    person_name = input("Enter the name of the person: ")
    num_images = int(input("Enter the number of images to capture: "))
    capture_images(person_name, num_images)