import os
import cv2

def crop_face(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Load the pre-trained face detection model (e.g., Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to load image file '{image_path}'. Check if it's a valid image file.")

    # Convert the image to grayscale (face detection works on grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) != 1:
        raise ValueError(f"Expected exactly 1 face in the image, but found {len(faces)} faces.")

    # Crop the image to isolate the face
    (x, y, w, h) = faces[0]
    face_crop = image[y:y+h, x:x+w]

    return face_crop

# Test the function with the image path
image_path = 'C:\\Users\\ROEE\\AppData\\Local\\Programs\\Python\\Python310\\angry.jpg'
cropped_face = crop_face(image_path)
