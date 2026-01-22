import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
import os
import sys

image_folder = "images"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(BASE_DIR, "..", image_folder)
image_folder = os.path.normpath(image_folder)

print(f"Checking folder...")
print(f"Path: {image_folder}")
print(f"Exists: {os.path.exists(image_folder)}")

if not os.path.exists(image_folder):
    print("FOLDER NOT FOUND!")
    sys.exit(1)

emotion_images = {}
emotions_list = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']

def load_emotion_images(folder):
    print(f"\n Loading images from: {folder}")
    emotion_images = {}

    for emotion in emotions_list:
        img_path = os.path.join(folder, f"{emotion}.jpg")
        print(f"    {emotion}.jpg... ", end="", flush=True)

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                h, w, c = img.shape
                emotion_images[emotion] = img
            else:
                print("read failed")
        else:
            print("not found")

    print(f"Total loaded: {len(emotion_images)}/{len(emotions_list)}")
    return emotion_images

emotion_images = load_emotion_images(image_folder)
current_emotion_image = emotion_images[list(emotion_images.keys())[0]]
current_emotion = list(emotion_images.keys())[0]

emotion_images = load_emotion_images(image_folder)
if not emotion_images:
    print("No emotion images loaded, exiting.")
    sys.exit(1)

current_emotion = list(emotion_images.keys())[0]
current_emotion_image = emotion_images[current_emotion]

print(f"\n Opening webcam...")
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # width
cap.set(4, 540)   # height

if not cap.isOpened():
    print("Cannot open webcam!")
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

classifier = pipeline("image-classification", 
                    model="trpakov/vit-face-expression")

print("\n" + "="*30)
print("Press 'q' to quit")
print("="*30 + "\n")

current_emotion_image = emotion_images[list(emotion_images.keys())[0]] 
current_emotion = list(emotion_images.keys())[0]
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    frame_count += 1
    h, w = frame.shape[:2]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, fw, fh) in faces:
        face_roi = frame[y:y+fh, x:x+fw]
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        
        try:
            predictions = classifier(face_pil)
            if predictions:
                emotion = predictions[0]['label'].lower()
                confidence = predictions[0]['score']
                
                if emotion in emotion_images:
                    current_emotion = emotion
                    current_emotion_image = emotion_images[emotion]
                
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 255), 2)
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        except Exception as e:
            print(f"  Error: {e}")
    
    try:
        if current_emotion_image is not None:
            emotion_img_resized = cv2.resize(current_emotion_image, (w, h))
            combined = np.hstack([frame, emotion_img_resized])
            
            cv2.putText(combined, f"Emotion: {current_emotion.upper()}", 
                    (w + 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            window_name = "  Your Face | Emotion Image"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            target_width = 1600
            target_height = 900
            combined_resized = cv2.resize(combined, (target_width, target_height))
            cv2.imshow(window_name, combined_resized)
        else:
            cv2.imshow(window_name, combined_resized)
            
    except Exception as e:
        print(f"Error creating display: {e}")
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('d'):
        if image_folder.endswith(os.path.sep + "images"):
            image_folder = os.path.join(os.path.dirname(image_folder), "images_2")
        else:
            image_folder = os.path.join(os.path.dirname(image_folder), "images")

        image_folder = os.path.normpath(image_folder)
        emotion_images = load_emotion_images(image_folder)

        if emotion_images:
            current_emotion = list(emotion_images.keys())[0]
            current_emotion_image = emotion_images[current_emotion]
        print(f"\n🔁 Dev toggle: now using {image_folder}")
        continue

    if key == ord('q'):
        print("\nQuitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
