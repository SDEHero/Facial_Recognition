import cv2
import os

name = input("Enter your name: ").strip().lower()
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{dataset_path}/{name}.{count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Collecting Faces', frame)
    if cv2.waitKey(1) == 27 or count >= 30:  # ESC or 30 images
        break

cam.release()
cv2.destroyAllWindows()
