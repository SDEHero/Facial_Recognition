import cv2
import json

# Load model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
with open("labels.json", "r") as f:
    labels = json.load(f)
# Reverse mapping: id → name
labels = {v: k for k, v in labels.items()}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)
        if confidence < 60:
            name = labels.get(id_, "Unknown")
            label = f"{name} ({round(100 - confidence)}%)"
        else:
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Recognizing Face', frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
