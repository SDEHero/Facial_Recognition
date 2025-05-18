import cv2
import numpy as np
import os
import json

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

dataset_path = 'dataset'
label_ids = {}
current_id = 0
faces = []
ids = []

for filename in os.listdir(dataset_path):
    if not filename.endswith(".jpg"):
        continue
    name = filename.split(".")[0].lower()
    if name not in label_ids:
        label_ids[name] = current_id
        current_id += 1
    user_id = label_ids[name]

    img_path = os.path.join(dataset_path, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    face_rects = face_cascade.detectMultiScale(img)
    for (x, y, w, h) in face_rects:
        face = img[y:y+h, x:x+w]
        faces.append(face)
        ids.append(user_id)

recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")

# Save the name-ID map
with open("labels.json", "w") as f:
    json.dump(label_ids, f)

print(f"[INFO] Training complete. Labels saved to labels.json.")
