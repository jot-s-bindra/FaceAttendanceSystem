import os
import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules import detection
import pandas as pd

img_path = "a2.png"  
db_path = "face_database/"

image = cv2.imread(img_path)

faces = detection.extract_faces(img_path, detector_backend="mtcnn", enforce_detection=False)

num_faces = len(faces)
print(f"✅ Detected {num_faces} face(s) in the image.")

assigned_persons = []
used_persons = set()
bounding_boxes = []  # Store bounding box coordinates

if num_faces > 0:
    results = DeepFace.find(
        img_path=img_path,
        db_path=db_path,
        model_name="Facenet512",
        detector_backend="mtcnn",
        enforce_detection=False
    )

    match_candidates = []
    for df in results:
        if not df.empty:
            for _, row in df.iterrows():
                identity_path = row["identity"]
                person_name = identity_path.split("/")[-2]  
                distance = row["distance"]  
                match_candidates.append((person_name, distance))

    for i, face in enumerate(faces):
        best_match = None
        best_distance = float("inf")

        x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
        bounding_boxes.append((x, y, w, h))  # Save bounding box

        for person, distance in match_candidates:
            if person not in used_persons and distance < best_distance:
                best_match = person
                best_distance = distance

        if best_match:
            assigned_persons.append(best_match)
            used_persons.add(best_match)  
        else:
            assigned_persons.append("Unknown")  
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    output_path = "output.jpg"
    cv2.imwrite(output_path, image)
    print(f"✅ Bounding boxes drawn. Saved as {output_path}")

    print(f"🔹 Identified persons: {', '.join(assigned_persons)}")

else:
    print("❌ No faces detected.")
