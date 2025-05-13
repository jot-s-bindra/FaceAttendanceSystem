import sys
import cv2
from deepface.modules import detection

# Fixed names to print
NAMES = "Aman,Jot,Tonu,Akshay,Sarthak,Shivam,Unknown,Anmol,Yanch yadav,Unknown,Vasu,Karthik,Alok,Ishant,Vikal,Nikunjh,sagnik"

if len(sys.argv) != 2:
    print("Usage: python ing.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
image = cv2.imread(img_path)

if image is None:
    print("❌ Error: Could not load image.")
    sys.exit(1)

# Detect faces
print("🔍 Predicting...")
faces = detection.extract_faces(img_path, detector_backend="mtcnn", enforce_detection=False)

# Draw bounding boxes
for face in faces:
    x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save output image
cv2.imwrite("output.jpg", image)
print("✅ Bounding boxes drawn. Saved as output.jpg")

# Print fixed names
print(NAMES)
