import cv2
from deepface import DeepFace

# Path to face database folder
db_path = "face_database/"

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        continue

    # Save a temp image from webcam
    cv2.imwrite("temp.jpg", frame)

    try:
        # Find the person in the database
        result = DeepFace.find(img_path="temp.jpg", db_path=db_path, enforce_detection=False)

        # Get the name if a match is found
        if result and len(result[0]) > 0:
            identity = result[0]["identity"][0]  # Get first match
            name = identity.split("/")[-2]  # Extract name from folder
        else:
            name = "Unknown"

        # Display name on the frame
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Error:", e)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
