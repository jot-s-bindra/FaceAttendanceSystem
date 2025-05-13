import os
import pickle
from deepface import DeepFace

# Disable GPU to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Get absolute path of the database
db_path = os.path.abspath("face_database")

# Check if database exists
if not os.path.exists(db_path):
    raise ValueError(f"❌ Directory not found: {db_path}")

print(f"✅ Using database path: {db_path}")

# List all subfolders (persons) and ignore non-directory files
people = [p for p in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, p))]
print(f"✅ Found subfolders (people): {people}")

# Store embeddings in a dictionary
embeddings_dict = {}

# Process each person
for person in people:
    person_folder = os.path.join(db_path, person)

    # List image files (ignore non-image files)
    images = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"⚠️ No images found for {person}, skipping...")
        continue

    print(f"🔍 {person}: {len(images)} images found -> {images[:5]}")  # Print first 5 images

    person_embeddings = []

    # Process each image
    for img_name in images:
        img_path = os.path.join(person_folder, img_name)
        print(f"📌 Processing {img_path}...")

        try:
            # Generate embeddings with `Facenet512`
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet512",
                enforce_detection=False
            )
            
            # Extract the embedding vector
            for obj in embedding_objs:
                embedding = obj["embedding"]
                person_embeddings.append(embedding)

        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

    # Store embeddings if at least one valid embedding exists
    if person_embeddings:
        embeddings_dict[person] = person_embeddings

# Save embeddings to a pickle file for faster recognition
embeddings_file = os.path.join(db_path, "embeddings.pkl")
with open(embeddings_file, "wb") as f:
    pickle.dump(embeddings_dict, f)

print(f"✅ Face embeddings saved in {embeddings_file}!")
