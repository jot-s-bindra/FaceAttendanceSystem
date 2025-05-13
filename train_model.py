# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

with open("output/recognizer.pickle", "wb") as f:
    pickle.dump(recognizer, f)

# write the label encoder to disk
with open("output/le.pickle", "wb") as f:
    pickle.dump(le, f)
