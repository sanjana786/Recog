import face_recognition
import os
import pickle

# Load known faces and their names from a directory
known_faces = []
known_names = []

known_faces_dir = "image_subset"
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg"):
        face_image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        face_encodings = face_recognition.face_encodings(face_image)
        
        # Check if a face was found in the image
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]  # Assuming one face per image
            known_faces.append(face_encoding)
            known_names.append(os.path.splitext(filename)[0])

# Save the known face data to a pickle file
with open("known_faces_data.pkl", "wb") as file:
    pickle.dump((known_faces, known_names), file)

# Now, known_faces_data.pkl contains your known face encodings and names.
