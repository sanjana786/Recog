import face_recognition
import pickle

# Load known face data from the pickle file
with open("known_faces_data.pkl", "rb") as file:
    known_faces, known_names = pickle.load(file)

# Load an unknown face for testing
unknown_image = face_recognition.load_image_file("test.jpg")
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

# Check if a face was found in the unknown image
if len(unknown_face_encodings) > 0:
    unknown_face_encoding = unknown_face_encodings[0]
    
    # Compare the unknown face to the known faces
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

    if True in results:
        # Match found
        match_index = results.index(True)
        matched_name = known_names[match_index]
        print(f"Match found: {matched_name}")
    else:
        # No match found
        print("No match found")
else:
    # No face found in the unknown image
    print("No face found in the unknown image")
