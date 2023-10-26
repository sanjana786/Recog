import face_recognition
import os

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

# Load an unknown face for matching
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
