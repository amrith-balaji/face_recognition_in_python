import face_recognition
import cv2

training_images = ["(replace with image name).png"]

known_encodings = []
known_names = []

for image_path in training_images:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)
    known_names.append(image_path[:-4])  


face_locations = []
face_encodings = []
face_names = []

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]


    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
      
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_index = matches.index(True)
            name = known_names[matched_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 1
        right *= 1
        bottom *= 1
        left *= 1


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
