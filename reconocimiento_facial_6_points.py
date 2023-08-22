import cv2
import mediapipe as mp
from mediapipe.python.solutions import face_detection, drawing_utils

mp_face_detection = face_detection
mp_drawing = drawing_utils

cap = cv2.VideoCapture(0)
print("cargando imagen...")
with mp_face_detection.FaceDetection(
    model_selection = 0,
    min_detection_confidence= 0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
        cv2.imshow("MediaPipe Face Detection", cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()


