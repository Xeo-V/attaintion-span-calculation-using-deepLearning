import cv2
import time
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import csv

csv_file = open('attention_details.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Name", "Attention Span"])

face_model = load_model("face_recognition_model.h5")
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
video_stream = cv2.VideoCapture(0)
time.sleep(2.0)

attention_details = {}
labels_dict = {0: "1032200746", 1: "1032200703", 2: "1032200614"}

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def gaze_direction(eye):
    eye_center = ((eye[0] + eye[3]) / 2)
    direction = eye[0] - eye_center
    return direction

def head_pose(landmarks):
    nose = np.array([(landmarks.landmark[4].x, landmarks.landmark[4].y)])
    left_eye_inner = np.array([(landmarks.landmark[33].x, landmarks.landmark[33].y)])
    right_eye_inner = np.array([(landmarks.landmark[263].x, landmarks.landmark[263].y)])
    return nose, left_eye_inner, right_eye_inner

while True:
    ret, frame = video_stream.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.face_landmarks:
        left_eye_lm = np.array([(point.x, point.y) for point in results.face_landmarks.landmark[33:41]])
        right_eye_lm = np.array([(point.x, point.y) for point in results.face_landmarks.landmark[263:271]])
        
        left_ear = eye_aspect_ratio(left_eye_lm)
        right_ear = eye_aspect_ratio(right_eye_lm)
        average_ear = (left_ear + right_ear) / 2.0

        if average_ear > 0.3:
            attention_details["Open Eyes Time"] = attention_details.get("Open Eyes Time", 0) + 1
        else:
            attention_details["Closed Eyes Time"] = attention_details.get("Closed Eyes Time", 0) + 1

        left_gaze = gaze_direction(left_eye_lm)
        right_gaze = gaze_direction(right_eye_lm)
        
        nose, left_eye_inner, right_eye_inner = head_pose(results.face_landmarks)
        pose_offset = np.linalg.norm(nose - (left_eye_inner + right_eye_inner) / 2)
        
        if np.linalg.norm(left_gaze) > 0.05 or np.linalg.norm(right_gaze) > 0.05 or pose_offset > 0.1:
            attention_details["Distracted Time"] = attention_details.get("Distracted Time", 0) + 1

        open_eyes_time = attention_details.get("Open Eyes Time", 0)
        closed_eyes_time = attention_details.get("Closed Eyes Time", 0)
        distracted_time = attention_details.get("Distracted Time", 0)

        if open_eyes_time + closed_eyes_time + distracted_time == 0:
            attention_span = 0
        else:
            attention_span = (open_eyes_time - distracted_time) / (open_eyes_time + closed_eyes_time + distracted_time) * 100

        cv2.putText(image, f"Attention: {attention_span:.2f}%", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

csv_writer.writerow(["Attention Span", attention_span])
cv2.destroyAllWindows()
video_stream.release()
csv_file.close()
