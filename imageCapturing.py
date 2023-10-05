import cv2
import os
import numpy as np
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
save_path = "face_images"
if not os.path.exists(save_path):
    os.mkdir(save_path)
person_id = input("Enter the ID for the person you're collecting data for: ")
person_path = os.path.join(save_path, person_id)
if not os.path.exists(person_path):
    os.mkdir(person_path)
img_count = 0

while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 
            face_crop = frame[startY:endY, startX:endX]
            face_filename = os.path.join(person_path, f"{img_count}.jpg")
            cv2.imwrite(face_filename, face_crop)
            img_count += 1
    cv2.imshow("Image Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
