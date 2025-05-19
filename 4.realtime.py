from collections.abc import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv

# Helper function to flatten nested lists
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

# File paths and settings
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5
student_csv = 'student.csv'

# Load face detector
print("[INFO] Loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Load face embedder
print("[INFO] Loading face embedder...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Load recognizer and label encoder
print("[INFO] Loading face recognizer and label encoder...")
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Start video stream
print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

recognized_set = set()  # Keep track of printed names

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Create embedding from face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Predict
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Search name in CSV
            Roll_Number = "Unknown"
            with open(student_csv, 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    if name in row:
                        flat_list = list(flatten(row))
                        if name in flat_list:
                            idx = flat_list.index(name)
                            name = flat_list[idx]
                            Roll_Number = flat_list[idx + 1]
                            break

            text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)

            # Show on frame
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # Print once to terminal
            if name not in recognized_set:
                print(f"[INFO] Recognized: Name: {name}, Roll Number: {Roll_Number}")
                recognized_set.add(name)

    # Show the output frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to break
        break

cam.release()
cv2.destroyAllWindows()
