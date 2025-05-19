import imutils
import time
import cv2
import csv
import os

# Load Haar Cascade for face detection
cascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)

# User input
Name = str(input("Enter your Name : "))
Roll_Number = int(input("Enter your Roll_Number : "))
dataset = 'dataset'
sub_data = Name
path = os.path.join(dataset, sub_data)

# Ensure folders exist
if not os.path.exists(dataset):
    os.mkdir(dataset)

if not os.path.exists(path):
    os.mkdir(path)

# Write student info to CSV
info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(info)

print("Starting video stream...")
cam = cv2.VideoCapture(0)  # You can change to 0 if 1 doesn't work
time.sleep(2.0)
total = 0

# Capture face images
while total < 100:
    ret, frame = cam.read()
    if not ret or frame is None:
        print("Failed to grab frame.")
        break

    img = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_path = os.path.join(path, f"{str(total).zfill(5)}.png")
        cv2.imwrite(img_path, img)
        total += 1

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
