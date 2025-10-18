import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

attendance_file = 'Attendance.csv'

# ===== Step 1: Load Images =====
path = 'dataset'
images = []
classNames = []

print("🔍 Loading student faces from dataset...")

for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            file_path = os.path.join(root, file)
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                student_name = os.path.basename(root)
                classNames.append(student_name)
            else:
                print(f"⚠️ Could not read image: {file_path}")

print(f"Loaded {len(images)} images for {len(set(classNames))} students.")

# ===== Step 2: Encode Faces =====
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print("⚠️ No face found in an image, skipping...")
    return encodeList

print("⚙️ Encoding faces, please wait...")
encodeListKnown = findEncodings(images)
print("✅ Encoding complete.")

# ===== Step 3: Mark Attendance =====
def markAttendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(attendance_file, index=False)

    df = pd.read_csv(attendance_file)

    if ((df["Name"] == name) & (df["Date"] == today)).any():
        print(f"⚠️ {name} already marked today.")
        return False
    else:
        new_row = pd.DataFrame([[name, today, time_now]], columns=["Name", "Date", "Time"])
        new_row.to_csv(attendance_file, mode='a', header=False, index=False)
        print(f"✅ Attendance marked for {name}")
        return True

# ===== Step 4: Start Camera =====
cap = cv2.VideoCapture(0)
print("📷 Camera started. Looking for faces...")

attendance_checked = False  # stops after 1 detection

while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Failed to access camera.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)
            attendance_checked = True
            break

    cv2.imshow('Smart Attendance System', img)
    cv2.waitKey(1)

    if attendance_checked:  # stop after first detection
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Attendance session ended automatically.")
