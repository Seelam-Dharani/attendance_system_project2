import cv2
import os

dataset_path = 'dataset'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

student_name = input("Enter student name: ").strip()
person_path = os.path.join(dataset_path, student_name)

if not os.path.exists(person_path):
    os.makedirs(person_path)
else:
    print("Folder already exists for this student.")

cap = cv2.VideoCapture(0)
count = 0

print(f"Capturing 30 images for {student_name}. Press 'q' to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Capturing Faces', frame)

    # Save one image per few frames
    if count < 30:
        file_path = os.path.join(person_path, f"{student_name}_{count+1}.jpg")
        cv2.imwrite(file_path, frame)
        count += 1
        print(f"Saved image {count}/30")

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Finished capturing 30 images for {student_name}")
