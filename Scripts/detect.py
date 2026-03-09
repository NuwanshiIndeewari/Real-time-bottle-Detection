import warnings
warnings.filterwarnings("ignore")

# detect_books.py
import cv2
import torch
from datetime import datetime
import os

# ------------------- CONFIG -------------------
ESP32_STREAM_URL = "http://172.20.10.2:81/stream"  # Replace with your ESP32-CAM IP
SAVE_PATH = "snapshots"
TARGET_CLASSES = ["book"]  # YOLOv5 class names you want to detect
# ----------------------------------------------

# Create folder for snapshots
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [model.names.index(cls) for cls in TARGET_CLASSES if cls in model.names]

# Open ESP32-CAM stream
cap = cv2.VideoCapture(ESP32_STREAM_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Cannot open stream. Check your URL or network!")
    exit()

print("Streaming from ESP32-CAM... Press 'q' to quit.")

while True:


    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        continue

    # Run detection
    results = model(frame)

    # Parse results
    detected = results.pandas().xyxy[0]  # pandas dataframe
    count = len(detected)
    
    # Draw bounding boxes
    for _, row in detected.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        conf = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f"Detected {count} {TARGET_CLASSES[0]}(s)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Book Detection", frame)

    # Save snapshot if something detected
    if count > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{SAVE_PATH}/snapshot_{timestamp}.jpg", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()