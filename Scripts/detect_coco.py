import torch
import cv2
import time

# -----------------------------
# Load YOLOv5 model
# -----------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # small model for CPU
model.conf = 0.4  # confidence threshold (optional)

# -----------------------------
# ESP32 Stream URL
# -----------------------------
stream_url = "http://10.55.42.168:81/stream"  # Update with your ESP32 IP

# -----------------------------
# OpenCV Video Capture
# -----------------------------
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Cannot open ESP32 stream. Check IP/port or Wi-Fi connection.")
    exit()

# -----------------------------
# Resizable window
# -----------------------------
cv2.namedWindow("ESP32 Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ESP32 Object Detection", 1280, 720)

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame not received. Retrying in 0.5s...")
        time.sleep(0.5)
        continue

    # -----------------------------
    # Resize frame for YOLO (faster)
    # -----------------------------
    frame_resized = cv2.resize(frame, (640, 360))

    # -----------------------------
    # Run YOLOv5 detection
    # -----------------------------
    results = model(frame_resized)

    # Convert results to pandas DataFrame
    detections = results.pandas().xyxy[0]

    # -----------------------------
    # Count bottles and draw boxes manually (faster than results.render)
    # -----------------------------
    bottle_count = 0
    for _, row in detections.iterrows():
        class_id = int(row['class'])
        if class_id == 39:  # Bottle class
            bottle_count += 1
            x1, y1 = int(row['xmin']), int(row['ymin'])
            x2, y2 = int(row['xmax']), int(row['ymax'])
            # Draw bounding box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame_resized,
                        f"Bottle ID: {class_id}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    # Display total bottle count
    cv2.putText(frame_resized,
                f"Bottles detected: {bottle_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    # -----------------------------
    # Show frame
    # -----------------------------
    cv2.imshow("ESP32 Object Detection", frame_resized)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Release resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()