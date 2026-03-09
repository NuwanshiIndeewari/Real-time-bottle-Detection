import cv2

url = "http://172.20.10.2:81/stream"   # replace with your IP

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to get frame")
        break

    cv2.imshow("ESP32 Stream", frame)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()