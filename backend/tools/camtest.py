"""Preview a webcam to find its index. Usage: python camtest.py [index]"""

import sys

import cv2

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

cap = cv2.VideoCapture(index, backend)
if not cap.isOpened():
    print(f"Cannot open camera {index}")
    sys.exit(1)

print(f"Camera {index} open - press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow(f"Camera {index}", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
