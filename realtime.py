"""Live webcam attendance (local PC only).  Run: python realtime.py  | press 'q' to quit.
Needs the GUI build of OpenCV:  pip install opencv-python
"""
import cv2

from face_engine import FaceEngine, mark_attendance

engine = FaceEngine()
cap = cv2.VideoCapture(0)
seen = set()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    results = engine.recognize(frame)
    for r in results:
        if r["name"] != "Unknown" and r["name"] not in seen:
            seen.add(r["name"])
            if mark_attendance(r["name"]):
                print(f"✅ {r['name']} marked present")
    cv2.imshow("Face Attendance - press q to quit", engine.annotate(frame, results))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
