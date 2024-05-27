from face_detection.face_detector import DNNDetector, MTCNNDetector, HaarCascadeDetector
import cv2

capture = cv2.VideoCapture(0)
root = 'face_detection'
detector = DNNDetector(root=root)

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    detector.imread(frame)
    faces = detector.detect_faces()
    for face, box in faces:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imshow("video", frame)
    c = cv2.waitKey(50)
    if c == 27:  # ese退出
        break

cv2.destroyAllWindows()
