import os

import cv2
import numpy as np


class HaarCascadeDetector:
    def __init__(self, img=None, root=None):
        self.haarcascade = 'haarcascade_frontalface_default.xml'
        self.raw_img = img

        if root:
            self.detector = os.path.join(root, self.haarcascade)

        self.detector = cv2.CascadeClassifier(self.haarcascade)

    def imread(self, frame):
        if isinstance(frame, np.ndarray):
            self.raw_img = frame
        else:
            self.raw_img = cv2.imread(frame)

    def detect_faces(self):
        faces = []
        # result_img = self.raw_img.copy()
        gray_img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        result = self.detector.detectMultiScale(gray_img, 1.3, 5)
        for x, y, w, h in result:
            box = (x, y, w, h)
            faces.append((self.raw_img[y:y + h, x:x + w], box))
            # cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        return faces


class MTCNNDetector:
    """
    Class for face detection using MTCNN
    @:param img : BGR image
    """

    def __init__(self, img=None):
        from mtcnn import MTCNN
        self.detector = MTCNN()
        self.raw_img = img

    def imread(self, frame):
        if isinstance(frame, np.ndarray):
            self.raw_img = frame
        else:
            self.raw_img = cv2.imread(frame)

    def detect_faces(self):
        faces = []
        # result_img = self.raw_img.copy()
        img = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_faces(img)
        for face in result:
            x, y, w, h = face['box']
            faces.append((self.raw_img[y:y + h, x:x + w], (x, y, w, h)))
            # cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return faces


class DNNDetector:
    """
    SSD (Single Shot Detectors) based face detection (ResNet-18 backbone(light feature extractor))
    """

    def __init__(self, img=None, root=None):
        self.raw_img = img
        self.prototxt = "deploy.prototxt.txt"
        self.model_weights = "res10_300x300_ssd_iter_140000.caffemodel"

        if root:
            self.prototxt = os.path.join(root, self.prototxt)
            self.model_weights = os.path.join(root, self.model_weights)

        self.detector = cv2.dnn.readNetFromCaffe(prototxt=self.prototxt, caffeModel=self.model_weights)
        self.threshold = 0.5  # to remove weak detections

    def imread(self, frame):
        if isinstance(frame, np.ndarray):
            self.raw_img = frame
        else:
            self.raw_img = cv2.imread(frame)

    def detect_faces(self):
        h = self.raw_img.shape[0]
        w = self.raw_img.shape[1]
        # required preprocessing(mean & variance(scale) & size) to use the dnn model
        """
            Problem of not detecting small faces if the image is big (720p or 1080p)
            because we resize to 300,300 ... but if we use the original size it will detect right but so slow
        """
        resized_frame = cv2.resize(self.raw_img, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, resized_frame.shape[0:2], (104.0, 177.0, 123.0))
        # detect
        self.detector.setInput(blob)
        detections = self.detector.forward()
        faces = []

        # result_img = self.raw_img.copy()

        # shape 2 is num of detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.threshold:
                continue

            # model output is percentage of bbox dims
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype("int")
            (x1, y1, x2, y2) = box
            faces.append((self.raw_img[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)))
            # cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return faces


# if __name__ == '__main__':
#     detector = DNNDetector()
#     detector.imread('../images/test.jpg')
#     faces = detector.detect_faces()
#     for face, box in faces:
#         x, y, w, h = box
#         cv2.rectangle(detector.raw_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
#         # cv2.putText(detector.raw_img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#     cv2.imshow('result', detector.raw_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
