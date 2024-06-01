import sys

from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from models.model import mini_XCEPTION
from face_detection.face_detector import DNNDetector
import numpy as np

emotion_labels = {0: '愤怒', 1: '厌恶', 2: '恐惧', 3: '快乐', 4: '悲伤', 5: '惊讶', 6: '中立'}
root = 'face_detection'
detector = DNNDetector(root=root)

# 加载训练好的模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = 'output/mini_xception_85.pth'
model = mini_XCEPTION(num_classes=7).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # 设置为评估模式


# 图像预处理函数
def preprocess_image(image):
    """将cv2读取的图片预处理为模型可接受的输入形式"""
    # 转换颜色空间从BGR到RGB，然后转换为PIL Image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0)  # 添加一个批处理维度
    return image


# 绘制标签
def draw_emotion_label(cv_image, box, predicted_class):
    emotion_label = emotion_labels[predicted_class]
    print("检测结果：", emotion_label)
    x, y, w, h = box
    # opencv图片转换为PIL图片格式
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    # 使用PIL绘制标签
    draw = ImageDraw.Draw(pil_image)
    font_size = (w / 100) * 20
    font = ImageFont.truetype('simsun.ttc', font_size, encoding='utf-8')
    draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 0, 0), width=4)
    draw.text((x, y - font_size), emotion_label, fill=(255, 0, 0), font=font)
    # 将PIL图片转换回OpenCV格式
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image


# 预测
def predict_emotion(image):  # 输入cv的bgr图像
    detector.imread(image)
    faces = detector.detect_faces()

    with torch.no_grad():
        for face, box in faces:
            face = preprocess_image(face)
            face = face.to(device)
            outputs = model(face)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            image = draw_emotion_label(image, box, predicted_class)

    return image


class FaceExpressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 图像显示区域
        self.image_label = QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        # 上传图片按钮
        upload_button = QPushButton('上传图片', self)
        upload_button.clicked.connect(self.uploadImage)

        # 开始/停止摄像头按钮
        self.camera_button = QPushButton('开始摄像头', self)
        self.camera_button.clicked.connect(self.startCamera)

        # 布局
        hbox = QHBoxLayout()
        hbox.addWidget(upload_button)
        hbox.addWidget(self.camera_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.setWindowTitle('人脸表情识别')
        self.setGeometry(100, 100, 800, 600)

    def uploadImage(self):
        self.image_label.setText('识别中')
        # 打开文件对话框选择图片
        file_name, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image Files (*.png *.jpg *.jpeg)')
        if file_name:
            frame = cv2.imread(file_name)
            frame = self.predict_frame(frame)
            # 将cv图像转换为QT支持的格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio))

    def startCamera(self):
        if self.camera_button.text() == '开始摄像头':
            self.capture = cv2.VideoCapture(0)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.predict_video)
            self.timer.start(30)  # 每30ms更新一次
            self.camera_button.setText('停止摄像头')
        else:
            self.timer.stop()
            self.capture.release()
            self.camera_button.setText('开始摄像头')

    def predict_video(self):
        ret, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        if ret:
            frame = predict_emotion(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_Qt_format)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio))

    def predict_frame(self, frame):
        frame = predict_emotion(frame)
        return frame


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceExpressionApp()
    ex.show()
    sys.exit(app.exec_())
