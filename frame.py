import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
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


def draw_emotion_label(cv_image, box, predicted_class):
    emotion_label = emotion_labels[predicted_class]
    x, y, w, h = box
    # opencv图片转换为PIL图片格式
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    # 使用PIL绘制标签
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype('simsun.ttc', 20, encoding='utf-8')
    draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 0, 0), width=2)
    draw.text((x, y - 20), emotion_label, fill=(255, 0, 0), font=font)
    # 将PIL图片转换回OpenCV格式
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image


# 使用OpenCV读取并预处理图片
raw_image_path = 'images/happy1.jpg'
raw_image = cv2.imread(raw_image_path)
detector.imread(raw_image)
faces = detector.detect_faces()

with torch.no_grad():
    for face, box in faces:
        face = preprocess_image(face)
        face = face.to(device)
        outputs = model(face)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        raw_image = draw_emotion_label(raw_image, box, predicted_class)

cv2.imshow("frame", raw_image)
c = cv2.waitKey(0)
cv2.destroyAllWindows()
