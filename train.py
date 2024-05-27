from torch.utils.tensorboard import SummaryWriter
from models.model import mini_XCEPTION
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.fer2013 import FER2013

num_epochs = 200
num_workers = 0

# 定义模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mini_XCEPTION(num_classes=7)
model.to(device)

# transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(size=(48, 48), scale=(0.8, 1.2)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# 数据加载
train_dataset = FER2013(root='./data', split='train', transform=train_transforms)
test_dataset = FER2013(root='./data', split='test', transform=test_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

# data_size
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 添加 tensorboard
writer = SummaryWriter("logs")

# 简单的训练循环
for epoch in range(num_epochs):
    print("-----第 {} 轮训练开始-----".format(epoch + 1))

    # 训练步骤
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，计算损失函数的梯度
        optimizer.step()  # 根据梯度，对网络的参数进行调优

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(model.state_dict(), "./output/mini_xception_{}.pth".format(epoch))
    print("模型已保存")

writer.close()
